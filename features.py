"""
Feature extraction and preprocessing pipeline.

Produces two dataset representations:

1. Static Snapshot Representation
   Game state at a fixed time window (default: first 10 s of round).
   Output: flattened feature vector per round → X_static.npy, y_static.npy

2. Temporal Sequence Representation
   Sequence of state snapshots sampled at [5 s, 10 s, 15 s, 20 s, 25 s, 30 s].
   Output: (N, T, F) array per round → X_seq.npy, y_seq.npy
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path("valorant.db")
OUT_DIR = Path("features")

# == Time configuration ========================================================
MAX_ROUND_MS = 100_000
DEFAULT_SNAP = 10_000
SEQ_STEPS_MS = [5_000, 10_000, 15_000, 20_000, 25_000, 30_000]

# == Agent role taxonomy =======================================================
AGENT_ROLES: dict[str, str] = {
    "Astra": "Controller", "Brimstone": "Controller", "Harbor": "Controller", "Miks": "Controller",
    "Omen": "Controller", "Viper": "Controller", "Clove": "Controller", 

    "Jett": "Duelist", "Neon": "Duelist", "Phoenix": "Duelist", "Raze": "Duelist",
    "Reyna": "Duelist", "Yoru": "Duelist", "Iso": "Duelist", "Waylay": "Duelist",

    "Breach": "Initiator", "Fade": "Initiator", "Gekko": "Initiator",
    "KAY/O": "Initiator", "KAYO": "Initiator", "Skye": "Initiator", "Sova": "Initiator",
    "Tejo": "Initiator",

    "Chamber": "Sentinel", "Cypher": "Sentinel", "Deadlock": "Sentinel", "Veto": "Sentinel",
    "Killjoy": "Sentinel", "Sage": "Sentinel", "Vyse": "Sentinel",

    "Unknown": "Unknown",
}
ROLE_ORDER = ["Controller", "Duelist", "Initiator", "Sentinel", "Unknown"]
N_ROLES = len(ROLE_ORDER)
ROLE_IDX = {r: i for i, r in enumerate(ROLE_ORDER)}

LABEL_NAMES = {
    0: "A-Site Rush",
    1: "B-Site Rush",
    2: "A Split",
    3: "B Split",
    4: "Default",
    5: "Slow Execute",
}

# == Feature layout ============================================================
# Previous 36 + 4 movement features = 40
FEATURE_DIM = 48

# [0]     timestamp_fraction
# [1]     a_ratio
# [2]     b_ratio
# [3]     mid_ratio
# [4]     total_kills_norm
# [5]     site_entropy
# [6]     kill_rate
# [7]     first_kill_time_norm
# [8..10] kill momentum windows
# [11..12] reserved
# [13..17] attacker role composition
# [18]    duelist_kill_fraction
# [19]    any_A_pressure
# [20]    any_B_pressure
# [21]    plant_is_A
# [22]    plant_is_B
# [23]    has_plant
# [24]    plant_time_norm
# [25]    late_plant_flag
# [26]    plant_x_norm
# [27]    plant_y_norm
# [28]    attacker_A_fraction
# [29]    attacker_B_fraction
# [30]    attacker_MID_fraction
# [31]    attacker_spread_mean_norm
# [32]    attacker_spread_max_norm
# [33]    attackers_alive_norm
# [34]    defenders_alive_norm
# [35]    alive_advantage_norm
# [36]    attacker_avg_speed_norm
# [37]    attacker_max_speed_norm
# [38]    toward_A_fraction
# [39]    toward_B_fraction
# [40]    pred_attacker_A_fraction
# [41]    pred_attacker_B_fraction
# [42]    pred_attacker_MID_fraction
# [43]    pred_attacker_spread_mean_norm
# [44]    pred_attacker_spread_max_norm
# [45]    pred_commitment_strength
# [46]    pred_mean_confidence
# [47]    pred_available_flag

def get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}

def _infer_attack_team_from_round_num(round_num: int) -> str:
    """
    Match convention:
    rounds 0-11   -> Red attacks
    rounds 12+    -> Blue attacks
    """
    return "Red" if int(round_num) <= 11 else "Blue"

def _site_norm(s: str) -> str:
    s = str(s).strip().upper()
    if s in ("A", "A SITE", "A-SITE", "ALPHA"):
        return "A"
    if s in ("B", "B SITE", "B-SITE", "BRAVO"):
        return "B"
    if s in ("MID", "M", "MIDDLE"):
        return "MID"
    return "MID"


def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _safe_norm_coord(val, denom: float = 20000.0) -> float:
    if val is None or pd.isna(val):
        return 0.0
    return float(val) / denom


def _compute_site_centroids_from_kills(df_kills: pd.DataFrame) -> dict[str, dict[str, tuple[float, float]]]:
    centroids: dict[str, dict[str, tuple[float, float]]] = {}

    usable = df_kills.copy()
    usable["site_norm"] = usable["site"].apply(_site_norm)

    usable = usable[
        usable["site_norm"].isin(["A", "B"])
        & usable["victim_location_x"].notna()
        & usable["victim_location_y"].notna()
    ]

    if usable.empty:
        return centroids

    for (map_name, site), grp in usable.groupby(["map", "site_norm"]):
        cx = float(grp["victim_location_x"].mean())
        cy = float(grp["victim_location_y"].mean())
        centroids.setdefault(map_name, {})[site] = (cx, cy)

    return centroids


def _infer_site_from_coords(
    x: float | None,
    y: float | None,
    map_name: str,
    site_centroids: dict[str, dict[str, tuple[float, float]]],
) -> str:
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return "MID"

    cmap = site_centroids.get(map_name, {})
    if "A" not in cmap or "B" not in cmap:
        return "MID"

    ax, ay = cmap["A"]
    bx, by = cmap["B"]

    d_a = float(np.hypot(x - ax, y - ay))
    d_b = float(np.hypot(x - bx, y - by))

    mid_x = (ax + bx) / 2.0
    mid_y = (ay + by) / 2.0
    d_mid = float(np.hypot(x - mid_x, y - mid_y))

    if d_mid < 1.00 * min(d_a, d_b):
        return "MID"
    if d_a < d_b:
        return "A"
    return "B"


def _latest_player_snapshot(loc_round: pd.DataFrame, t_ms: int) -> pd.DataFrame:
    if loc_round.empty:
        return pd.DataFrame(columns=loc_round.columns)

    snap = loc_round[loc_round["timestamp_ms"] <= t_ms].copy()
    if snap.empty:
        return snap

    snap = snap.sort_values(["player_name", "timestamp_ms"])
    snap = snap.groupby("player_name", as_index=False).tail(1)
    return snap

def _prediction_bucket_for_t(t_ms: int) -> str | None:
    if t_ms <= 15_000:
        return "0-15"
    if t_ms <= 30_000:
        return "15-30"
    return "15-30"


def _load_prediction_rows(conn: sqlite3.Connection) -> pd.DataFrame:
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "player_predictions" not in tables:
        return pd.DataFrame(
            columns=[
                "round_id", "match_id", "round_num", "player_name", "team",
                "time_bucket", "predicted_x", "predicted_y", "confidence"
            ]
        )

    query = """
        SELECT
            round_id,
            match_id,
            round_num,
            player_name,
            team,
            time_bucket,
            predicted_x,
            predicted_y,
            confidence
        FROM player_predictions
    """
    return pd.read_sql_query(query, conn)


def _prediction_snapshot(
    pred_round: pd.DataFrame,
    t_ms: int,
) -> pd.DataFrame:
    if pred_round.empty:
        return pred_round

    bucket = _prediction_bucket_for_t(t_ms)
    if bucket is None:
        return pd.DataFrame(columns=pred_round.columns)

    snap = pred_round[pred_round["time_bucket"] == bucket].copy()
    if snap.empty:
        return snap

    snap = snap.sort_values(["player_name", "confidence"], ascending=[True, False])
    snap = snap.groupby("player_name", as_index=False).head(1)
    return snap


def _compute_prediction_presence_features(
    pred_snapshot: pd.DataFrame,
    map_name: str,
    attack_team: str,
    site_centroids: dict[str, dict[str, tuple[float, float]]],
) -> tuple[float, float, float, float]:
    if pred_snapshot.empty:
        return 0.0, 0.0, 0.0, 0.0

    attackers = pred_snapshot[pred_snapshot["team"] == attack_team].copy()
    if attackers.empty:
        return 0.0, 0.0, 0.0, 0.0

    attackers["pred_site"] = attackers.apply(
        lambda r: _infer_site_from_coords(
            r["predicted_x"], r["predicted_y"], map_name, site_centroids
        ),
        axis=1,
    )

    counts = attackers["pred_site"].value_counts()
    total = len(attackers)

    a_frac = float(counts.get("A", 0) / total)
    b_frac = float(counts.get("B", 0) / total)
    m_frac = float(counts.get("MID", 0) / total)
    commitment = max(a_frac, b_frac)

    return a_frac, b_frac, m_frac, commitment


def _compute_prediction_spread_features(
    pred_snapshot: pd.DataFrame,
    attack_team: str,
    norm_denom: float = 20000.0,
) -> tuple[float, float, float]:
    if pred_snapshot.empty:
        return 0.0, 0.0, 0.0

    attackers = pred_snapshot[
        (pred_snapshot["team"] == attack_team)
        & pred_snapshot["predicted_x"].notna()
        & pred_snapshot["predicted_y"].notna()
    ][["predicted_x", "predicted_y"]].to_numpy(dtype=np.float32)

    if len(attackers) == 0:
        return 0.0, 0.0, 0.0

    if len(attackers) == 1:
        return 0.0, 0.0, 1.0

    centroid = attackers.mean(axis=0)
    dists = np.linalg.norm(attackers - centroid, axis=1)
    mean_spread = float(dists.mean()) / norm_denom

    max_pairwise = 0.0
    for i in range(len(attackers)):
        for j in range(i + 1, len(attackers)):
            d = float(np.linalg.norm(attackers[i] - attackers[j]))
            if d > max_pairwise:
                max_pairwise = d

    max_spread = max_pairwise / norm_denom
    mean_conf = float(
        pred_snapshot.loc[pred_snapshot["team"] == attack_team, "confidence"].mean()
    ) if "confidence" in pred_snapshot.columns else 0.0

    return mean_spread, max_spread, mean_conf

def _compute_presence_features(
    player_snapshot: pd.DataFrame,
    map_name: str,
    attack_team: str,
    site_centroids: dict[str, dict[str, tuple[float, float]]],
) -> tuple[float, float, float]:
    if player_snapshot.empty:
        return 0.0, 0.0, 0.0

    attackers = player_snapshot[player_snapshot["team"] == attack_team].copy()
    if attackers.empty:
        return 0.0, 0.0, 0.0

    attackers["pos_site"] = attackers.apply(
        lambda r: _infer_site_from_coords(r["location_x"], r["location_y"], map_name, site_centroids),
        axis=1
    )

    counts = attackers["pos_site"].value_counts()
    total = len(attackers)

    a_frac = float(counts.get("A", 0) / total)
    b_frac = float(counts.get("B", 0) / total)
    m_frac = float(counts.get("MID", 0) / total)
    return a_frac, b_frac, m_frac


def _compute_spread_features(
    player_snapshot: pd.DataFrame,
    attack_team: str,
    norm_denom: float = 20000.0,
) -> tuple[float, float]:
    if player_snapshot.empty:
        return 0.0, 0.0

    attackers = player_snapshot[
        (player_snapshot["team"] == attack_team)
        & player_snapshot["location_x"].notna()
        & player_snapshot["location_y"].notna()
    ][["location_x", "location_y"]].to_numpy(dtype=np.float32)

    if len(attackers) <= 1:
        return 0.0, 0.0

    centroid = attackers.mean(axis=0)
    dists_to_centroid = np.linalg.norm(attackers - centroid, axis=1)
    mean_spread = float(dists_to_centroid.mean()) / norm_denom

    max_pairwise = 0.0
    for i in range(len(attackers)):
        for j in range(i + 1, len(attackers)):
            d = float(np.linalg.norm(attackers[i] - attackers[j]))
            if d > max_pairwise:
                max_pairwise = d
    max_spread = max_pairwise / norm_denom

    return mean_spread, max_spread


def _compute_alive_features(
    kills_t: pd.DataFrame,
    all_players_round: pd.DataFrame,
    attack_team: str,
) -> tuple[float, float, float]:
    if all_players_round.empty:
        return 0.0, 0.0, 0.0

    all_players_round = all_players_round.drop_duplicates(subset=["player_name"])

    attackers_start = int((all_players_round["team"] == attack_team).sum())
    defenders_start = int((all_players_round["team"] != attack_team).sum())

    if attackers_start == 0 and defenders_start == 0:
        return 0.0, 0.0, 0.0

    victim_team_map = all_players_round.set_index("player_name")["team"].to_dict()

    att_dead = 0
    def_dead = 0

    if not kills_t.empty:
        dead_names = kills_t["victim"].dropna().astype(str).unique().tolist()
        for victim in dead_names:
            team = victim_team_map.get(victim)
            if team == attack_team:
                att_dead += 1
            elif team is not None:
                def_dead += 1

    attackers_alive = max(0, attackers_start - att_dead)
    defenders_alive = max(0, defenders_start - def_dead)

    attackers_alive_norm = attackers_alive / max(1, attackers_start)
    defenders_alive_norm = defenders_alive / max(1, defenders_start)
    alive_advantage_norm = (attackers_alive - defenders_alive) / max(1, attackers_start + defenders_start)

    return attackers_alive_norm, defenders_alive_norm, alive_advantage_norm


def _unit_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        return np.zeros_like(vec)
    return vec / norm


def _compute_movement_features(
    loc_round: pd.DataFrame,
    t_ms: int,
    attack_team: str,
    map_name: str,
    site_centroids: dict[str, dict[str, tuple[float, float]]],
    norm_speed: float = 1500.0,
) -> tuple[float, float, float, float]:
    """
    Movement features from the last two known positions per attacker:
    - avg speed
    - max speed
    - fraction moving toward A
    - fraction moving toward B
    """
    if loc_round.empty:
        return 0.0, 0.0, 0.0, 0.0

    hist = loc_round[
        (loc_round["timestamp_ms"] <= t_ms)
        & (loc_round["team"] == attack_team)
        & loc_round["location_x"].notna()
        & loc_round["location_y"].notna()
    ].copy()

    if hist.empty:
        return 0.0, 0.0, 0.0, 0.0

    cmap = site_centroids.get(map_name, {})
    if "A" not in cmap or "B" not in cmap:
        return 0.0, 0.0, 0.0, 0.0

    a_cent = np.array(cmap["A"], dtype=np.float32)
    b_cent = np.array(cmap["B"], dtype=np.float32)

    speeds = []
    toward_a = 0
    toward_b = 0
    movers = 0

    for _, grp in hist.sort_values("timestamp_ms").groupby("player_name"):
        if len(grp) < 2:
            continue

        last2 = grp.tail(2)
        p0 = last2.iloc[0]
        p1 = last2.iloc[1]

        dt_ms = float(p1["timestamp_ms"] - p0["timestamp_ms"])
        if dt_ms <= 0:
            continue

        pos0 = np.array([p0["location_x"], p0["location_y"]], dtype=np.float32)
        pos1 = np.array([p1["location_x"], p1["location_y"]], dtype=np.float32)

        move_vec = pos1 - pos0
        dist = float(np.linalg.norm(move_vec))
        if dist < 1e-6:
            continue

        speed = dist / (dt_ms / 1000.0)
        speeds.append(speed)
        movers += 1

        move_dir = _unit_vector(move_vec)
        dir_to_a = _unit_vector(a_cent - pos1)
        dir_to_b = _unit_vector(b_cent - pos1)

        score_a = float(np.dot(move_dir, dir_to_a))
        score_b = float(np.dot(move_dir, dir_to_b))

        if score_a > score_b and score_a > 0.15:
            toward_a += 1
        elif score_b > score_a and score_b > 0.15:
            toward_b += 1

    if not speeds:
        return 0.0, 0.0, 0.0, 0.0

    avg_speed = float(np.mean(speeds)) / norm_speed
    max_speed = float(np.max(speeds)) / norm_speed
    toward_a_frac = toward_a / max(1, movers)
    toward_b_frac = toward_b / max(1, movers)

    return avg_speed, max_speed, toward_a_frac, toward_b_frac


def _extract_snapshot(
    kills_t: pd.DataFrame,
    t_ms: int,
    all_kills: pd.DataFrame,
    agents_series: pd.Series,
    round_meta: dict,
    player_snapshot: pd.DataFrame,
    all_players_round: pd.DataFrame,
    loc_round: pd.DataFrame,
    pred_round: pd.DataFrame,
    site_centroids: dict[str, dict[str, tuple[float, float]]],
) -> np.ndarray:
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    map_name = str(round_meta.get("map", ""))
    attack_team = str(round_meta.get("attack_team", "Red"))

    # [0] snapshot time
    vec[0] = t_ms / MAX_ROUND_MS

    # Kill-site distribution up to snapshot
    if not kills_t.empty:
        sites = kills_t["site_norm"].value_counts()
        a_cnt = int(sites.get("A", 0))
        b_cnt = int(sites.get("B", 0))
        m_cnt = int(sites.get("MID", 0))
        total = a_cnt + b_cnt + m_cnt

        vec[1] = a_cnt / max(1, total)
        vec[2] = b_cnt / max(1, total)
        vec[3] = m_cnt / max(1, total)
        vec[4] = total / 10.0
        vec[5] = _entropy(np.array([a_cnt, b_cnt, m_cnt]))

        t_sec = max(t_ms / 1000.0, 1.0)
        vec[6] = total / t_sec
    else:
        vec[1:7] = 0.0

    # [7] first kill time normalized
    if not all_kills.empty:
        fk_ms = all_kills["timestamp_ms"].min()
        vec[7] = min(float(fk_ms), MAX_ROUND_MS) / MAX_ROUND_MS
    else:
        vec[7] = 1.0

    # [8..10] kill momentum windows
    windows = [(0, 10_000), (10_000, 20_000), (20_000, 30_000)]
    for i, (lo, hi) in enumerate(windows):
        if not kills_t.empty:
            cnt = ((kills_t["timestamp_ms"] >= lo) & (kills_t["timestamp_ms"] < hi)).sum()
        else:
            cnt = 0
        vec[8 + i] = cnt / 5.0

    vec[11] = 0.0
    vec[12] = 0.0

    # [13..17] attacker role composition
    role_counts = np.zeros(N_ROLES, dtype=np.float32)
    for agent in agents_series:
        role = AGENT_ROLES.get(str(agent), "Unknown")
        role_counts[ROLE_IDX[role]] += 1
    n_agents = role_counts.sum()
    if n_agents > 0:
        vec[13:13 + N_ROLES] = role_counts / n_agents

    # [18] fraction of kills by duelists
    if not kills_t.empty:
        killer_roles = kills_t["killer_agent"].apply(lambda a: AGENT_ROLES.get(str(a), "Unknown"))
        duelist_kills = (killer_roles == "Duelist").sum()
        vec[18] = duelist_kills / max(1, len(kills_t))
    else:
        vec[18] = 0.0

    vec[19] = 1.0 if vec[1] > 0 else 0.0
    vec[20] = 1.0 if vec[2] > 0 else 0.0

    # Plant-aware features
    plant_site = str(round_meta.get("plant_site") or "").strip().upper()
    plant_x = round_meta.get("plant_x")
    plant_y = round_meta.get("plant_y")
    plant_time_ms = round_meta.get("plant_time_ms")

    vec[21] = 1.0 if plant_site == "A" else 0.0
    vec[22] = 1.0 if plant_site == "B" else 0.0
    vec[23] = 1.0 if plant_site in ("A", "B") else 0.0

    if plant_time_ms is not None and not pd.isna(plant_time_ms):
        vec[24] = min(float(plant_time_ms), MAX_ROUND_MS) / MAX_ROUND_MS
    else:
        vec[24] = 1.0

    if plant_time_ms is not None and not pd.isna(plant_time_ms):
        vec[25] = 1.0 if float(plant_time_ms) >= 30_000 else 0.0
    else:
        vec[25] = 0.0

    vec[26] = _safe_norm_coord(plant_x)
    vec[27] = _safe_norm_coord(plant_y)

    # Positional presence
    a_frac, b_frac, mid_frac = _compute_presence_features(
        player_snapshot=player_snapshot,
        map_name=map_name,
        attack_team=attack_team,
        site_centroids=site_centroids,
    )
    vec[28] = a_frac
    vec[29] = b_frac
    vec[30] = mid_frac

    # Spread / compactness
    spread_mean, spread_max = _compute_spread_features(
        player_snapshot=player_snapshot,
        attack_team=attack_team,
    )
    vec[31] = spread_mean
    vec[32] = spread_max

    # Alive counts
    attackers_alive_norm, defenders_alive_norm, alive_advantage_norm = _compute_alive_features(
        kills_t=kills_t,
        all_players_round=all_players_round,
        attack_team=attack_team,
    )
    vec[33] = attackers_alive_norm
    vec[34] = defenders_alive_norm
    vec[35] = alive_advantage_norm

    # Movement / velocity
    avg_speed, max_speed, toward_a_frac, toward_b_frac = _compute_movement_features(
        loc_round=loc_round,
        t_ms=t_ms,
        attack_team=attack_team,
        map_name=map_name,
        site_centroids=site_centroids,
    )
    vec[36] = avg_speed
    vec[37] = max_speed
    vec[38] = toward_a_frac
    vec[39] = toward_b_frac

    # Prediction-derived attacker features
    pred_snapshot = _prediction_snapshot(pred_round, t_ms)

    pred_a_frac, pred_b_frac, pred_mid_frac, pred_commitment = _compute_prediction_presence_features(
        pred_snapshot=pred_snapshot,
        map_name=map_name,
        attack_team=attack_team,
        site_centroids=site_centroids,
    )
    pred_spread_mean, pred_spread_max, pred_mean_conf = _compute_prediction_spread_features(
        pred_snapshot=pred_snapshot,
        attack_team=attack_team,
    )

    vec[40] = pred_a_frac
    vec[41] = pred_b_frac
    vec[42] = pred_mid_frac
    vec[43] = pred_spread_mean
    vec[44] = pred_spread_max
    vec[45] = pred_commitment
    vec[46] = pred_mean_conf
    vec[47] = 1.0 if not pred_snapshot.empty else 0.0

    return vec


def build_datasets(
    db_path: Path = DB_PATH,
    snapshot_ms: int = DEFAULT_SNAP,
    seq_steps: list[int] | None = None
) -> dict:
    if seq_steps is None:
        seq_steps = SEQ_STEPS_MS

    conn = sqlite3.connect(db_path)

    # Labels
    labels_sql = "SELECT round_id, label, label_name FROM round_labels"
    try:
        df_labels = pd.read_sql_query(labels_sql, conn)
    except Exception:
        conn.close()
        raise RuntimeError("round_labels table not found. Run: python label_rounds.py --update-db")

    df_labels["label"] = df_labels["label"].astype(int)
    df_labels["label_name"] = df_labels["label"].map(LABEL_NAMES)

    label_map = df_labels.set_index("round_id")["label"].to_dict()

    # Kills
    kills_sql = """
        SELECT
            k.kill_id,
            k.round_id,
            k.match_id,
            r.side,
            m.map,
            COALESCE(k.timestamp_ms, CAST(k.timestamp_sec * 1000 AS INTEGER), 0) AS timestamp_ms,
            COALESCE(k.site, 'MID') AS site,
            k.killer,
            k.victim,
            k.killer_agent,
            k.victim_agent,
            k.victim_location_x,
            k.victim_location_y
        FROM kills k
        JOIN rounds r ON r.round_id = k.round_id
        JOIN matches m ON m.match_id = k.match_id
        ORDER BY k.round_id, timestamp_ms
    """
    df_kills = pd.read_sql_query(kills_sql, conn)
    df_kills["site_norm"] = df_kills["site"].apply(_site_norm)

    # Player locations
    pl_sql = """
        SELECT
            round_id,
            match_id,
            timestamp_ms,
            player_name,
            team,
            location_x,
            location_y,
            view_radians
        FROM player_locations
        ORDER BY round_id, timestamp_ms
    """
    try:
        df_pl = pd.read_sql_query(pl_sql, conn)
    except Exception:
        df_pl = pd.DataFrame(
            columns=[
                "round_id", "match_id", "timestamp_ms", "player_name",
                "team", "location_x", "location_y", "view_radians"
            ]
        )

    # Player predictions
    df_pred = _load_prediction_rows(conn)

    # Players table
    players_sql = """
        SELECT
            match_id,
            name AS player_name,
            team_id AS team,
            agent
        FROM players
    """
    try:
        df_players = pd.read_sql_query(players_sql, conn)
        has_players = not df_players.empty
    except Exception:
        df_players = pd.DataFrame(columns=["match_id", "player_name", "team", "agent"])
        has_players = False

    # Round metadata
    round_cols = get_table_columns(conn, "rounds")
    has_plant_site = "plant_site" in round_cols

    if has_plant_site:
        rounds_sql = """
            SELECT
                r.round_id,
                r.match_id,
                r.round_num,
                m.map,
                r.side,
                r.plant_site,
                r.plant_x,
                r.plant_y,
                r.plant_time_ms
            FROM rounds r
            JOIN matches m ON m.match_id = r.match_id
        """
    else:
        rounds_sql = """
            SELECT
                r.round_id,
                r.match_id,
                r.round_num,
                m.map,
                r.side,
                NULL AS plant_site,
                r.plant_x,
                r.plant_y,
                r.plant_time_ms
            FROM rounds r
            JOIN matches m ON m.match_id = r.match_id
        """

    df_rounds = pd.read_sql_query(rounds_sql, conn)
    conn.close()

    if df_kills.empty:
        raise RuntimeError("No kills found in database.")

    site_centroids = _compute_site_centroids_from_kills(df_kills)
    round_info_map = df_rounds.set_index("round_id").to_dict("index")

    X_static, X_seq, y, round_ids_out = [], [], [], []

    kills_by_round = {rid: grp.copy() for rid, grp in df_kills.groupby("round_id", sort=False)}
    pl_by_round = (
        {rid: grp.copy() for rid, grp in df_pl.groupby("round_id", sort=False)}
        if not df_pl.empty else {}
    )
    pred_by_round = (
        {rid: grp.copy() for rid, grp in df_pred.groupby("round_id", sort=False)}
        if not df_pred.empty else {}
    )

    empty_loc_df = pd.DataFrame(columns=[
        "round_id", "match_id", "timestamp_ms", "player_name",
        "team", "location_x", "location_y", "view_radians"
    ])

    empty_pred_df = pd.DataFrame(columns=[
        "round_id", "match_id", "round_num", "player_name",
        "team", "time_bucket", "predicted_x", "predicted_y", "confidence"
    ])

    for round_id, grp in kills_by_round.items():
        if round_id not in label_map:
            continue

        label = int(label_map[round_id])
        round_meta = dict(round_info_map.get(round_id, {}))

        match_id = round_meta.get("match_id", None)
        map_name = round_meta.get(
            "map",
            grp["map"].iloc[0] if "map" in grp.columns and not grp.empty else ""
        )

        round_num = int(round_meta.get("round_num", 0))
        round_meta["attack_team"] = _infer_attack_team_from_round_num(round_num)
        round_meta["map"] = map_name

        loc_grp = pl_by_round.get(round_id, empty_loc_df)
        pred_grp = pred_by_round.get(round_id, empty_pred_df)

        if has_players and match_id is not None:
            all_players_round = (
                df_players[df_players["match_id"] == match_id][["player_name", "team"]]
                .drop_duplicates()
            )
            agents_in_round = df_players[df_players["match_id"] == match_id]["agent"]
        else:
            all_players_round = loc_grp[["player_name", "team"]].drop_duplicates()
            agents_in_round = grp["killer_agent"].dropna()

        # Static snapshot
        kills_at_t = grp[grp["timestamp_ms"] <= snapshot_ms]
        player_snapshot_t = _latest_player_snapshot(loc_grp, snapshot_ms)

        feat_static = _extract_snapshot(
            kills_t=kills_at_t,
            t_ms=snapshot_ms,
            all_kills=grp,
            agents_series=agents_in_round,
            round_meta=round_meta,
            player_snapshot=player_snapshot_t,
            all_players_round=all_players_round,
            loc_round=loc_grp,
            pred_round=pred_grp,
            site_centroids=site_centroids,
        )
        X_static.append(feat_static)
        y.append(label)
        round_ids_out.append(round_id)

        # Temporal sequence
        seq = []
        for t_ms in seq_steps:
            kills_at_t = grp[grp["timestamp_ms"] <= t_ms]
            player_snapshot_t = _latest_player_snapshot(loc_grp, t_ms)

            seq.append(_extract_snapshot(
                kills_t=kills_at_t,
                t_ms=t_ms,
                all_kills=grp,
                agents_series=agents_in_round,
                round_meta=round_meta,
                player_snapshot=player_snapshot_t,
                all_players_round=all_players_round,
                loc_round=loc_grp,
                pred_round=pred_grp,
                site_centroids=site_centroids,
            ))

        X_seq.append(np.stack(seq, axis=0))

    X_static = np.array(X_static, dtype=np.float32)
    X_seq = np.array(X_seq, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"[features] Built datasets: N={len(y)}, T={len(seq_steps)}, F={FEATURE_DIM}")
    print(f"[features] X_static shape: {X_static.shape}")
    print(f"[features]   X_seq shape:  {X_seq.shape}")
    print(f"[features]   Class dist:   {dict(zip(*np.unique(y, return_counts=True)))}")

    return {
        "X_static": X_static,
        "y_static": y,
        "X_seq": X_seq,
        "y_seq": y,
        "round_ids": round_ids_out,
        "label_names": LABEL_NAMES,
    }


def save_datasets(datasets: dict, out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "X_static.npy", datasets["X_static"])
    np.save(out_dir / "y_static.npy", datasets["y_static"])
    np.save(out_dir / "X_seq.npy", datasets["X_seq"])
    np.save(out_dir / "y_seq.npy", datasets["y_seq"])

    with open(out_dir / "round_ids.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(datasets["round_ids"]))

    print(f"[features] Datasets saved to {out_dir}/")


def load_datasets(out_dir: Path = OUT_DIR) -> dict:
    data = {
        "X_static": np.load(out_dir / "X_static.npy"),
        "y_static": np.load(out_dir / "y_static.npy"),
        "X_seq": np.load(out_dir / "X_seq.npy"),
        "y_seq": np.load(out_dir / "y_seq.npy"),
    }

    round_ids_path = out_dir / "round_ids.txt"
    if round_ids_path.exists():
        data["round_ids"] = [
            line.strip() for line in round_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build feature datasets for tactical intent prediction.")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument(
        "--snapshot-ms",
        type=int,
        default=DEFAULT_SNAP,
        help="Static snapshot time in milliseconds (default: 10000)"
    )
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"[features] Database not found: {db}")
        raise SystemExit(1)

    datasets = build_datasets(db, snapshot_ms=args.snapshot_ms)
    save_datasets(datasets, Path(args.out_dir))