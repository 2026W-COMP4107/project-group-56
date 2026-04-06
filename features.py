"""
features.py
-----------
Week 2 deliverable: Feature extraction and preprocessing pipeline.

Produces two dataset representations as described in the project proposal:

1. Static Snapshot Representation
   Game state at a fixed time window (default: first 10 s of round).
   Output: flattened feature vector per round → X_static.npy, y_static.npy

2. Temporal Sequence Representation
   Sequence of state snapshots sampled at [5 s, 10 s, 15 s, 20 s, 25 s, 30 s].
   Output: (N, T, F) array per round → X_seq.npy, y_seq.npy

Feature vector (per time step):
    [0]    timestamp_fraction         — t / MAX_ROUND_MS  (normalised time)
    [1..5] a_kills_t0…t5             — cumulative A-site kills at each snapshot
    [2..6] b_kills_t0…t5             — cumulative B-site kills at each snapshot
    [7]    total_kills                — total kills up to snapshot
    [8]    site_entropy               — H(A, B, MID kill dist) — spread measure
    [9]    kill_rate                  — kills per second up to snapshot
    [10]   a_ratio                    — fraction of kills on A
    [11]   b_ratio                    — fraction of kills on B
    [12]   mid_ratio                  — fraction of kills on MID
    [13]   first_kill_ms_norm         — first kill time (normalised)
    [14..20] agent_one_hot[0..6]      — attacker agent type composition
                                         (Controller, Duelist, Initiator,
                                          Sentinel, Unknown × count)
    [21]   has_spike_info             — 1 if spike-carrier kill data available
    Per agent category (5) × normalised count = 5 extra dims  → total F = 26

Usage
-----
    python features.py                        # build & save both datasets
    python features.py --snapshot-ms 15000    # change static snapshot time
    python features.py --out-dir features/    # save to custom directory
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path("valorant.db")
OUT_DIR = Path("features")

# ── Time configuration ────────────────────────────────────────────────────────
MAX_ROUND_MS = 100_000
DEFAULT_SNAP = 10_000
SEQ_STEPS_MS = [5_000, 10_000, 15_000, 20_000, 25_000, 30_000]

# ── Agent role taxonomy ───────────────────────────────────────────────────────
AGENT_ROLES: dict[str, str] = {
    # Controllers
    "Astra": "Controller", "Brimstone": "Controller", "Harbor": "Controller",
    "Omen": "Controller", "Viper": "Controller", "Clove": "Controller",

    # Duelists
    "Jett": "Duelist", "Neon": "Duelist", "Phoenix": "Duelist", "Raze": "Duelist",
    "Reyna": "Duelist", "Yoru": "Duelist", "Iso": "Duelist", "Waylay": "Duelist",

    # Initiators
    "Breach": "Initiator", "Fade": "Initiator", "Gekko": "Initiator",
    "KAY/O": "Initiator", "KAYO": "Initiator", "Skye": "Initiator", "Sova": "Initiator",
    "Tejo": "Initiator",

    # Sentinels
    "Chamber": "Sentinel", "Cypher": "Sentinel", "Deadlock": "Sentinel",
    "Killjoy": "Sentinel", "Sage": "Sentinel", "Vyse": "Sentinel",

    # Custom / Unknown
    "Miks": "Controller",
    "Veto": "Sentinel",
    "Unknown": "Unknown",
}
ROLE_ORDER = ["Controller", "Duelist", "Initiator", "Sentinel", "Unknown"]
N_ROLES = len(ROLE_ORDER)
ROLE_IDX = {r: i for i, r in enumerate(ROLE_ORDER)}

# Original 21 + 7 plant features
FEATURE_DIM = 28


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


def _extract_snapshot(
    kills_t: pd.DataFrame,
    t_ms: int,
    all_kills: pd.DataFrame,
    agents_series: pd.Series,
    round_meta: dict,
) -> np.ndarray:
    """
    Build a single feature vector for round state at time t_ms.

    Features [0..20] = original kill/role features
    Features [21..27] = plant-aware round features
    """
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)

    # [0] snapshot time
    vec[0] = t_ms / MAX_ROUND_MS

    # Kill-site distribution up to snapshot
    if not kills_t.empty:
        sites = kills_t["site_norm"].value_counts()
        a_cnt = int(sites.get("A", 0))
        b_cnt = int(sites.get("B", 0))
        m_cnt = int(sites.get("MID", 0))
        total = a_cnt + b_cnt + m_cnt

        # [1] A ratio, [2] B ratio, [3] MID ratio
        vec[1] = a_cnt / max(1, total)
        vec[2] = b_cnt / max(1, total)
        vec[3] = m_cnt / max(1, total)

        # [4] total kills normalized
        vec[4] = total / 10.0

        # [5] site entropy
        vec[5] = _entropy(np.array([a_cnt, b_cnt, m_cnt]))

        # [6] kill rate
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

    # [11..12] reserved/padding
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

    # [19] any A-site pressure
    vec[19] = 1.0 if vec[1] > 0 else 0.0

    # [20] any B-site pressure
    vec[20] = 1.0 if vec[2] > 0 else 0.0

    # ── Plant-aware features ────────────────────────────────────────────────
    plant_site = str(round_meta.get("plant_site") or "").strip().upper()
    plant_x = round_meta.get("plant_x")
    plant_y = round_meta.get("plant_y")
    plant_time_ms = round_meta.get("plant_time_ms")

    # [21] plant_is_A
    vec[21] = 1.0 if plant_site == "A" else 0.0

    # [22] plant_is_B
    vec[22] = 1.0 if plant_site == "B" else 0.0

    # [23] has_plant
    vec[23] = 1.0 if plant_site in ("A", "B") else 0.0

    # [24] plant_time_norm
    if plant_time_ms is not None and not pd.isna(plant_time_ms):
        vec[24] = min(float(plant_time_ms), MAX_ROUND_MS) / MAX_ROUND_MS
    else:
        vec[24] = 1.0

    # [25] late_plant_flag
    if plant_time_ms is not None and not pd.isna(plant_time_ms):
        vec[25] = 1.0 if float(plant_time_ms) >= 30_000 else 0.0
    else:
        vec[25] = 0.0

    # [26] plant_x_norm
    vec[26] = _safe_norm_coord(plant_x)

    # [27] plant_y_norm
    vec[27] = _safe_norm_coord(plant_y)

    return vec


def build_datasets(
    db_path: Path = DB_PATH,
    snapshot_ms: int = DEFAULT_SNAP,
    seq_steps: list[int] | None = None
) -> dict:
    """
    Build static and temporal datasets from the database.

    Returns
    -------
    dict with keys:
        X_static   (N, FEATURE_DIM)
        y_static   (N,)
        X_seq      (N, T, FEATURE_DIM)
        y_seq      (N,)
        round_ids  list[str]
        label_names dict[int, str]
    """
    if seq_steps is None:
        seq_steps = SEQ_STEPS_MS

    conn = sqlite3.connect(db_path)

    # Load labels
    labels_sql = "SELECT round_id, label, label_name FROM round_labels"
    try:
        df_labels = pd.read_sql_query(labels_sql, conn)
    except Exception:
        conn.close()
        raise RuntimeError("round_labels table not found. Run: python label_rounds.py --update-db")

    label_map = df_labels.set_index("round_id")["label"].to_dict()

    # Load kills
    kills_sql = """
        SELECT
            k.kill_id,
            k.round_id,
            k.match_id,
            r.side,
            COALESCE(k.timestamp_ms, CAST(k.timestamp_sec * 1000 AS INTEGER), 0) AS timestamp_ms,
            COALESCE(k.site, 'MID') AS site,
            k.killer,
            k.killer_agent,
            k.victim
        FROM kills k
        JOIN rounds r ON r.round_id = k.round_id
        JOIN matches m ON m.match_id = k.match_id
        ORDER BY k.round_id, timestamp_ms
    """
    df_kills = pd.read_sql_query(kills_sql, conn)

    # Load attacker agents per round
    agents_sql = """
        SELECT p.match_id, p.agent, p.team_id, r.round_id, r.side
        FROM players p
        JOIN rounds r ON r.match_id = p.match_id
        WHERE p.team_id IS NOT NULL
    """
    try:
        df_agents = pd.read_sql_query(agents_sql, conn)
        has_agents = not df_agents.empty
    except Exception:
        df_agents = pd.DataFrame()
        has_agents = False

    # Load round-level plant metadata
    rounds_sql = """
        SELECT
            round_id,
            plant_site,
            plant_x,
            plant_y,
            plant_time_ms
        FROM rounds
    """
    try:
        df_rounds = pd.read_sql_query(rounds_sql, conn)
        round_info_map = df_rounds.set_index("round_id").to_dict("index")
    except Exception:
        round_info_map = {}

    conn.close()

    if df_kills.empty:
        raise RuntimeError("No kills found in database.")

    df_kills["site_norm"] = df_kills["site"].apply(_site_norm)

    X_static, X_seq, y, round_ids_out = [], [], [], []

    for round_id, grp in df_kills.groupby("round_id", sort=False):
        if round_id not in label_map:
            continue

        label = int(label_map[round_id])
        round_meta = round_info_map.get(round_id, {})

        if has_agents:
            agents_in_round = df_agents[df_agents["round_id"] == round_id]["agent"]
        else:
            agents_in_round = grp["killer_agent"].dropna()

        # Static snapshot
        kills_at_t = grp[grp["timestamp_ms"] <= snapshot_ms]
        feat_static = _extract_snapshot(kills_at_t, snapshot_ms, grp, agents_in_round, round_meta)
        X_static.append(feat_static)
        y.append(label)
        round_ids_out.append(round_id)

        # Temporal sequence
        seq = []
        for t_ms in seq_steps:
            kills_at_t = grp[grp["timestamp_ms"] <= t_ms]
            seq.append(_extract_snapshot(kills_at_t, t_ms, grp, agents_in_round, round_meta))
        X_seq.append(np.stack(seq, axis=0))

    X_static = np.array(X_static, dtype=np.float32)
    X_seq = np.array(X_seq, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    label_names = df_labels.set_index("label")["label_name"].drop_duplicates().to_dict()

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
        "label_names": label_names,
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
    return {
        "X_static": np.load(out_dir / "X_static.npy"),
        "y_static": np.load(out_dir / "y_static.npy"),
        "X_seq": np.load(out_dir / "X_seq.npy"),
        "y_seq": np.load(out_dir / "y_seq.npy"),
    }


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