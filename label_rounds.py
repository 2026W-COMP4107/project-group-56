"""
label_rounds.py
---------------
Week 1 deliverable: Define and apply tactical-intent labeling heuristics.

Assigns one of six class labels to each attacking round based on observable
game-state signals derived from kill timestamps and player positions:

    0  A-Site Rush       — fast A commitment, first kill < 20 s on A side
    1  B-Site Rush       — fast B commitment, first kill < 20 s on B side
    2  Split             — kills spread across both sites in first 30 s
    3  Default           — slow, balanced spread with no early commitment
    4  Fake              — early A/B pressure followed by switch to other site
    5  Slow Execute      — coordinated late push (first kill >= 30 s)

Heuristics are intentionally transparent so they can be validated by hand
and refined later once we have ground-truth labels.

Usage
-----
    python label_rounds.py              # prints label distribution to stdout
    python label_rounds.py --update-db  # writes labels into valorant.db

The labels are stored in a new `round_labels` table:
    round_id  TEXT PK
    label     INTEGER  (0-5)
    label_name TEXT
    confidence REAL    (heuristic confidence, 0-1)
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

DB_PATH = Path("valorant.db")

# ── Label definitions ─────────────────────────────────────────────────────────
LABEL_NAMES = {
    0: "A-Site Rush",
    1: "B-Site Rush",
    2: "A Split",
    3: "B Split",
    4: "Default",
    5: "Fake",
    6: "Slow Execute",
}

# ── Heuristic thresholds ──────────────────────────────────────────────────────
RUSH_THRESHOLD_MS = 15_000
EARLY_WINDOW_MS = 30_000
MID_WINDOW_MS = 45_000
LATE_THRESHOLD_MS = 30_000
FAKE_SWITCH_GAP_MS = 10_000

# Split/fake tuning
MIN_KNOWN_EARLY_EVENTS = 2
SPLIT_OTHER_SITE_RATIO_MIN = 0.22
SPLIT_OTHER_SITE_RATIO_MAX = 0.55
FAKE_OTHER_SITE_RATIO_MIN = 0.60
FINAL_SITE_DOMINANCE_MIN = 0.60


def _norm_site(site_str: str | None) -> str:
    if site_str is None:
        return ""
    s = str(site_str).strip().upper()
    if s in {"A", "A SITE", "A-SITE"}:
        return "A"
    if s in {"B", "B SITE", "B-SITE"}:
        return "B"
    if s in {"MID", "M", "MIDDLE"}:
        return "MID"
    return ""


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def compute_site_centroids(df_rounds: pd.DataFrame) -> dict[str, dict[str, tuple[float, float]]]:
    """
    Build per-map A/B centroids from planted rounds.

    Returns:
        {map_name: {"A": (x, y), "B": (x, y)}}
    """
    centroids: dict[str, dict[str, tuple[float, float]]] = {}

    usable = df_rounds.dropna(subset=["plant_site", "plant_x", "plant_y"]).copy()
    usable["plant_site"] = usable["plant_site"].apply(_norm_site)
    usable = usable[usable["plant_site"].isin(["A", "B"])]

    if usable.empty:
        return centroids

    for (map_name, site), grp in usable.groupby(["map", "plant_site"]):
        cx = float(grp["plant_x"].mean())
        cy = float(grp["plant_y"].mean())
        centroids.setdefault(map_name, {})[site] = (cx, cy)

    return centroids


def infer_site_from_coords(
    x: float | None,
    y: float | None,
    map_name: str,
    site_centroids: dict[str, dict[str, tuple[float, float]]],
) -> str:
    """
    Infer whether a coordinate is closer to A or B for a given map,
    based on planted-site centroids from the dataset.
    """
    if x is None or y is None or pd.isna(x) or pd.isna(y):
        return ""

    cmap = site_centroids.get(map_name, {})
    if "A" not in cmap or "B" not in cmap:
        return ""

    ax, ay = cmap["A"]
    bx, by = cmap["B"]

    d_a = float(np.hypot(x - ax, y - ay))
    d_b = float(np.hypot(x - bx, y - by))

    # If one centroid is clearly closer, assign it
    if d_a < d_b:
        return "A"
    if d_b < d_a:
        return "B"
    return ""


def choose_final_site(round_row: pd.Series, round_kills: pd.DataFrame) -> str:
    """
    Final committed site:
    1. plant_site if present
    2. majority of late-site events
    3. majority of all known site events
    """
    plant_site = _norm_site(round_row.get("plant_site"))
    if plant_site in {"A", "B"}:
        return plant_site

    if round_kills.empty:
        return ""

    late = round_kills[round_kills["timestamp_ms"] > EARLY_WINDOW_MS]
    late_known = late[late["event_site"].isin(["A", "B"])]
    if not late_known.empty:
        counts = late_known["event_site"].value_counts()
        return str(counts.idxmax())

    all_known = round_kills[round_kills["event_site"].isin(["A", "B"])]
    if not all_known.empty:
        counts = all_known["event_site"].value_counts()
        return str(counts.idxmax())

    return ""


def classify_round(round_row: pd.Series, kills_df: pd.DataFrame) -> tuple[int, str, float]:
    """
    Classify a single round using plant site/time + inferred event-site flow.
    """
    plant_time_ms = round_row.get("plant_time_ms")
    if pd.isna(plant_time_ms):
        plant_time_ms = None
    else:
        plant_time_ms = float(plant_time_ms)

    result = str(round_row.get("result") or "")
    final_site = choose_final_site(round_row, kills_df)

    if kills_df.empty:
        if plant_time_ms is not None and plant_time_ms >= LATE_THRESHOLD_MS:
            return 6, LABEL_NAMES[6], 0.65
        return 4, LABEL_NAMES[4], 0.40

    df = kills_df.sort_values("timestamp_ms").reset_index(drop=True)

    first_kill_ms = float(df["timestamp_ms"].min()) if not df.empty else None

    early = df[df["timestamp_ms"] <= EARLY_WINDOW_MS]
    mid = df[df["timestamp_ms"] <= MID_WINDOW_MS]
    late = df[df["timestamp_ms"] > EARLY_WINDOW_MS]

    early_known = early[early["event_site"].isin(["A", "B"])]
    mid_known = mid[mid["event_site"].isin(["A", "B"])]
    late_known = late[late["event_site"].isin(["A", "B"])]

    # Count early site activity
    early_a = int((early_known["event_site"] == "A").sum())
    early_b = int((early_known["event_site"] == "B").sum())
    early_total = early_a + early_b

    final_early = int((early_known["event_site"] == final_site).sum()) if final_site else 0
    opp_site = "B" if final_site == "A" else "A" if final_site == "B" else ""
    opp_early = int((early_known["event_site"] == opp_site).sum()) if opp_site else 0

    late_final = int((late_known["event_site"] == final_site).sum()) if final_site else 0
    late_total = len(late_known)

    # ── Slow Execute ─────────────────────────────────────────────────────────
    if plant_time_ms is not None and plant_time_ms >= LATE_THRESHOLD_MS:
        return 6, LABEL_NAMES[6], 0.82

    if first_kill_ms is not None and first_kill_ms >= LATE_THRESHOLD_MS:
        return 6, LABEL_NAMES[6], 0.75

    # ── Rush ─────────────────────────────────────────────────────────────────
    if final_site == "A":
        if first_kill_ms is not None and first_kill_ms <= RUSH_THRESHOLD_MS:
            if early_total == 0 or (final_early / max(1, early_total)) >= 0.75:
                return 0, LABEL_NAMES[0], 0.80

    if final_site == "B":
        if first_kill_ms is not None and first_kill_ms <= RUSH_THRESHOLD_MS:
            if early_total == 0 or (final_early / max(1, early_total)) >= 0.75:
                return 1, LABEL_NAMES[1], 0.80

    # ── Fake ─────────────────────────────────────────────────────────────────
    # Strong early pressure opposite final site, then later commitment to final site
    if final_site in {"A", "B"} and early_total >= MIN_KNOWN_EARLY_EVENTS and opp_site:
        opp_ratio = opp_early / max(1, early_total)
        late_final_ratio = late_final / max(1, late_total) if late_total > 0 else 0.0

        # Also require a timing switch
        opp_times = early_known.loc[early_known["event_site"] == opp_site, "timestamp_ms"]
        final_times = df.loc[df["event_site"] == final_site, "timestamp_ms"]

        if not opp_times.empty and not final_times.empty:
            gap = float(final_times.min()) - float(opp_times.max())
        else:
            gap = 0.0

        if (
            opp_ratio >= FAKE_OTHER_SITE_RATIO_MIN
            and late_total >= 1
            and late_final_ratio >= FINAL_SITE_DOMINANCE_MIN
            and gap >= FAKE_SWITCH_GAP_MS
        ):
            return 5, LABEL_NAMES[5], 0.72

    # ── Split ────────────────────────────────────────────────────────────────
    # Approximation: meaningful early action on both sites, but eventual commit to one
    if final_site == "A" and early_total >= MIN_KNOWN_EARLY_EVENTS:
        opp_ratio = early_b / max(1, early_total)
        final_ratio = early_a / max(1, early_total)
        if (
            SPLIT_OTHER_SITE_RATIO_MIN <= opp_ratio <= SPLIT_OTHER_SITE_RATIO_MAX
            and final_ratio > opp_ratio
        ):
            return 2, LABEL_NAMES[2], 0.66

    if final_site == "B" and early_total >= MIN_KNOWN_EARLY_EVENTS:
        opp_ratio = early_a / max(1, early_total)
        final_ratio = early_b / max(1, early_total)
        if (
            SPLIT_OTHER_SITE_RATIO_MIN <= opp_ratio <= SPLIT_OTHER_SITE_RATIO_MAX
            and final_ratio > opp_ratio
        ):
            return 3, LABEL_NAMES[3], 0.66

    # ── Default ──────────────────────────────────────────────────────────────
    return 4, LABEL_NAMES[4], 0.55


def label_all_rounds(db_path: Path = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    round_cols = get_table_columns(conn, "rounds")
    kill_cols = get_table_columns(conn, "kills")

    if "plant_site" not in round_cols:
        raise RuntimeError(
            "rounds.plant_site not found. Rebuild the database using the updated data_loader.py."
        )

    # Load rounds
    rounds_sql = """
        SELECT
            r.round_id,
            r.match_id,
            m.map,
            r.round_num,
            r.side,
            r.result,
            r.plant_site,
            r.plant_x,
            r.plant_y,
            r.plant_time_ms
        FROM rounds r
        JOIN matches m ON m.match_id = r.match_id
        ORDER BY r.match_id, r.round_num
    """
    df_rounds = pd.read_sql_query(rounds_sql, conn)

    # Compute per-map plant centroids
    site_centroids = compute_site_centroids(df_rounds)

    # Choose best available location fields
    kill_x_expr = "k.kill_location_x" if "kill_location_x" in kill_cols else (
        "k.killer_location_x" if "killer_location_x" in kill_cols else "NULL"
    )
    kill_y_expr = "k.kill_location_y" if "kill_location_y" in kill_cols else (
        "k.killer_location_y" if "killer_location_y" in kill_cols else "NULL"
    )
    site_expr = "COALESCE(k.site, '')" if "site" in kill_cols else "''"

    kills_sql = f"""
        SELECT
            k.kill_id,
            k.round_id,
            k.match_id,
            m.map,
            COALESCE(k.timestamp_ms, CAST(k.timestamp_sec * 1000 AS INTEGER), 0) AS timestamp_ms,
            {site_expr} AS raw_site,
            {kill_x_expr} AS event_x,
            {kill_y_expr} AS event_y
        FROM kills k
        JOIN matches m ON m.match_id = k.match_id
        ORDER BY k.match_id, k.round_id, timestamp_ms
    """
    df_kills = pd.read_sql_query(kills_sql, conn)
    conn.close()

    if df_rounds.empty:
        print("[label_rounds] No rounds found in database.")
        return pd.DataFrame()

    if df_kills.empty:
        print("[label_rounds] No kills found in database.")
        return pd.DataFrame()

    # Normalize or infer kill/event site
    df_kills["raw_site"] = df_kills["raw_site"].apply(_norm_site)
    df_kills["event_site"] = df_kills["raw_site"]

    missing_mask = df_kills["event_site"] == ""
    if missing_mask.any():
        inferred_sites = []
        for _, row in df_kills.loc[missing_mask].iterrows():
            inferred_sites.append(
                infer_site_from_coords(
                    row["event_x"],
                    row["event_y"],
                    row["map"],
                    site_centroids,
                )
            )
        df_kills.loc[missing_mask, "event_site"] = inferred_sites

    results = []

    for _, round_row in df_rounds.iterrows():
        side = str(round_row["side"] or "")
        if side not in ("attack", "Attack"):
            continue

        grp = df_kills[df_kills["round_id"] == round_row["round_id"]].copy()
        label, name, conf = classify_round(round_row, grp)

        results.append({
            "round_id": round_row["round_id"],
            "match_id": round_row["match_id"],
            "map": round_row["map"],
            "round_num": int(round_row["round_num"]),
            "label": label,
            "label_name": name,
            "confidence": round(conf, 3),
        })

    return pd.DataFrame(results)


def save_labels_to_db(labels_df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS round_labels (
            round_id   TEXT PRIMARY KEY,
            label      INTEGER NOT NULL,
            label_name TEXT    NOT NULL,
            confidence REAL    NOT NULL
        )
    """)
    conn.execute("DELETE FROM round_labels")
    labels_df[["round_id", "label", "label_name", "confidence"]].to_sql(
        "round_labels", conn, if_exists="append", index=False
    )
    conn.commit()
    conn.close()
    print(f"[label_rounds] Saved {len(labels_df)} round labels to database.")


def print_distribution(labels_df: pd.DataFrame) -> None:
    print("\n=== Tactical Label Distribution ===")
    counts = labels_df["label_name"].value_counts()
    total = len(labels_df)

    for name, cnt in counts.items():
        bar = "█" * int(40 * cnt / total)
        print(f"  {name:<18} {cnt:>4}  ({100*cnt/total:5.1f}%)  {bar}")

    print(f"\n  Total attack rounds labelled: {total}")
    print(f"  Mean confidence:              {labels_df['confidence'].mean():.3f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label Valorant rounds with tactical intent classes.")
    parser.add_argument("--db", default=str(DB_PATH), help="Path to valorant.db")
    parser.add_argument("--update-db", action="store_true", help="Write labels into the database")
    parser.add_argument("--csv", default="", help="Also write labels to this CSV path")
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"[label_rounds] Database not found: {db}")
        raise SystemExit(1)

    labels = label_all_rounds(db)
    if labels.empty:
        raise SystemExit(1)

    print_distribution(labels)

    if args.update_db:
        save_labels_to_db(labels, db)

    if args.csv:
        labels.to_csv(args.csv, index=False)
        print(f"[label_rounds] Labels written to {args.csv}")