"""
Generic attacker position prediction generation for all supported maps.

This extends the original Bind-only predictor with map-aware topology from
`map_adjacency_graph.map_zones` and configurable attacker spawn anchors for:

- Bind
- Breeze
- Fracture
- Pearl
- Split
"""

import math
import sqlite3
from typing import Optional

from map_adjacency_graph.map_zones import ADJACENCY, MAP_SCALARS, ZONE_POLYGONS
from serve_db import normalize_coord


MAP_ATTACKER_STARTS: dict[str, list[tuple[float, float]]] = {
    "Bind": [
        (5900, 3400),   #showers
        (6250, 1800),   #a main right
        (7400, -400),   #a main left
        (6250, -3000),  #hookah
        (5300, -5000),  #b long
    ],
    "Breeze": [
        (4000, -3162),  # b elbow 
        (2473, 234),    # mid
        (1246, 3600),   # a main
    ],
    "Fracture": [
        (3726, 998),    # a main
        (13229, 1146),  # a dish
        (4194, -2526),  # b tree
        (11796, -3000), # b arcade
    ],
    "Pearl": [
        (4217, 5808),   # a main
        (2150, 1609),   # top mid
        (2467, -1021),  # mid shops
        (2150, -4026),  # b long
    ],
    "Split": [
        (-1316, -3232), # b lobby
        (1213, -2598),  # b link
        (2954, -2960),  # sewers
        (7284, -3428),  # a lobby
    ],
}

MAP_DEFAULT_TARGET_ZONES: dict[str, tuple[str, str]] = {
    "Bind": ("a_site", "b_site"),
    "Breeze": ("a_site", "b_site"),
    "Fracture": ("a_site", "b_site"),
    "Pearl": ("a_site", "b_site"),
    "Split": ("a_site", "b_site"),
}

BUCKET_ORDER = ["0-15", "15-30"]


def _build_graph(map_name: str) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for node, neighbors in ADJACENCY[map_name].items():
      graph.setdefault(node, set()).update(neighbors)
      for nb in neighbors:
        graph.setdefault(nb, set()).add(node)
    return graph


_GRAPHS = {map_name: _build_graph(map_name) for map_name in ADJACENCY}


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _point_in_polygon(px: float, py: float, polygon: list[list[float]]) -> bool:
    n, inside, j = len(polygon), False, len(polygon) - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _zone_centroid(zone_name: str, map_name: str) -> tuple[float, float]:
    for zone in ZONE_POLYGONS[map_name]:
        if zone["name"] == zone_name:
            pts = zone["points"]
            return (
                sum(p[0] for p in pts) / len(pts),
                sum(p[1] for p in pts) / len(pts),
            )
    return (0.0, 0.0)


def _get_zone_for_db_xy(map_name: str, db_xy: tuple[float, float]) -> str:
    for zone in ZONE_POLYGONS[map_name]:
        if _point_in_polygon(db_xy[0], db_xy[1], zone["points"]):
            return zone["name"]
    return "attacker_spawn"


def _graph_path(map_name: str, start_zone: str, end_zone: str) -> list[str]:
    if start_zone == end_zone:
        return [start_zone]

    graph = _GRAPHS[map_name]
    if start_zone not in graph or end_zone not in graph:
        return [start_zone]

    queue = [(start_zone, [start_zone])]
    seen = {start_zone}

    while queue:
        node, path = queue.pop(0)
        for nb in graph.get(node, ()):
            if nb in seen:
                continue
            next_path = path + [nb]
            if nb == end_zone:
                return next_path
            seen.add(nb)
            queue.append((nb, next_path))

    return [start_zone]


def _path_position_between_zones(map_name: str, start_zone: str, end_zone: str, step_index: int) -> str:
    path = _graph_path(map_name, start_zone, end_zone)
    if not path:
        return start_zone
    return path[min(max(step_index, 0), len(path) - 1)]


def _bucket_window_ms(bucket: str) -> tuple[int, int]:
    start_s, end_s = bucket.split("-")
    return int(start_s) * 1000, int(end_s) * 1000


def _normalized_to_game_coords(map_name: str, x: float, y: float) -> tuple[float, float]:
    scalars = MAP_SCALARS[map_name]
    raw_x = (x - scalars["xScalarToAdd"]) / scalars["xMultiplier"]
    raw_y = (y - scalars["yScalarToAdd"]) / scalars["yMultiplier"]
    return raw_x, raw_y


def _zone_db_point(map_name: str, zone_name: str) -> tuple[float, float]:
    return _zone_centroid(zone_name, map_name)


def _spawn_offset(start_xy: tuple[float, float], overlap_idx: int, spacing: float = 350.0) -> tuple[float, float]:
    if overlap_idx == 0:
        return start_xy

    patterns = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1),
    ]
    dx_mult, dy_mult = patterns[(overlap_idx - 1) % len(patterns)]
    return (start_xy[0] + dx_mult * spacing, start_xy[1] + dy_mult * spacing)


def _choose_site_target(map_name: str, anchor_zone: str) -> str:
    a_target, b_target = MAP_DEFAULT_TARGET_ZONES[map_name]
    zone_name = anchor_zone.lower()

    if zone_name.startswith("a_") or zone_name in {"a", "a_site", "a_main", "a_lobby", "a_short", "a_heaven", "a_link", "a_flowers", "a_dugout"}:
        return a_target
    if zone_name.startswith("b_") or zone_name in {"b", "b_site", "b_main", "b_lobby", "b_long", "b_heaven", "b_link", "hookah"}:
        return b_target

    a_path = _graph_path(map_name, "attacker_spawn", a_target)
    b_path = _graph_path(map_name, "attacker_spawn", b_target)
    return a_target if len(a_path) <= len(b_path) else b_target


def _bucket_explicit_ping(
    map_name: str,
    name: str,
    kills: list[dict],
    bucket_start_ms: int,
    bucket_end_ms: int,
    prefer_latest: bool = False,
) -> Optional[tuple[int, tuple[float, float], str]]:
    chosen = None

    for k in kills:
        ts = k["timestamp_ms"]
        if not (bucket_start_ms <= ts < bucket_end_ms):
            continue

        sample = None

        for loc in k.get("player_locations", []):
            if loc.get("player_name") != name:
                continue
            x = loc.get("location_x")
            y = loc.get("location_y")
            if x is None or y is None:
                continue
            db_xy = normalize_coord(x, y, MAP_SCALARS[map_name])
            zone = _get_zone_for_db_xy(map_name, db_xy)
            sample = (ts, db_xy, zone)
            break

        if sample is None and k["killer"] == name and k.get("killer_x") is not None and k.get("killer_y") is not None:
            db_xy = normalize_coord(k["killer_x"], k["killer_y"], MAP_SCALARS[map_name])
            zone = _get_zone_for_db_xy(map_name, db_xy)
            sample = (ts, db_xy, zone)

        if sample is None:
            continue

        if chosen is None:
            chosen = sample
        elif prefer_latest and ts > chosen[0]:
            chosen = sample
        elif not prefer_latest and ts < chosen[0]:
            chosen = sample

    return chosen


def _choose_spawn_for_anchor(map_name: str, anchor_db_xy: tuple[float, float], anchor_zone: str) -> int:
    starts = MAP_ATTACKER_STARTS[map_name]
    best_idx = 0
    best_score = float("inf")

    for idx, start_raw_xy in enumerate(starts):
        start_db_xy = normalize_coord(start_raw_xy[0], start_raw_xy[1], MAP_SCALARS[map_name])
        start_zone = _get_zone_for_db_xy(map_name, start_db_xy)
        path_len = max(0, len(_graph_path(map_name, start_zone, anchor_zone)) - 1)
        distance_score = _dist(start_db_xy, anchor_db_xy)
        score = path_len * 100000 + distance_score
        if score < best_score:
            best_score = score
            best_idx = idx

    return best_idx


def predict_attacker_positions(map_name: str, players: list[dict], kills: list[dict]) -> list[dict]:
    results = []
    death_time: dict[str, int] = {}

    for k in sorted(kills, key=lambda x: x["timestamp_ms"]):
        victim = k["victim"]
        if k.get("victim_side") == "attack" and victim not in death_time:
            death_time[victim] = k["timestamp_ms"]

    player_starts: dict[str, int] = {}
    start_overlap_index: dict[str, int] = {}
    start_counts: dict[int, int] = {}

    for p in sorted(players, key=lambda player: player["name"]):
        name = p["name"]
        anchor_1530 = _bucket_explicit_ping(map_name, name, kills, 15000, 30000, prefer_latest=True)

        if anchor_1530 is not None:
            _, anchor_db_xy, anchor_zone = anchor_1530
            start_idx = _choose_spawn_for_anchor(map_name, anchor_db_xy, anchor_zone)
        else:
            start_idx = 0

        player_starts[name] = start_idx
        overlap_idx = start_counts.get(start_idx, 0)
        start_overlap_index[name] = overlap_idx
        start_counts[start_idx] = overlap_idx + 1

    for p in players:
        name = p["name"]
        agent = p.get("agent", "Unknown")
        start_idx = player_starts.get(name, 0)

        start_raw_xy = MAP_ATTACKER_STARTS[map_name][start_idx]
        spaced_start_raw_xy = _spawn_offset(start_raw_xy, start_overlap_index.get(name, 0))
        start_db_xy = normalize_coord(start_raw_xy[0], start_raw_xy[1], MAP_SCALARS[map_name])
        spaced_start_db_xy = normalize_coord(spaced_start_raw_xy[0], spaced_start_raw_xy[1], MAP_SCALARS[map_name])
        spawn_zone = _get_zone_for_db_xy(map_name, start_db_xy)

        explicit_1530 = _bucket_explicit_ping(map_name, name, kills, 15000, 30000, prefer_latest=True)

        bucket_predictions = {
            "0-15": (spaced_start_db_xy, spawn_zone, 0.95),
        }

        if explicit_1530 is not None:
            _, db_xy_1530, zone_1530 = explicit_1530
            step_zone = _path_position_between_zones(map_name, spawn_zone, zone_1530, 1)
            if step_zone == zone_1530:
                bucket_predictions["15-30"] = (db_xy_1530, zone_1530, 0.9)
            else:
                bucket_predictions["15-30"] = (_zone_db_point(map_name, step_zone), step_zone, 0.72)
        else:
            target_zone = _choose_site_target(map_name, spawn_zone)
            step_zone = _path_position_between_zones(map_name, spawn_zone, target_zone, 1)
            bucket_predictions["15-30"] = (_zone_db_point(map_name, step_zone), step_zone, 0.6)

        for bucket in BUCKET_ORDER:
            _, bucket_end_ms = _bucket_window_ms(bucket)
            if name in death_time and death_time[name] <= bucket_end_ms:
                break

            (px, py), zone, confidence = bucket_predictions[bucket]
            results.append({
                "player": name,
                "agent": agent,
                "bucket": bucket,
                "predicted_x": round(px, 4),
                "predicted_y": round(py, 4),
                "zone": zone,
                "confidence": round(confidence, 2),
                "start_idx": start_idx,
                "alive": True,
            })

    return results


def generate_attacker_predictions(db_path: str) -> None:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS player_predictions")
    cur.execute("""
        CREATE TABLE player_predictions (
            prediction_id  TEXT PRIMARY KEY,
            match_id       TEXT,
            round_id       TEXT,
            round_num      INTEGER,
            player_name    TEXT,
            agent          TEXT,
            team           TEXT,
            time_bucket    TEXT,
            predicted_x    REAL,
            predicted_y    REAL,
            confidence     REAL
        )
    """)
    conn.commit()

    rows = []
    pred_counter = 0
    supported_maps = tuple(MAP_ATTACKER_STARTS.keys())

    all_players: dict[str, dict[str, str]] = {}
    player_teams: dict[str, dict[str, str]] = {}
    all_kills: dict[str, list[dict]] = {}

    map_placeholders = ",".join("?" for _ in supported_maps)

    for row in conn.execute(f"""
        SELECT DISTINCT match_id, name, agent
        FROM players
        WHERE match_id IN (
            SELECT DISTINCT match_id FROM matches WHERE map IN ({map_placeholders})
        )
    """, supported_maps):
        all_players.setdefault(row["match_id"], {})[row["name"]] = row["agent"]

    for row in conn.execute(f"""
        SELECT DISTINCT match_id, name, team_id
        FROM players
        WHERE match_id IN (
            SELECT DISTINCT match_id FROM matches WHERE map IN ({map_placeholders})
        )
    """, supported_maps):
        player_teams.setdefault(row["match_id"], {})[row["name"]] = row["team_id"]

    for row in conn.execute(f"""
        SELECT k.round_id, k.timestamp_ms, k.killer, k.victim,
               k.killer_location_x, k.killer_location_y,
               k.victim_location_x, k.victim_location_y,
               k.match_id, m.map
        FROM kills k
        JOIN rounds r ON k.round_id = r.round_id
        JOIN matches m ON k.match_id = m.match_id
        WHERE m.map IN ({map_placeholders}) AND r.round_num < 24 AND k.timestamp_ms <= 50000
        ORDER BY k.round_id, k.timestamp_ms
    """, supported_maps):
        round_id = row["round_id"]
        all_kills.setdefault(round_id, [])
        match_id = row["match_id"]
        victim_name = row["victim"]
        victim_team = player_teams.get(match_id, {}).get(victim_name)

        player_locations = conn.execute("""
            SELECT player_name, location_x, location_y
            FROM player_locations
            WHERE round_id = ? AND timestamp_ms = ?
        """, (round_id, row["timestamp_ms"])).fetchall()

        all_kills[round_id].append({
            "timestamp_ms": int(row["timestamp_ms"] or 0),
            "killer": row["killer"],
            "victim": victim_name,
            "killer_x": row["killer_location_x"],
            "killer_y": row["killer_location_y"],
            "victim_x": row["victim_location_x"],
            "victim_y": row["victim_location_y"],
            "match_id": match_id,
            "map_name": row["map"],
            "victim_team": victim_team,
            "player_locations": [
                {
                    "player_name": loc["player_name"],
                    "location_x": loc["location_x"],
                    "location_y": loc["location_y"],
                }
                for loc in player_locations
                if loc["location_x"] is not None and loc["location_y"] is not None
            ],
        })

    rounds = conn.execute(f"""
        SELECT m.match_id, m.map, r.round_id, r.round_num, r.side
        FROM rounds r
        JOIN matches m ON m.match_id = r.match_id
        WHERE m.map IN ({map_placeholders}) AND r.round_num < 24
        ORDER BY m.map, m.match_id, r.round_num
    """, supported_maps).fetchall()

    print(f"[attacker_predictions] Processing {len(rounds)} rounds across {', '.join(supported_maps)}...")

    for rnd in rounds:
        match_id = rnd["match_id"]
        round_id = rnd["round_id"]
        round_num = rnd["round_num"]
        map_name = rnd["map"]

        attacking_team = "Red" if round_num < 12 else "Blue"

        attacker_names_set = set()
        for row in conn.execute("""
            SELECT DISTINCT player_name
            FROM player_locations
            WHERE round_id = ? AND team = ?
        """, (round_id, attacking_team)):
            attacker_names_set.add(row["player_name"])

        if not attacker_names_set:
            continue

        players = [
            {"name": name, "agent": all_players.get(match_id, {}).get(name, "Unknown")}
            for name in attacker_names_set
        ]

        kills = []
        for k in all_kills.get(round_id, []):
            k_copy = dict(k)
            k_copy["victim_side"] = "attack" if k.get("victim_team") == attacking_team else "defense"
            kills.append(k_copy)

        predictions = predict_attacker_positions(map_name, players, kills)

        for pred in predictions:
            pid = f"{map_name.lower()}_{pred_counter:06d}"
            pred_counter += 1
            raw_x, raw_y = _normalized_to_game_coords(map_name, pred["predicted_x"], pred["predicted_y"])
            rows.append((
                pid,
                match_id,
                round_id,
                round_num,
                pred["player"],
                pred["agent"],
                attacking_team,
                pred["bucket"],
                raw_x,
                raw_y,
                pred["confidence"],
            ))

    cur.executemany("""
        INSERT OR REPLACE INTO player_predictions
            (prediction_id, match_id, round_id, round_num,
             player_name, agent, team,
             time_bucket, predicted_x, predicted_y, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()
    print(f"[attacker_predictions] Generated {len(rows)} predictions across all supported maps.")


if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else "valorant.db"
    print(f"Generating attacker predictions for {db_path}...")
    generate_attacker_predictions(db_path)
    print("Done!")
