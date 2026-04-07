"""
============================
Predicts where each attacker is during the first 30 s of a Bind round,
split into two 15-second buckets (0-15 / 15-30).
"""


import math
import sqlite3
import networkx as nx
from typing import Optional

# Import normalization utilities from serve_db
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from serve_db import normalize_coord, MAP_SCALARS

BIND_DB_OFFSETS = (
    MAP_SCALARS["Bind"]["xScalarToAdd"],
    MAP_SCALARS["Bind"]["yScalarToAdd"],
)

def _unnormalize_db_coords(x: float, y: float) -> tuple[float, float]:
    """Convert normalized viewer/DB coordinates back to raw in-game Bind coordinates."""
    scalars = MAP_SCALARS["Bind"]
    raw_x = (x - scalars["xScalarToAdd"]) / scalars["xMultiplier"]
    raw_y = (y - scalars["yScalarToAdd"]) / scalars["yMultiplier"]
    return raw_x, raw_y

def _to_db_coords(x: float, y: float) -> tuple[float, float]:
    """Convert Bind map_zones space to viewer/DB normalized [0,1] space."""
    return (x + BIND_DB_OFFSETS[0], y + BIND_DB_OFFSETS[1])

def _to_mz_coords(x: float, y: float) -> tuple[float, float]:
    """Convert viewer/DB normalized [0,1] space to Bind map_zones space."""
    return (x - BIND_DB_OFFSETS[0], y - BIND_DB_OFFSETS[1])

# == Bind zone polygons (normalized coords) ================
BIND_ZONES = [
    {"name": "attacker_spawn", "points": [[0.0312,-0.2497],[0.1571,-0.2497],[0.1535,0.2343],[0.01,0.2241],[0.0112,0.1159],[0.0159,0.0202],[-0.0394,0.0259],[-0.0382,-0.0504],[0.0359,-0.047]]},
    {"name": "defender_spawn", "points": [[-0.5402,-0.0604],[-0.5414,0.0073],[-0.4336,0.0476],[-0.3637,0.0488],[-0.365,0.0844],[-0.2485,0.0856],[-0.2473,0.0405],[-0.36,-0.0093],[-0.3588,-0.0343],[-0.3061,-0.0817],[-0.3074,-0.1707],[-0.3686,-0.1719],[-0.3723,-0.109],[-0.4287,-0.1078]]},
    {"name": "heaven",         "points": [[-0.3711,0.0903],[-0.3123,0.0903],[-0.2853,0.1935],[-0.2841,0.2517],[-0.3061,0.26],[-0.3699,0.2612]]},
    {"name": "a_site",         "points": [[-0.2853,0.0891],[-0.2853,0.2671],[-0.224,0.2695],[-0.1419,0.2683],[-0.1468,0.1722],[-0.1162,0.1686],[-0.1088,0.0986],[-0.1934,0.0927],[-0.1971,0.1164],[-0.2424,0.1129],[-0.24,0.0903]]},
    {"name": "lamps",          "points": [[-0.2363,0.1034],[-0.1983,0.1034],[-0.1995,0.0808],[-0.1603,0.082],[-0.1554,0.0191],[-0.2412,0.0179]]},
    {"name": "a_short",        "points": [[-0.1542,0.0595],[-0.1542,0.0963],[-0.0414,0.0974],[-0.0414,0.1781],[0.0407,0.1781],[0.0395,0.0215],[-0.1064,0.0227]]},
    {"name": "a_lobby",        "points": [[0.0456,0.1105],[0.0431,0.2291],[0.0064,0.2279],[0.0039,0.2588],[-0.039,0.26],[-0.0377,0.3217],[0.1069,0.3264],[0.1081,0.1093]]},
    {"name": "showers",        "points": [[-0.1358,0.1876],[-0.1395,0.33],[-0.0684,0.33],[-0.0672,0.3169],[-0.0426,0.3157],[-0.0414,0.2623],[-0.0966,0.2623],[-0.0953,0.1888]]},
    {"name": "a_teleporter",   "points": [[-0.0549,0.324],[0.0051,0.3252],[0.0088,0.3845],[-0.0623,0.3869]]},
    {"name": "b_teleporter",   "points": [[-0.137,-0.1149],[-0.0623,-0.1126],[-0.0623,-0.0497],[-0.1395,-0.0485]]},
    {"name": "cave",           "points": [[0.1449,0.1496],[0.2061,0.1461],[0.2343,0.0334],[0.1963,-0.0509],[0.1608,-0.0437],[0.1473,-0.0129]]},
    {"name": "a_link",         "points": [[0.0358,-0.0876],[0.0762,-0.0876],[0.075,-0.007],[0.0297,-0.0034],[0.0162,0.0215],[-0.039,0.0227],[-0.0402,-0.0437],[0.0309,-0.0449]]},
    {"name": "b_link",         "points": [[0.0358,-0.0876],[0.1473,-0.0829],[0.151,-0.2348],[0.0346,-0.2288]]},
    {"name": "b_lobby",        "points": [[-0.0059,-0.3842],[-0.0059,-0.2265],[0.1608,-0.2098],[0.1522,-0.3866]]},
    {"name": "b_short",        "points": [[0.0309,-0.23],[-0.0267,-0.2324],[-0.0279,-0.1885],[-0.1223,-0.1826],[-0.1407,-0.1422],[-0.1284,-0.1161],[0.0284,-0.1126]]},
    {"name": "b_long",         "points": [[0.0002,-0.3771],[-0.001,-0.3226],[-0.1468,-0.3261],[-0.148,-0.4032],[-0.0794,-0.4044],[-0.0733,-0.3783]]},
    {"name": "b_garden",       "points": [[-0.1493,-0.3285],[-0.1505,-0.2656],[-0.088,-0.2644],[-0.0794,-0.281],[-0.0819,-0.3178],[-0.0966,-0.3273]]},
    {"name": "b_halls",        "points": [[-0.3686,-0.1778],[-0.3135,-0.1766],[-0.3196,-0.3095],[-0.3699,-0.3107]]},
    {"name": "b_elbow",        "points": [[-0.3699,-0.3119],[-0.24,-0.3131],[-0.24,-0.3439],[-0.1983,-0.3487],[-0.1983,-0.4044],[-0.2988,-0.4032],[-0.3,-0.3439],[-0.3699,-0.3427]]},
    {"name": "hookah",         "points": [[-0.1787,-0.2526],[-0.1775,-0.198],[-0.1174,-0.1861],[-0.0463,-0.1873],[-0.0451,-0.2573]]},
    {"name": "b_site",         "points": [[-0.3172,-0.3095],[-0.3172,-0.1885],[-0.2853,-0.1861],[-0.2804,-0.1766],[-0.2314,-0.1766],[-0.2314,-0.1968],[-0.1799,-0.198],[-0.1799,-0.2573],[-0.1517,-0.2763],[-0.1505,-0.3475],[-0.235,-0.3475],[-0.235,-0.3083]]},
]

# == Bind adjacency ================================
BIND_ADJACENCY = {
    "attacker_spawn": ["cave","a_lobby","b_link"],
    "a_lobby":        ["attacker_spawn", "a_teleporter", "showers"],
    "a_short":        ["a_lobby", "a_site", "lamps","a_link"],
    "lamps":          ["a_short", "a_site"],
    "a_site":         ["showers", "a_short", "lamps", "heaven", "defender_spawn"],
    "heaven":         ["a_site", "defender_spawn"],
    "a_link":         ["a_site", "b_link", "a_short"],
    "a_teleporter":   ["a_lobby", "b_teleporter"],
    "showers":        ["a_lobby","a_site"],
    "b_teleporter":   ["a_short","b_short"],
    "b_link":         ["a_link", "attacker_spawn", "b_lobby", "b_short"],
    "b_lobby":        ["b_long", "b_link", "b_short"],
    "b_long":         ["b_lobby", "b_garden"],
    "b_garden":       ["b_long", "b_site"],
    "b_elbow":        ["b_halls", "b_site"],
    "b_halls":        ["b_site", "defender_spawn", "b_elbow"],
    "b_short":        ["b_link", "b_lobby", "b_teleporter", "hookah"],
    "hookah":         ["b_short", "b_site",],
    "b_site":         ["hookah", "b_garden", "b_halls", "b_elbow"],
    "cave":           ["attacker_spawn"],
    "defender_spawn": ["heaven", "b_halls", "a_site"],
}

# Build NetworkX graph once
_G = nx.Graph()
for _node, _neighbors in BIND_ADJACENCY.items():
    for _nb in _neighbors:
        _G.add_edge(_node, _nb)

# == The 5 fixed attacker start positions (raw in-game coords) =================
ATTACKER_STARTS = [
    (   5900,	3400    ),  # 0 – showers spawn      
    (   6250,   1800    ),  # 1 – A main right side  
    (   7400,   -400    ),  # 2 – A main / cave-side 
    (   6250,   -3000   ),  # 3 – hookah spawn       
    (   5300,   -5000   ),  # 4 – B long spawn       
]
# ================ Coordinate space note ==================
# All coordinates use "map_zones" space (output of normalize_coord):
#   ranges roughly [-0.54, -0.23] to [0.15, 0.26]
# This is what the HTML viewer's toCanvas() expects directly - no conversion needed.
BUCKET_T = {
    "0-15":  0,
    "15-30": 15,
}

def _zone_centroid(zone_name: str) -> tuple[float, float]:
    for z in BIND_ZONES:
        if z["name"] == zone_name:
            pts = z["points"]
            return (sum(p[0] for p in pts) / len(pts),
                    sum(p[1] for p in pts) / len(pts))
    return (0.0, 0.0)


def _point_in_polygon(px: float, py: float, polygon: list) -> bool:
    n, inside, j = len(polygon), False, len(polygon) - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def _get_zone(x: float, y: float) -> str:
    for z in BIND_ZONES:
        if _point_in_polygon(x, y, z["points"]):
            return z["name"]
    return "unknown"


def _dist(a: tuple, b: tuple) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _nearest_start(ref_xy_raw: tuple[float, float]) -> int:
    """Return the index (0-4) of the closest ATTACKER_STARTS entry in raw in-game space."""
    return min(
        range(len(ATTACKER_STARTS)),
        key=lambda i: _dist(ATTACKER_STARTS[i], ref_xy_raw)
    )


def _spawn_offset(start_idx: int, overlap_idx: int, spacing: float = 500.0) -> tuple[float, float]:
    """
    Spread overlapping players assigned to the same spawn so their icons remain visible.
    Offsets are applied in raw in-game coordinates.
    """
    if overlap_idx == 0:
        return ATTACKER_STARTS[start_idx]

    patterns = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1, 1),
        (-1, 1),
        (1, -1),
        (-1, -1),
    ]
    dx_mult, dy_mult = patterns[(overlap_idx - 1) % len(patterns)]
    base_x, base_y = ATTACKER_STARTS[start_idx]
    return (
        base_x + dx_mult * spacing,
        base_y + dy_mult * spacing,
    )


def _graph_path(start_zone: str, end_zone: str) -> list[str]:
    if start_zone == end_zone:
        return [start_zone]
    try:
        return nx.shortest_path(_G, start_zone, end_zone)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [start_zone]


def _bucket_window_ms(bucket: str) -> tuple[int, int]:
    start_s, end_s = bucket.split("-")
    return int(start_s) * 1000, int(end_s) * 1000


def _zone_for_db_xy(db_xy: tuple[float, float]) -> str:
    zone = _get_zone(*_to_mz_coords(*db_xy))
    return zone if zone != "unknown" else "attacker_spawn"



def _choose_spawn_for_anchor(anchor_db_xy: tuple[float, float], anchor_zone: str) -> int:
    best_idx = 0
    best_score = float("inf")

    for idx, start_raw_xy in enumerate(ATTACKER_STARTS):
        start_db_xy = normalize_coord(start_raw_xy[0], start_raw_xy[1], MAP_SCALARS["Bind"])
        start_zone = _zone_for_db_xy(start_db_xy)
        path_len = max(0, len(_graph_path(start_zone, anchor_zone)) - 1)
        distance_score = _dist(_unnormalize_db_coords(*start_db_xy), _unnormalize_db_coords(*anchor_db_xy))
        score = path_len * 100000 + distance_score
        if score < best_score:
            best_score = score
            best_idx = idx

    return best_idx


def _path_position_between_zones(
    start_zone: str,
    end_zone: str,
    step_index: int,
) -> str:
    path = _graph_path(start_zone, end_zone)
    if not path:
        return start_zone
    return path[min(max(step_index, 0), len(path) - 1)]


def _zone_db_point(zone: str) -> tuple[float, float]:
    mz_x, mz_y = _zone_centroid(zone)
    return _to_db_coords(mz_x, mz_y)


def _bucket_explicit_ping(
    name: str,
    kills: list[dict],
    bucket_start_ms: int,
    bucket_end_ms: int,
    prefer_latest: bool = False,
) -> Optional[tuple[int, tuple[float, float], str]]:
    """
    Return a player's actual sampled location inside this bucket as:
    (timestamp_ms, (db_x, db_y), zone_name)

    Prefer player_locations snapshots over killer shot coordinates because a kill
    location is noisy and can jump across walls / map segments.
    """
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
            db_x, db_y = normalize_coord(x, y, MAP_SCALARS["Bind"])
            zone = _zone_for_db_xy((db_x, db_y))
            sample = (ts, (db_x, db_y), zone)
            break

        if sample is None and k["killer"] == name:
            db_x, db_y = normalize_coord(k["killer_x"], k["killer_y"], MAP_SCALARS["Bind"])
            zone = _zone_for_db_xy((db_x, db_y))
            sample = (ts, (db_x, db_y), zone)

        if sample is None:
            continue

        if chosen is None:
            chosen = sample
        elif prefer_latest and ts > chosen[0]:
            chosen = sample
        elif not prefer_latest and ts < chosen[0]:
            chosen = sample

    return chosen


# ================ Main prediction function ================

def predict_attacker_positions(
    players: list[dict],
    kills: list[dict],
    first_kill_snapshot: Optional[dict[str, tuple[float, float]]] = None,
) -> list[dict]:
    """
    Conservative Bind prediction model:
    - 0-15 stays on the assigned spawn.
    - Prefer a real 15-30 killer ping as the anchor.
    - Never place a bucket farther than 3 zone transitions from the previous 15-second state.
    - If there is no explicit ping for a bucket, use a reachable zone centroid on the graph path.
    """
    results = []
    death_time: dict[str, int] = {}
    death_pos: dict[str, tuple[float, float]] = {}

    for k in sorted(kills, key=lambda x: x["timestamp_ms"]):
        victim = k["victim"]
        if k.get("victim_side") == "attack" and victim not in death_time:
            death_time[victim] = k["timestamp_ms"]

            victim_x = k.get("victim_x")
            victim_y = k.get("victim_y")
            if victim_x is not None and victim_y is not None:
                db_x, db_y = normalize_coord(victim_x, victim_y, MAP_SCALARS["Bind"])
                death_pos[victim] = _to_mz_coords(db_x, db_y)

    player_starts: dict[str, int] = {}
    start_overlap_index: dict[str, int] = {}
    start_counts: dict[int, int] = {}

    for p in sorted(players, key=lambda player: player["name"]):
        name = p["name"]
        anchor_1530 = _bucket_explicit_ping(name, kills, 15000, 30000, prefer_latest=True)
        anchor = anchor_1530

        if anchor is not None:
            _, anchor_db_xy, anchor_zone = anchor
            start_idx = _choose_spawn_for_anchor(anchor_db_xy, anchor_zone)
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

        start_raw_xy = ATTACKER_STARTS[start_idx]
        spaced_start_raw_xy = _spawn_offset(start_idx, start_overlap_index.get(name, 0))
        start_db_xy = normalize_coord(start_raw_xy[0], start_raw_xy[1], MAP_SCALARS["Bind"])
        spaced_start_db_xy = normalize_coord(
            spaced_start_raw_xy[0], spaced_start_raw_xy[1], MAP_SCALARS["Bind"]
        )
        spawn_zone = _zone_for_db_xy(start_db_xy)

        explicit_1530 = _bucket_explicit_ping(name, kills, 15000, 30000, prefer_latest=True)

        bucket_predictions = {
            "0-15": (spaced_start_db_xy, spawn_zone, 0.95),
        }

        if explicit_1530 is not None:
            _, db_xy_1530, zone_1530 = explicit_1530
            step_zone = _path_position_between_zones(
                spawn_zone,
                zone_1530,
                1   # second bucket
            )
            if step_zone == zone_1530:
                bucket_predictions["15-30"] = (db_xy_1530, zone_1530, 0.9)
            else:
                bucket_predictions["15-30"] = (_zone_db_point(step_zone), step_zone, 0.72)
        else:
            target = _furthest_attacker_zone(spawn_zone)

            step_zone = _path_position_between_zones(
                spawn_zone,
                target,
                1
            )

            bucket_predictions["15-30"] = (
                _zone_db_point(step_zone),
                step_zone,
                0.6
            )


        for bucket in ["0-15", "15-30"]:
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


def _furthest_attacker_zone(current_zone: str) -> str:
    """Determine which site zone the player is heading toward."""
    a_side = {"a_lobby", "a_short", "lamps", "showers", "a_site",
              "a_link", "a_teleporter", "heaven", "cave"}
    b_side = {"b_long", "b_lobby", "b_link", "b_short", "b_garden",
              "b_elbow", "b_halls", "hookah", "b_teleporter", "b_site"}
    if current_zone in a_side:
        return "a_site"
    if current_zone in b_side:
        return "b_site"
    return "a_site"


# ================ DB integration ================

def generate_bind_attacker_predictions(db_path: str) -> None:
    """
    Pull rounds, players, and kills for Bind from the DB, run predictions,
    and write results to the `player_predictions` table.
    Only generates predictions for Bind attack rounds.
    Drops and recreates the table to start fresh (ready for other maps later).
    """
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Drop existing table and recreate fresh
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

    pred_counter = 0
    rows = []

    # Load ALL Bind data upfront to avoid nested queries in loop
    all_players = {}  # match_id -> {name -> agent}
    player_teams = {}  # match_id -> {name -> team}
    all_kills = {}  # round_id -> [kill dicts]
    
    # Load all player agents
    for row in conn.execute("""
        SELECT DISTINCT match_id, name, agent FROM players WHERE match_id IN (
            SELECT DISTINCT match_id FROM matches WHERE map = 'Bind'
        )
    """):
        match_id = row["match_id"]
        name = row["name"]
        agent = row["agent"]
        if match_id not in all_players:
            all_players[match_id] = {}
        all_players[match_id][name] = agent

    # Load all player team assignments
    for row in conn.execute("""
        SELECT DISTINCT match_id, name, team_id FROM players WHERE match_id IN (
            SELECT DISTINCT match_id FROM matches WHERE map = 'Bind'
        )
    """):
        match_id = row["match_id"]
        name = row["name"]
        team = row["team_id"]
        if match_id not in player_teams:
            player_teams[match_id] = {}
        player_teams[match_id][name] = team

    # Load all kills for Bind rounds
    for row in conn.execute("""
        SELECT k.round_id, k.timestamp_ms, k.killer, k.victim,
               k.killer_location_x, k.killer_location_y,
               k.victim_location_x, k.victim_location_y,
               k.match_id
        FROM   kills k
        JOIN   rounds r ON k.round_id = r.round_id
        JOIN   matches m ON k.match_id = m.match_id
        WHERE  m.map = 'Bind' AND r.round_num < 24 AND k.timestamp_ms <= 50000
        ORDER  BY k.round_id, k.timestamp_ms
    """):
        round_id = row["round_id"]
        if round_id not in all_kills:
            all_kills[round_id] = []
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
            "killer_x": row["killer_location_x"] or 0.0,
            "killer_y": row["killer_location_y"] or 0.0,
            "victim_x": row["victim_location_x"],
            "victim_y": row["victim_location_y"],
            "match_id": match_id,
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

    # Generate predictions for all Bind rounds
    # Red team attacks rounds 0-11, Blue team attacks rounds 12-23
    rounds = conn.execute("""
        SELECT m.match_id, m.map, r.round_id, r.round_num, r.side
        FROM   rounds r
        JOIN   matches m ON m.match_id = r.match_id
        WHERE  m.map = 'Bind' AND r.round_num < 24
        ORDER  BY m.match_id, r.round_num
    """).fetchall()

    print(f"[bind_attacker_predictions] Processing {len(rounds)} Bind rounds (both attack phases)...")
    
    processed = 0

    for rnd in rounds:
        match_id  = rnd["match_id"]
        round_id  = rnd["round_id"]
        round_num = rnd["round_num"]
        
        # Debug: print each round being processed
        if processed % 10 == 0:
            print(f"  [DEBUG] Processing round {processed}: {match_id[:8]}... round_num={round_num}", flush=True)
        
        # Determine which team is attacking this round
        # Red attacks 0-11, Blue attacks 12-23
        attacking_team = 'Red' if round_num < 12 else 'Blue'

        # Get attacker players for this round (from player_locations)
        attacker_names_set = set()
        for row in conn.execute("""
            SELECT DISTINCT player_name
            FROM   player_locations
            WHERE  round_id = ? AND team = ?
        """, (round_id, attacking_team)):
            attacker_names_set.add(row["player_name"])

        if not attacker_names_set:
            continue
        
        processed += 1
        if processed % 100 == 0:
            print(f"  ... processed {processed} rounds, {len(rows)} predictions so far")

        # Build player list with agents
        players = []
        for name in attacker_names_set:
            agent = all_players.get(match_id, {}).get(name, "Unknown")
            players.append({"name": name, "agent": agent})

        if not players:
            continue

        # Get kills from preloaded data and annotate whether the victim was an attacker
        kills = []
        for k in all_kills.get(round_id, []):
            k_copy = dict(k)
            k_copy["victim_side"] = "attack" if k.get("victim_team") == attacking_team else "defense"
            kills.append(k_copy)

        first_kill_snapshot = {}
        if kills:
            first_kill_ms = min(k["timestamp_ms"] for k in kills)
            for row in conn.execute("""
                SELECT player_name, location_x, location_y
                FROM player_locations
                WHERE round_id = ? AND team = ? AND ABS(timestamp_ms - ?) = (
                    SELECT MIN(ABS(timestamp_ms - ?))
                    FROM player_locations pl2
                    WHERE pl2.round_id = ? AND pl2.team = ? AND pl2.player_name = player_locations.player_name
                )
            """, (round_id, attacking_team, first_kill_ms, first_kill_ms, round_id, attacking_team)):
                if row["player_name"] and row["location_x"] is not None and row["location_y"] is not None:
                    first_kill_snapshot[row["player_name"]] = (row["location_x"], row["location_y"])

        predictions = predict_attacker_positions(players, kills, first_kill_snapshot)

        for pred in predictions:
            pid = f"bind_{pred_counter:06d}"
            pred_counter += 1
            raw_x, raw_y = _unnormalize_db_coords(
                pred["predicted_x"], pred["predicted_y"]
            )
            rows.append((
                pid, match_id, round_id, round_num,
                pred["player"], pred["agent"], attacking_team,
                pred["bucket"],
                raw_x, raw_y,
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
    print(f"[bind_attacker_predictions] Generated {len(rows)} Bind predictions.")


if __name__ == "__main__":
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "valorant.db"
    print(f"Generating Bind attacker predictions for {db_path}...")
    generate_bind_attacker_predictions(db_path)
    print("Done!")
