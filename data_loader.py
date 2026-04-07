"""
data_loader.py
--------------
Handles all SQLite database interactions.
Loads Valorant match data from JSON or API responses into a normalized database.

Tables:
-------
- matches: Match metadata (match_id, map, date)
- rounds: Round data (round_id, match_num, side, winner)
- kills: Individual kill events with location data
- player_locations: All player positions during each kill event

Features:
---------
- Loads real Valorant API match data from JSON files
- Filters by map pool and game mode (competitive only)
- Captures killer/victim locations and all player positions
- Provides query functions for kills, locations, and player statistics
"""

from __future__ import annotations

import sqlite3
import pandas as pd
import numpy as np
import os
import random
import json
import math
from pathlib import Path
from datetime import datetime

DB_PATH = "valorant.db"

AGENTS   = ["Astra", "Breach", "Brimstone", "Chamber", "Clove",
            "Cypher", "Deadlock", "Fade", "Gekko", "Harbor",
            "Iso", "Jett", "KAYO", "Killjoy", "Miks", "Neon",
            "Omen", "Phoenix", "Raze", "Reyna", "Sage", "Skye",
            "Sova", "Tejo", "Veto", "Viper", "Vyse", "Waylay", "Yoru"]

MAPS     = ["Breeze", "Bind", "Fracture", "Pearl", "Split"]
SITES    = ["A", "B"]
SIDES    = ["attack", "defense"]
PLAYERS  = [f"Player{i}" for i in range(1, 21)] 


def load_from_json_file(json_path: str, db_path: str = DB_PATH) -> None:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "data" in data:
            matches = data["data"]
        elif isinstance(data, list):
            matches = data
        else:
            matches = [data]

        _insert_matches_to_db(matches, db_path)
        print(f"[data_loader] Loaded data from JSON file '{json_path}'.")
    except Exception as e:
        print(f"[data_loader] Error loading JSON file: {e}")


def prompt_for_file_selection(json_folder: str = "json", db_path: str = DB_PATH) -> None:
    folder_path = Path(json_folder)

    if not folder_path.exists():
        print(f"[data_loader] Error: Folder '{json_folder}' does not exist.")
        return

    json_files = sorted(list(folder_path.glob("**/*.json")))

    if not json_files:
        print(f"[data_loader] No JSON files found in '{json_folder}' or subdirectories.")
        return

    files_by_map = {}
    for file_path in json_files:
        relative_path = file_path.relative_to(folder_path)
        map_name = relative_path.parts[0] if len(relative_path.parts) > 1 else "root"
        files_by_map.setdefault(map_name, []).append(file_path)

    print(f"\n[data_loader] Available files in '{json_folder}':")
    print("=" * 70)
    file_index = 1
    index_to_path = {}

    for map_name in sorted(files_by_map.keys()):
        if map_name != "root":
            print(f"\n  {map_name}:")
        for file_path in sorted(files_by_map[map_name]):
            print(f"    [{file_index}] {file_path.name}")
            index_to_path[file_index] = file_path
            file_index += 1

    print("=" * 70)

    while True:
        try:
            selection = input("[data_loader] Enter the file number to load (or 'q' to quit): ").strip()

            if selection.lower() == "q":
                print("[data_loader] Cancelled.")
                return

            file_idx = int(selection)
            if file_idx not in index_to_path:
                print(f"[data_loader] Invalid selection. Please enter a number between 1 and {len(json_files)}.")
                continue

            selected_file = index_to_path[file_idx]
            print(f"\n[data_loader] Loading file: {selected_file}")

            with open(selected_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and "data" in data:
                matches = data["data"]
            elif isinstance(data, list):
                matches = data
            else:
                matches = [data]

            _insert_matches_to_db(matches, db_path)
            print(f"[data_loader] Successfully loaded data from '{selected_file.name}'.")
            break

        except ValueError:
            print("[data_loader] Invalid input. Please enter a valid number or 'q' to quit.")
        except Exception as e:
            print(f"[data_loader] Error: {e}")
            break


def load_all_json_files(json_folder: str = "json", db_path: str = DB_PATH) -> int:
    folder_path = Path(json_folder)

    if not folder_path.exists():
        print(f"[data_loader] Error: Folder '{json_folder}' does not exist.")
        return 0

    json_files = sorted(list(folder_path.glob("**/*.json")))

    if not json_files:
        print(f"[data_loader] No JSON files found in '{json_folder}' or subdirectories.")
        return 0

    total_matches = 0
    print(f"\n[data_loader] Found {len(json_files)} JSON files. Loading...")
    print("=" * 70)

    for idx, file_path in enumerate(json_files, 1):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, dict) and "data" in data:
                matches = data["data"]
            elif isinstance(data, list):
                matches = data
            else:
                matches = [data]

            match_count = len(matches) if isinstance(matches, list) else 1
            _insert_matches_to_db(matches, db_path)
            total_matches += match_count

            relative_path = file_path.relative_to(folder_path)
            print(f"  [{idx}/{len(json_files)}] {relative_path} → {match_count} match(es)")
        except Exception as e:
            relative_path = file_path.relative_to(folder_path)
            print(f"  [{idx}/{len(json_files)}] {relative_path} → ERROR: {e}")

    print("=" * 70)
    print(f"[data_loader] Total loaded: {total_matches} matches from {len(json_files)} files.")
    return total_matches


def load_from_api_response(api_response: dict, db_path: str = DB_PATH) -> None:
    matches = api_response.get("data", [api_response]) if isinstance(api_response, dict) else api_response
    _insert_matches_to_db(matches, db_path)
    print(f"[data_loader] Loaded {len(matches) if isinstance(matches, list) else 1} match(es) from API response.")


def _safe_site(site_value) -> str | None:
    if site_value is None:
        return None
    s = str(site_value).strip().upper()
    if s in {"A", "A SITE", "A-SITE"}:
        return "A"
    if s in {"B", "B SITE", "B-SITE"}:
        return "B"
    return None


def _extract_plant_info(round_info: dict) -> dict:
    plant = round_info.get("plant")
    if not plant:
        return {
            "plant_site": None,
            "plant_x": None,
            "plant_y": None,
            "plant_time_ms": None,
        }

    loc = plant.get("location", {}) or {}
    return {
        "plant_site": _safe_site(plant.get("site")),
        "plant_x": loc.get("x"),
        "plant_y": loc.get("y"),
        "plant_time_ms": plant.get("round_time_in_ms"),
    }


def _infer_round_side(round_num: int) -> str:
    return "attack" if round_num < 12 else "defense"


def _insert_matches_to_db(matches_data, db_path: str = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id   TEXT PRIMARY KEY,
            map        TEXT,
            date       TEXT
        );

        CREATE TABLE IF NOT EXISTS players (
            puuid             TEXT,
            match_id          TEXT,
            name              TEXT,
            tag               TEXT,
            team_id           TEXT,
            agent             TEXT,
            PRIMARY KEY (puuid, match_id),
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        );

        CREATE TABLE IF NOT EXISTS rounds (
            round_id          TEXT PRIMARY KEY,
            match_id          TEXT,
            round_num         INTEGER,
            side              TEXT,
            winner            TEXT,
            result            TEXT,
            plant_site        TEXT,
            plant_x           REAL,
            plant_y           REAL,
            plant_time_ms     INTEGER,
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        );

        CREATE TABLE IF NOT EXISTS kills (
            kill_id           TEXT PRIMARY KEY,
            round_id          TEXT,
            round_num         INTEGER,
            match_id          TEXT,
            killer            TEXT,
            victim            TEXT,
            killer_puuid      TEXT,
            victim_puuid      TEXT,
            killer_agent      TEXT,
            victim_agent      TEXT,
            site              TEXT,
            weapon            TEXT,
            killer_location_x REAL,
            killer_location_y REAL,
            victim_location_x REAL,
            victim_location_y REAL,
            timestamp_sec     REAL,
            timestamp_ms      INTEGER,
            is_first_blood    INTEGER,
            FOREIGN KEY (round_id) REFERENCES rounds(round_id)
        );

        CREATE TABLE IF NOT EXISTS player_locations (
            location_id  TEXT PRIMARY KEY,
            kill_id      TEXT,
            match_id     TEXT,
            round_id     TEXT,
            round_num    INTEGER,
            timestamp_ms INTEGER,
            puuid        TEXT,
            player_name  TEXT,
            team         TEXT,
            location_x   REAL,
            location_y   REAL,
            view_radians REAL,
            FOREIGN KEY (kill_id) REFERENCES kills(kill_id)
        );
    """)

    try:
        cur.execute("ALTER TABLE rounds ADD COLUMN plant_site TEXT")
    except sqlite3.OperationalError:
        pass

    if not isinstance(matches_data, list):
        matches_data = [matches_data]

    total_kills = 0
    matches_processed = 0

    for match in matches_data:
        if not isinstance(match, dict):
            continue

        metadata = match.get("metadata", {})
        match_id = metadata.get("match_id")
        if not match_id:
            continue

        map_data = metadata.get("map", {}) or {}
        map_name = map_data.get("name", "Unknown")

        if map_name not in MAPS:
            continue

        queue = metadata.get("queue", {}) or {}
        queue_id = str(queue.get("id", "")).lower()
        if queue_id != "competitive":
            continue

        started_at = metadata.get("started_at", "")
        try:
            date_obj = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            date = date_obj.strftime("%Y-%m-%d")
        except Exception:
            date = datetime.today().strftime("%Y-%m-%d")

        cur.execute(
            "INSERT OR IGNORE INTO matches VALUES (?,?,?)",
            (match_id, map_name, date)
        )
        matches_processed += 1

        players = match.get("players", []) or []
        rounds_data = match.get("rounds", []) or []
        kill_data = match.get("kills", []) or []

        player_agent_map = {}
        player_puuid_map = {}
        player_team_map = {}

        for player in players:
            player_name = player.get("name", "")
            player_tag = player.get("tag", "")
            player_puuid = player.get("puuid", "")
            agent_name = (player.get("agent", {}) or {}).get("name", "Unknown")
            team_id = player.get("team_id", "")

            player_agent_map[player_name] = agent_name
            player_puuid_map[player_name] = player_puuid
            player_team_map[player_name] = team_id

            cur.execute(
                "INSERT OR IGNORE INTO players VALUES (?,?,?,?,?,?)",
                (player_puuid, match_id, player_name, player_tag, team_id, agent_name)
            )

        rounds_dict = {}

        for round_info in rounds_data:
            if not round_info:
                continue

            round_num = round_info.get("id")
            if round_num is None:
                continue

            round_id = f"{match_id}_r{int(round_num):02d}"
            plant_info = _extract_plant_info(round_info)

            rounds_dict[round_id] = {
                "match_id": match_id,
                "round_num": int(round_num),
                "side": _infer_round_side(int(round_num)),
                "winner": round_info.get("winning_team"),
                "result": round_info.get("result"),
                "plant_site": plant_info["plant_site"],
                "plant_x": plant_info["plant_x"],
                "plant_y": plant_info["plant_y"],
                "plant_time_ms": plant_info["plant_time_ms"],
            }

        match_kill_counter = 0

        for kill in kill_data:
            if not kill:
                continue

            round_num = int(kill.get("round", 0))
            round_id = f"{match_id}_r{round_num:02d}"

            if round_id not in rounds_dict:
                rounds_dict[round_id] = {
                    "match_id": match_id,
                    "round_num": round_num,
                    "side": _infer_round_side(round_num),
                    "winner": None,
                    "result": None,
                    "plant_site": None,
                    "plant_x": None,
                    "plant_y": None,
                    "plant_time_ms": None,
                }

            killer_obj = kill.get("killer", {}) or {}
            victim_obj = kill.get("victim", {}) or {}

            killer = killer_obj.get("name", f"Player{match_kill_counter}")
            victim = victim_obj.get("name", f"Player{match_kill_counter + 1}")

            killer_puuid = killer_obj.get("puuid", player_puuid_map.get(killer, ""))
            victim_puuid = victim_obj.get("puuid", player_puuid_map.get(victim, ""))

            killer_agent = player_agent_map.get(killer, "Unknown")
            victim_agent = player_agent_map.get(victim, "Unknown")

            weapon_obj = kill.get("weapon", {}) or {}
            weapon = weapon_obj.get("name", "Unknown") if isinstance(weapon_obj, dict) else str(weapon_obj)

            # victim location (death location)
            kill_location = kill.get("location", {}) or {}
            victim_loc_x = kill_location.get("x")
            victim_loc_y = kill_location.get("y")

            # killer location (from player_locations)
            killer_loc_x = None
            killer_loc_y = None
            player_locations = kill.get("player_locations", []) or []

            for p in player_locations:
                player = p.get("player", {}) or {}
                if player.get("puuid") == killer_puuid:
                    loc = p.get("location", {}) or {}
                    killer_loc_x = loc.get("x")
                    killer_loc_y = loc.get("y")
                    break


            timestamp_ms = int(kill.get("time_in_round_in_ms", 0) or 0)
            timestamp_sec = timestamp_ms / 1000.0
            is_first_blood = 1 if kill.get("is_elimination_first_blood", False) else 0

            raw_site = kill.get("site")
            site = _safe_site(raw_site) or "Unknown"

            kill_id = f"{match_id}_kill_{match_kill_counter:06d}"
            cur.execute(
                "INSERT OR IGNORE INTO kills VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    kill_id, round_id, round_num, match_id,
                    killer, victim, killer_puuid, victim_puuid,
                    killer_agent, victim_agent,
                    site, weapon,
                    killer_loc_x, killer_loc_y,
                    victim_loc_x, victim_loc_y,
                    timestamp_sec, timestamp_ms, is_first_blood
                )
            )


            player_locations = kill.get("player_locations", []) or []
            for idx, player_loc in enumerate(player_locations):
                if not player_loc:
                    continue

                player_name = (player_loc.get("player", {}) or {}).get("name", f"Unknown_{idx}")
                player_puuid = (player_loc.get("player", {}) or {}).get("puuid", player_puuid_map.get(player_name, ""))
                player_team = player_team_map.get(player_name, "")
                location = player_loc.get("location", {}) or {}
                loc_x = location.get("x")
                loc_y = location.get("y")
                view_radians = player_loc.get("view_radians")

                location_id = f"{kill_id}_loc_{idx:02d}"
                cur.execute(
                    "INSERT OR IGNORE INTO player_locations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        location_id, kill_id, match_id, round_id, round_num,
                        timestamp_ms, player_puuid, player_name, player_team,
                        loc_x, loc_y, view_radians
                    )
                )

            total_kills += 1
            match_kill_counter += 1

        for round_id, round_info in rounds_dict.items():
            cur.execute("""
                INSERT OR REPLACE INTO rounds
                (round_id, match_id, round_num, side, winner, result, plant_site, plant_x, plant_y, plant_time_ms)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                round_id,
                round_info["match_id"],
                round_info["round_num"],
                round_info["side"],
                round_info["winner"],
                round_info["result"],
                round_info.get("plant_site"),
                round_info.get("plant_x"),
                round_info.get("plant_y"),
                round_info.get("plant_time_ms"),
            ))

    conn.commit()
    conn.close()
    print(f"[data_loader] Inserted {total_kills} kill events from {matches_processed} match(es).")


def compute_spawn_centroids(db_path: str = DB_PATH) -> dict:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    centroids = {}

    query = """
        SELECT m.map, r.side, k.killer_agent, k.killer_location_x, k.killer_location_y
        FROM kills k
        JOIN rounds r ON k.round_id = r.round_id
        JOIN matches m ON k.match_id = m.match_id
        WHERE k.killer_location_x IS NOT NULL AND k.killer_location_y IS NOT NULL
        ORDER BY m.map, r.side, k.killer_agent, k.timestamp_ms
    """

    for row in cur.execute(query):
        map_name = row["map"]
        side = row["side"]
        agent = row["killer_agent"]
        x = row["killer_location_x"]
        y = row["killer_location_y"]

        centroids.setdefault(map_name, {}).setdefault(side, {})
        if agent not in centroids[map_name][side]:
            centroids[map_name][side][agent] = {"x": x, "y": y}

    conn.close()
    return centroids


def generate_player_predictions(db_path: str = DB_PATH) -> None:
    """
    Generate predicted player positions using backwards-trace logic:
    
    For each attacker:
    1. Find their first kill location
    2. Determine which zone the kill happened in (using point-in-polygon)
    3. Trace BACKWARDS (south) through the adjacency graph to find starting position
    4. Use that starting position as the pred icted spawn location
    
    This logic is map-aware and uses bind_attacker_predictions.py for Bind
    (which has full topology). For other maps, use this fallback approach.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()


    cur.execute("""
        CREATE TABLE IF NOT EXISTS player_predictions (
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


    cur.execute("DELETE FROM player_predictions")
    conn.commit()

    # Get all rounds
    rounds_query = """
        SELECT m.match_id, m.map, r.round_id, r.round_num, r.side
        FROM   rounds  r
        JOIN   matches m ON m.match_id = r.match_id
        WHERE  m.map != 'Bind'
        ORDER  BY m.match_id, r.round_num
    """

    prediction_id_counter = 0
    predictions: list[tuple] = []

    for round_row in conn.execute(rounds_query):
        match_id  = round_row["match_id"]
        map_name  = round_row["map"]
        round_id  = round_row["round_id"]
        round_num = round_row["round_num"]
        side = round_row["side"]

        players_query = """
            SELECT DISTINCT
                pl.player_name,
                pl.team,
                k.killer_agent AS kill_agent
            FROM   player_locations pl
            JOIN   kills            k  ON pl.kill_id = k.kill_id
            WHERE  k.round_id = ?
        """
        players: dict[str, dict] = {}
        for p_row in conn.execute(players_query, (round_id,)):
            pname = p_row["player_name"]
            if pname not in players:
                agent = p_row["kill_agent"] or "Unknown"
                players[pname] = {"team": p_row["team"], "agent": agent}

        # Get first kill for each player (earliest timestamp)
        kills_query = """
            SELECT k.timestamp_ms,
                   k.killer,
                   k.killer_location_x,
                   k.killer_location_y
            FROM   kills k
            WHERE  k.round_id = ?
            ORDER  BY k.timestamp_ms
        """
        first_kill_per_player: dict[str, tuple[float, float]] = {}
        for k_row in conn.execute(kills_query, (round_id,)):
            pname = k_row["killer"]
            if pname not in first_kill_per_player:
                x = k_row["killer_location_x"]
                y = k_row["killer_location_y"]
                if x is not None and y is not None:
                    first_kill_per_player[pname] = (x, y)

        # Generate predictions for each player
        for pname, p_info in players.items():
            agent = p_info["agent"]
            team  = p_info["team"]

            # Determine spawn location: if we have kill data, infer from first kill
            # Otherwise use origin (0, 0) as fallback
            if pname in first_kill_per_player:
                kill_x, kill_y = first_kill_per_player[pname]
                spawn_x, spawn_y = kill_x, kill_y
                confidence = 0.90
            else:
                spawn_x, spawn_y = 0.0, 0.0
                confidence = 0.50

            # t=0-15s: spawn position
            pred_id = f"pred_{prediction_id_counter:06d}"
            prediction_id_counter += 1
            predictions.append((
                pred_id, match_id, round_id, round_num, pname, agent, team,
                "0-15", spawn_x, spawn_y, confidence,
            ))

            # t=15-30s: use same spawn position (simple model)
            for bucket in ["15-30"]:
                pred_id = f"pred_{prediction_id_counter:06d}"
                prediction_id_counter += 1
                predictions.append((
                    pred_id, match_id, round_id, round_num, pname, agent, team,
                    bucket, spawn_x, spawn_y, confidence * 0.8,
                ))

    cur.executemany(
        """
        INSERT OR IGNORE INTO player_predictions
            (prediction_id, match_id, round_id, round_num,
             player_name, agent, team,
             time_bucket, predicted_x, predicted_y, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        predictions,
    )
    conn.commit()
    conn.close()
    print(f"[data_loader] Generated {len(predictions)} player position predictions (non-Bind maps).")

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_kills(db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = """
        SELECT
            k.kill_id,
            k.round_id,
            k.match_id,
            k.killer,
            k.victim,
            k.killer_agent,
            k.site,
            k.timestamp_sec,
            k.is_first_blood,
            r.round_num,
            r.side,
            m.map
        FROM kills k
        JOIN rounds  r ON k.round_id  = r.round_id
        JOIN matches m ON k.match_id  = m.match_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    print(f"[data_loader] Loaded {len(df):,} kill events.")
    return df


def load_kills_for_players(players: list, last_n: int = 50, db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" * len(players))

    match_query = f"""
        SELECT DISTINCT k.match_id
        FROM kills k
        WHERE k.killer IN ({placeholders})
        ORDER BY k.match_id DESC
        LIMIT ?
    """
    match_ids = pd.read_sql_query(
        match_query, conn, params=players + [last_n]
    )["match_id"].tolist()

    if not match_ids:
        conn.close()
        return pd.DataFrame()

    mid_ph = ",".join("?" * len(match_ids))
    query = f"""
        SELECT
            k.kill_id,
            k.round_id,
            k.match_id,
            k.killer,
            k.victim,
            k.killer_agent,
            k.site,
            k.timestamp_sec,
            k.is_first_blood,
            r.round_num,
            r.side,
            m.map
        FROM kills k
        JOIN rounds  r ON k.round_id  = r.round_id
        JOIN matches m ON k.match_id  = m.match_id
        WHERE k.killer IN ({placeholders})
          AND k.match_id IN ({mid_ph})
    """
    df = pd.read_sql_query(query, conn, params=players + match_ids)
    conn.close()
    print(f"[data_loader] Loaded {len(df):,} kill events for {len(players)} player(s).")
    return df


def load_kill_locations(kill_id: str, db_path: str = DB_PATH) -> dict:
    conn = sqlite3.connect(db_path)

    kill_query = """
        SELECT killer, killer_location_x, killer_location_y,
               victim, victim_location_x, victim_location_y
        FROM kills
        WHERE kill_id = ?
    """
    kill_result = pd.read_sql_query(kill_query, conn, params=[kill_id])

    player_query = """
        SELECT player_name, location_x, location_y
        FROM player_locations
        WHERE kill_id = ?
    """
    players = pd.read_sql_query(player_query, conn, params=[kill_id])

    conn.close()

    if kill_result.empty:
        return None

    row = kill_result.iloc[0]
    return {
        "killer": {
            "name": row["killer"],
            "location_x": row["killer_location_x"],
            "location_y": row["killer_location_y"]
        },
        "victim": {
            "name": row["victim"],
            "location_x": row["victim_location_x"],
            "location_y": row["victim_location_y"]
        },
        "all_players": players.to_dict("records")
    }


def load_round_locations(round_id: str, db_path: str = DB_PATH) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = """
        SELECT pl.kill_id, pl.player_name, pl.location_x, pl.location_y
        FROM player_locations pl
        JOIN kills k ON pl.kill_id = k.kill_id
        WHERE k.round_id = ?
        ORDER BY k.timestamp_sec
    """
    df = pd.read_sql_query(query, conn, params=[round_id])
    conn.close()
    print(f"[data_loader] Loaded {len(df):,} player location records for round {round_id}.")
    return df