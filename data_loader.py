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

import sqlite3
import pandas as pd
import numpy as np
import os
import random
import json
from pathlib import Path
from datetime import datetime


# ── Constants ────────────────────────────────────────────────────────────────
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


conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cur.fetchall()

for table in tables:
    cur.execute(f"DROP TABLE IF EXISTS {table[0]};")

conn.commit()
conn.close()


def load_from_json_file(json_path: str, db_path: str = DB_PATH) -> None:
    """
    Loads Valorant match data from a JSON file into the SQLite database.
    
    Parameters
    ----------
    json_path : str
        Path to the JSON file containing Valorant API response data
    db_path : str
        Path to the SQLite database
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract the matches array from the JSON
        if isinstance(data, dict) and 'data' in data:
            matches = data['data']
        elif isinstance(data, list):
            matches = data
        else:
            matches = [data]
        
        _insert_matches_to_db(matches, db_path)
        print(f"[data_loader] Loaded data from JSON file '{json_path}'.")
    except Exception as e:
        print(f"[data_loader] Error loading JSON file: {e}")


### OPTION 1: Load from all TXT files in the json folder (commented out - uncomment to use)
### def load_from_all_json_files(json_folder: str = "json", db_path: str = DB_PATH) -> None:
###     """
###     Loads Valorant match data from all TXT files in the json folder into the SQLite database.
###     
###     Parameters
###     ----------
###     json_folder : str
###         Path to the folder containing JSON/TXT files
###     db_path : str
###         Path to the SQLite database
###     """
###     folder_path = Path(json_folder)
###     
###     if not folder_path.exists():
###         print(f"[data_loader] Error: Folder '{json_folder}' does not exist.")
###         return
###     
###     txt_files = list(folder_path.glob("*.txt"))
###     
###     if not txt_files:
###         print(f"[data_loader] No TXT files found in '{json_folder}'.")
###         return
###     
###     print(f"[data_loader] Found {len(txt_files)} TXT file(s) in '{json_folder}'.")
###     
###     for file_path in txt_files:
###         print(f"[data_loader] Loading file: {file_path.name}")
###         try:
###             with open(file_path, 'r') as f:
###                 data = json.load(f)
###             
###             if isinstance(data, dict) and 'data' in data:
###                 matches = data['data']
###             elif isinstance(data, list):
###                 matches = data
###             else:
###                 matches = [data]
###             
###             _insert_matches_to_db(matches, db_path)
###             print(f"[data_loader] Successfully loaded data from '{file_path.name}'.")
###         except Exception as e:
###             print(f"[data_loader] Error loading file '{file_path.name}': {e}")
###     
###     print(f"[data_loader] Completed loading all TXT files from '{json_folder}'.")


def prompt_for_file_selection(json_folder: str = "json", db_path: str = DB_PATH) -> None:
    """
    Prompts user to select a TXT file from the json folder and loads it into the database.
    Displays indexed file names for easy selection.
    
    Parameters
    ----------
    json_folder : str
        Path to the folder containing JSON/TXT files
    db_path : str
        Path to the SQLite database
    """
    folder_path = Path(json_folder)
    
    if not folder_path.exists():
        print(f"[data_loader] Error: Folder '{json_folder}' does not exist.")
        return
    
    txt_files = sorted(list(folder_path.glob("*.txt")))
    
    if not txt_files:
        print(f"[data_loader] No TXT files found in '{json_folder}'.")
        return
    
    print(f"\n[data_loader] Available files in '{json_folder}':")
    print("=" * 60)
    for idx, file_path in enumerate(txt_files, 1):
        print(f"  [{idx}] {file_path.name}")
    print("=" * 60)
    
    while True:
        try:
            selection = input("[data_loader] Enter the file number to load (or 'q' to quit): ").strip()
            
            if selection.lower() == 'q':
                print("[data_loader] Cancelled.")
                return
            
            file_idx = int(selection) - 1
            
            if file_idx < 0 or file_idx >= len(txt_files):
                print(f"[data_loader] Invalid selection. Please enter a number between 1 and {len(txt_files)}.")
                continue
            
            selected_file = txt_files[file_idx]
            
            print(f"[data_loader] Loading file: {selected_file.name}")
            try:
                with open(selected_file, 'r') as f:
                    data = json.load(f)
                
                if isinstance(data, dict) and 'data' in data:
                    matches = data['data']
                elif isinstance(data, list):
                    matches = data
                else:
                    matches = [data]
                
                _insert_matches_to_db(matches, db_path)
                print(f"[data_loader] Successfully loaded data from '{selected_file.name}'.")
            except Exception as e:
                print(f"[data_loader] Error loading file '{selected_file.name}': {e}")
            
            break
        except ValueError:
            print("[data_loader] Invalid input. Please enter a valid number or 'q' to quit.")


###
def load_from_api_response(api_response: dict, db_path: str = DB_PATH) -> None:
    """
    Loads Valorant match data from an API response into the SQLite database.
    
    Parameters
    ----------
    api_response : dict
        API response containing match data (e.g., from Valorant API)
    db_path : str
        Path to the SQLite database
    """
    matches = api_response.get('data', [api_response]) if isinstance(api_response, dict) else api_response
    _insert_matches_to_db(matches, db_path)
    print(f"[data_loader] Loaded {len(matches) if isinstance(matches, list) else 1} match(es) from API response.")


def _insert_matches_to_db(matches_data, db_path: str = DB_PATH) -> None:
    """
    Helper function to parse and insert match data into the database.
    Handles both JSON file and API response formats.
    
    Parameters
    ----------
    matches_data : list or dict
        Match data from JSON/API
    db_path : str
        Path to the SQLite database
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Ensure tables exist
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
            weapon            TEXT,
            killer_location_x REAL,
            killer_location_y REAL,
            victim_location_x REAL,
            victim_location_y REAL,
            kill_location_x   REAL,
            kill_location_y   REAL,
            site              TEXT,
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
    


    if not isinstance(matches_data, list):
        matches_data = [matches_data]
    
    kill_counter = 0
    total_kills = 0
    matches_processed = 0
    
    for match in matches_data:
        if not isinstance(match, dict):
            continue
            
        # Extract match-level data
        metadata = match.get('metadata', {})
        match_id = metadata.get('match_id', f"api_match_{kill_counter}")
        map_data = metadata.get('map', {})
        map_name = map_data.get('name', 'Unknown')
        started_at = metadata.get('started_at', '')
        
        # Skip matches not in current map pool
        if map_name not in MAPS:
            continue
        
        # Skip non-competitive matches (e.g., Deathmatch)
        queue = metadata.get('queue', {})
        queue_id = queue.get('id', '').lower()
        if queue_id != 'competitive':
            continue
        
        # Parse date from ISO format
        try:
            date_obj = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
            date = date_obj.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"[data_loader] Warning: Failed to parse date '{started_at}' for match_id '{match_id}': {e}. Using today's date as fallback.")
            date = datetime.today().strftime('%Y-%m-%d')
        
        # Insert match
        cur.execute(
            "INSERT OR IGNORE INTO matches VALUES (?,?,?)",
            (match_id, map_name, date)
        )
        matches_processed += 1
        
        # Process players and their kill data
        players = match.get('players', [])
        kill_data = match.get('kills', [])
        
        # Create player name -> agent mapping and insert players
        player_agent_map = {}
        player_puuid_map = {}
        player_team_map = {}
        
        for player in players:
            player_name = player.get('name', '')
            player_tag = player.get('tag', '')
            player_puuid = player.get('puuid', '')
            agent_name = player.get('agent', {}).get('name', 'Unknown')
            team_id = player.get('team_id', '')
            
            player_agent_map[player_name] = agent_name
            player_puuid_map[player_name] = player_puuid
            player_team_map[player_name] = team_id
            
            # Insert player
            cur.execute(
                "INSERT OR IGNORE INTO players VALUES (?,?,?,?,?,?)",
                (player_puuid, match_id, player_name, player_tag, team_id, agent_name)
            )
        
        # Create a round mapping based on kill data and round data from match
        rounds_dict = {}
        rounds_data = match.get('rounds', [])
        
        # Build rounds dict from explicit round data first
        for round_info in rounds_data:
            if not round_info:
                continue
            round_id_from_data = round_info.get('id')
            if round_id_from_data is not None:
                # Find the actual round_id in our format
                # Try to match with our generated round_ids
                pass
        
        for kill in kill_data:
            if not kill:  # Skip empty kill entries
                continue
                
            round_num = kill.get('round', 0)
            round_id = f"{match_id}_r{round_num:02d}"
            
            if round_id not in rounds_dict:
                # Guess which team won based on kill data
                winner = "team_a"  # Default
                rounds_dict[round_id] = {
                    'match_id': match_id,
                    'round_num': round_num,
                    'side': 'attack' if round_num < 12 else 'defense',
                    'winner': winner
                }
            
            # Extract kill details from nested structure
            killer_obj = kill.get('killer', {})
            victim_obj = kill.get('victim', {})
            
            killer = killer_obj.get('name', f"Player{kill_counter}")
            victim = victim_obj.get('name', f"Player{kill_counter + 1}")
            
            # Get PUUIDs
            killer_puuid = killer_obj.get('puuid', player_puuid_map.get(killer, ''))
            victim_puuid = victim_obj.get('puuid', player_puuid_map.get(victim, ''))
            
            # Look up agent from player data
            killer_agent = player_agent_map.get(killer, 'Unknown')
            victim_agent = player_agent_map.get(victim, 'Unknown')
            
            # Extract weapon
            weapon_obj = kill.get('weapon', {})
            weapon = weapon_obj.get('name', 'Unknown') if isinstance(weapon_obj, dict) else str(weapon_obj)
            
            # Extract locations (killer shot from, victim died at)
            killer_location = kill.get('location', {})
            killer_loc_x = killer_location.get('x', None)
            killer_loc_y = killer_location.get('y', None)
            
            victim_location = victim_obj.get('location', {})
            victim_loc_x = victim_location.get('x', None)
            victim_loc_y = victim_location.get('y', None)
            
            site = kill.get('site', 'Unknown')
            timestamp_ms = kill.get('time_in_round_in_ms', 0)
            timestamp_sec = timestamp_ms / 1000.0
            is_first_blood = 1 if kill.get('is_elimination_first_blood', False) else 0
            
            # Insert kill with all new fields
            kill_id = f"kill_{kill_counter:06d}"
            cur.execute(
                "INSERT OR IGNORE INTO kills VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (kill_id, round_id, round_num, match_id, killer, victim, killer_puuid, victim_puuid,
                 killer_agent, victim_agent, weapon,
                 killer_loc_x, killer_loc_y, victim_loc_x, victim_loc_y,
                 killer_loc_x, killer_loc_y,  # kill_location = killer_location
                 site, timestamp_sec, timestamp_ms, is_first_blood)
            )
            
            # Insert all player locations for this kill
            player_locations = kill.get('player_locations', [])
            for idx, player_loc in enumerate(player_locations):
                if not player_loc:
                    continue
                player_name = player_loc.get('player', {}).get('name', f'Unknown_{idx}')
                player_puuid = player_loc.get('player', {}).get('puuid', player_puuid_map.get(player_name, ''))
                player_team = player_team_map.get(player_name, '')
                location = player_loc.get('location', {})
                loc_x = location.get('x')
                loc_y = location.get('y')
                view_radians = player_loc.get('view_radians')
                
                location_id = f"loc_{kill_counter:06d}_{idx:02d}"
                cur.execute(
                    "INSERT OR IGNORE INTO player_locations VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (location_id, kill_id, match_id, round_id, round_num, timestamp_ms, 
                     player_puuid, player_name, player_team, loc_x, loc_y, view_radians)
                )
            
            kill_counter += 1
            total_kills += 1
        
        # Insert rounds with all fields
        for round_id, round_info in rounds_dict.items():
            cur.execute(
                "INSERT OR IGNORE INTO rounds VALUES (?,?,?,?,?,?,?,?,?)",
                (round_id, round_info['match_id'], round_info['round_num'], 
                 round_info['side'], round_info['winner'], 
                 round_info.get('result'), round_info.get('plant_x'), 
                 round_info.get('plant_y'), round_info.get('plant_time_ms'))
            )
    
    conn.commit()
    conn.close()
    print(f"[data_loader] Inserted {total_kills} kill events from {matches_processed} match(es).")


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_kills(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Loads all kill events joined with round metadata into a single DataFrame.

    Returns
    -------
    pd.DataFrame with columns:
        kill_id, round_id, match_id, killer, victim, killer_agent,
        site, timestamp_sec, is_first_blood,
        round_num, side, map
    """
    conn  = sqlite3.connect(db_path)
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


def load_kills_for_players(players: list,
                           last_n: int = 50,
                           db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Loads the most recent `last_n` matches for a specific list of players.

    Parameters
    ----------
    players : list of player identifiers (strings)
    last_n  : maximum number of matches to include per player
    db_path : path to the SQLite database

    Returns
    -------
    pd.DataFrame - same schema as load_kills()
    """
    conn        = sqlite3.connect(db_path)
    placeholders = ",".join("?" * len(players))

    # Grab the last `last_n` distinct match IDs for these players
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
    query  = f"""
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
    """
    Loads all location data for a specific kill event.
    
    Parameters
    ----------
    kill_id : str
        The kill ID to retrieve locations for
    db_path : str
        Path to the SQLite database
    
    Returns
    -------
    dict with 'killer', 'victim', and 'players' locations
    """
    conn = sqlite3.connect(db_path)
    
    # Get killer and victim locations
    kill_query = """
        SELECT killer, killer_location_x, killer_location_y,
               victim, victim_location_x, victim_location_y
        FROM kills
        WHERE kill_id = ?
    """
    kill_result = pd.read_sql_query(kill_query, conn, params=[kill_id])
    
    # Get all player locations
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
        'killer': {
            'name': row['killer'],
            'location_x': row['killer_location_x'],
            'location_y': row['killer_location_y']
        },
        'victim': {
            'name': row['victim'],
            'location_x': row['victim_location_x'],
            'location_y': row['victim_location_y']
        },
        'all_players': players.to_dict('records')
    }


def load_round_locations(round_id: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Loads all player locations for every kill in a specific round.
    Useful for positioning analysis and heatmaps.
    
    Parameters
    ----------
    round_id : str
        The round ID (e.g., "match_0000_r05")
    db_path : str
        Path to the SQLite database
    
    Returns
    -------
    pd.DataFrame with kill_id, player_name, location_x, location_y
    """
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
