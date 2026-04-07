"""
Example script showing how to load Valorant match data and query it.
"""

from pathlib import Path
import sqlite3

from data_loader import (
    load_from_json_file,
    prompt_for_file_selection,
    load_all_json_files,
    generate_player_predictions,
    load_kills,
    load_kill_locations,
    load_round_locations,
    DB_PATH
)


def clear_loaded_json_data():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for table in [
        "player_predictions",
        "player_locations",
        "kills",
        "rounds",
        "players",
        "matches",
    ]:
        cur.execute(f"DELETE FROM {table}")
    conn.commit()
    conn.close()
    print("Cleared loaded JSON data from matches, players, rounds, kills, player_locations, and player_predictions.")


if __name__ == "__main__":
    # Load from your JSON file
    print("\n=== Loading Valorant Match Data ===")
    print("Options:")
    print("  [1] Load all JSON files")
    print("  [2] Interactively select a file")
    print("  [3] Delete loaded JSON data from DB tables")
    print("  [4] Load 1 games from each map (Bind, Haven, Split) - 5 total")
    print("  [5] Load all Split map files (for testing)")
    print("  [q] Quit")

    choice = input("Select option (1, 2, 3, or q): ").strip().lower()

    if choice == 'q':
        print("Cancelled.")
        exit(0)
    elif choice == '1':
        load_all_json_files("json")
    elif choice == '2':
        prompt_for_file_selection("json")
    elif choice == '3':
        clear_loaded_json_data()
    elif choice == '4':
        bind_files = [
            "json/Bind/0c442f72-11ba-4929-a111-d24b14424fee.json"
        ]
        pearl_files = [
            "json/Pearl/0a8b2cd5-afe8-4dea-b10b-50e9a22950e1.json"
        ]
        split_files = [
            "json/Split/0cd106fa-e1bb-4f23-8e6b-24c179fecdd3.json"
        ]
        fracture_files = [
            "json/Fracture/4cf0eb05-0832-46f6-b43b-557645215d6d.json"
        ]
        breeze_files = [
            "json/Breeze/0aee7eec-87b8-44dc-9923-c3e5c40d4b2d.json"
        ]
        for path in bind_files + pearl_files + split_files + fracture_files + breeze_files:
            load_from_json_file(path)
    elif choice == '5':
        # Load all split files 
        split_files = list(Path('json/Split').glob('*.json'))
        for path in split_files:
            load_from_json_file(path)
    else:
        print("Invalid choice. Defaulting to 1 Bind file.")
        bind_files = [
            "json/Bind/0c442f72-11ba-4929-a111-d24b14424fee.json"
        ]
        # load_all_json_files("json")
        for path in bind_files:
            load_from_json_file(path)

    # Generate position predictions
    print("\n=== Generating Position Predictions ===")
    # # # generate_player_predictions()
    # AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

    # Load and display all kill events
    print("\n=== Loading all kill events ===")
    df_kills = load_kills()
    if not df_kills.empty:
        print(f"Loaded {len(df_kills)} kill events")
        print(df_kills.head(10))
    else:
        print("No kill events found in database.")

    # Example: Query a specific kill's location data
    print("\n=== Kill Location Details ===")
    kill_loc = load_kill_locations('kill_000003')
    if kill_loc:
        print(f"Killer: {kill_loc['killer']['name']} at ({kill_loc['killer']['location_x']}, {kill_loc['killer']['location_y']})")
        print(f"Victim: {kill_loc['victim']['name']}")
        print(f"All player positions: {len(kill_loc['all_players'])} recorded")

    # Example: Query round positioning data using first available round
    print("\n=== Round Positioning Data ===")
    if not df_kills.empty:
        first_round_id = df_kills.iloc[0]['round_id']
        round_positions = load_round_locations(first_round_id)
        if not round_positions.empty:
            print(f"Round has {len(round_positions)} player location records across all kills")
            print(round_positions.groupby('kill_id').size().describe())
        else:
            print(f"No player location records found for round {first_round_id}")
    else:
        print("No kill data available to query round positions")
