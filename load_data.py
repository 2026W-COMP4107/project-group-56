"""
load_data.py
-----------
Example script showing how to load Valorant match data and query it.
"""

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

if __name__ == "__main__":
    # Load from your JSON file
    print("\n=== Loading Valorant Match Data ===")
    print("Options:")
    print("  [1] Load all JSON files from json/ folder")
    print("  [2] Interactively select a file")
    print("  [q] Quit")
    
    choice = input("Select option (1, 2, or q): ").strip().lower()
    
    if choice == 'q':
        print("Cancelled.")
        exit(0)
    elif choice == '1':
        load_all_json_files("json")
    elif choice == '2':
        prompt_for_file_selection("json")
    else:
        print("Invalid choice. Defaulting to load all files.")
        load_all_json_files("json")
    
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
