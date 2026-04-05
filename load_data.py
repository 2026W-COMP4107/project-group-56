"""
load_data.py
-----------
Example script showing how to load Valorant match data and query it.
"""

from data_loader import (
    load_from_json_file,
    prompt_for_file_selection,
    load_kills,
    load_kills_for_players,
    load_kill_locations,
    load_round_locations,
    DB_PATH
)

if __name__ == "__main__":
    # Load from your JSON file
    print("\n=== Loading from JSON file ===")
    ### Uncomment one of the following options:
    
    ### Option 1: Load from a specific file path (original method)
    # load_from_json_file("valorant_json.txt")
    
    ### Option 2: Prompt user to select a file from the json folder
    prompt_for_file_selection("json")
    
    # Load and display all kill events
    print("\n=== Loading all kill events ===")
    df_kills = load_kills()
    print(df_kills.head(10))
    
    # Load specific player data
    print("\n=== Loading specific player data ===")
    df_player = load_kills_for_players(["cookie"], last_n=50)
    print(df_player.head())
    
    # Example: Query a specific kill's location data
    print("\n=== Kill Location Details ===")
    kill_loc = load_kill_locations('kill_000003')
    if kill_loc:
        print(f"Killer: {kill_loc['killer']['name']} at ({kill_loc['killer']['location_x']}, {kill_loc['killer']['location_y']})")
        print(f"Victim: {kill_loc['victim']['name']}")
        print(f"All player positions: {len(kill_loc['all_players'])} recorded")
    
    # Example: Query round positioning data
    print("\n=== Round Positioning Data ===")
    round_positions = load_round_locations('0ff58254-5347-48b5-ba50-1b29ac898aad_r00')
    print(f"Round has {len(round_positions)} player location records across all kills")
    print(round_positions.groupby('kill_id').size().describe())
