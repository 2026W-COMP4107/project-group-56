"""
BFS match collector for Valorant tactical ML project.

Output structure:
  json/
    cookie#tvoid/
      {match_id}.json
      {match_id}.json
    playername#tag/
      {match_id}.json
      ...

Flow per player:
  1. Fetch 10 competitive games * 5 maps = 50 API calls
  2. Save each match to json/{name#tag}/{match_id}.json
  3. Only AFTER all 50 games downloaded → enqueue opponents
  4. Never re-fetch a match_id already in that player's folder

Rate limit : 90 calls/minute  one call every 0.72s
Resume-safe: skips players already in done_players.txt
"""

import requests
import time
import json
import os
import re
from collections import deque
import dotenv

# == Config =====================================================================
BASE_URL     = "https://val-api-wrapper.up.railway.app/valorant/v4/matches/na/pc"
JSON_ROOT    = "json"
SEEN_FILE    = "seen_matches.txt"
DONE_FILE    = "done_players.txt"

SEED_NAME    = "cookie"
SEED_TAG     = "tvoid"

MAPS         = ["Breeze", "Bind", "Fracture", "Pearl", "Split"]
PER_MAP      = 20
EXPECTED     = len(MAPS) * PER_MAP          # 50 per player

RATE_LIMIT   = 90
CALL_INTERVAL = 60 / RATE_LIMIT + 0.05     # 0.72s for each

dotenv.load_dotenv() 
HENDRICK_API_KEY = os.getenv("HENDRICK_API_KEY")
# == Filesystem helpers =========================================================

def safe_name(name, tag):
    """Sanitize name#tag for use as a folder name."""
    raw = f"{name}#{tag}"
    return re.sub(r'[<>:"/\\|?*]', '_', raw)

def player_dir(name, tag):
    return os.path.join(JSON_ROOT, safe_name(name, tag))

def player_key(name, tag):
    return f"{name}#{tag}".lower()

def load_set(path):
    if not os.path.exists(path):
        return set()
    with open(path, encoding="utf-8") as f:
        return set(l.strip() for l in f if l.strip())

def append_line(path, value):
    with open(path, "a", encoding="utf-8") as f:
        f.write(value + "\n")

def saved_match_ids(name, tag):
    """Set of match IDs already on disk for this player."""
    d = player_dir(name, tag)
    os.makedirs(d, exist_ok=True)  # create folder if missing
    return {f[:-5] for f in os.listdir(d) if f.endswith(".json")}


# == API ========================================================================
_last_call = 0.0

def fetch(name, tag, map_name):
    """Fetch up to PER_MAP competitive matches for one player on one map."""
    global _last_call
    wait = CALL_INTERVAL - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)

    url = (f"{BASE_URL}/{name}/{tag}"
           f"?mode=competitive&size={PER_MAP}&map={map_name}")
    
    headers = {
        "Authorization": HENDRICK_API_KEY
    }

    try:
        r = requests.get(url, headers=headers, timeout=15)
        _last_call = time.time()

        if r.status_code == 429:
            print("    [429] rate limited — sleeping 60s")
            time.sleep(60)
            _last_call = time.time()
            return []
        if r.status_code != 200:
            print(f"    [!] HTTP {r.status_code}  {name}#{tag}  {map_name}")
            return []

        data = r.json()
        if isinstance(data, dict):
            data = data.get("data", [])
        return data if isinstance(data, list) else []

    except requests.exceptions.Timeout:
        print(f"    [!] timeout  {name}#{tag}  {map_name}")
        _last_call = time.time()
        return []
    except Exception as e:
        print(f"    [!] {e}")
        _last_call = time.time()
        return []

# == Save =======================================================================

def save_match(match, map_name):
    """Save match to json/{map_name}/{match_id}.json"""
    match_id = match.get("metadata", {}).get("match_id")
    if not match_id:
        return None

    d = os.path.join(JSON_ROOT, map_name)
    os.makedirs(d, exist_ok=True)

    path = os.path.join(d, f"{match_id}.json")
    if not os.path.exists(path):   # avoid overwriting
        with open(path, "w", encoding="utf-8") as f:
            json.dump(match, f, ensure_ascii=False)
    return match_id


def extract_players(match):
    out = []
    for p in match.get("players", []):
        n = p.get("name", "").strip()
        t = p.get("tag", "").strip()
        if n and t:
            out.append((n, t))
    return out

# == Process one player (download all 50 games first) ==========================

def process_player(name, tag, seen_matches, done_players):
    """
    Download 10 competitive games per map (50 total) for this player.
    Save matches by map, only enqueue opponents after all 50 games.
    """
    pkey = player_key(name, tag)
    opponents = []

    print(f"\n  [{name}#{tag}]")

    for map_name in MAPS:
        matches = fetch(name, tag, map_name)
        saved_now = 0

        for match in matches:
            mid = match.get("metadata", {}).get("match_id")
            if not mid or mid in seen_matches:
                continue

            save_match(match, map_name)
            saved_now += 1
            seen_matches.add(mid)
            append_line(SEEN_FILE, mid)

            # collect opponents for BFS queue (but enqueue later)
            for opp_name, opp_tag in extract_players(match):
                if player_key(opp_name, opp_tag) != pkey:
                    opponents.append((opp_name, opp_tag))

        total_matches = len([f for f in os.listdir(os.path.join(JSON_ROOT, map_name)) if f.endswith(".json")])
        print(f"    {map_name:10s}  +{saved_now:2d} new  ({total_matches} total)")

    # mark player as done after all 50 games
    done_players.add(pkey)
    append_line(DONE_FILE, pkey)
    print(f"    ✓ done — all 50 games saved for {name}#{tag}")

    return opponents

# == BFS ========================================================================

def collect():
    os.makedirs(JSON_ROOT, exist_ok=True)

    seen_matches = load_set(SEEN_FILE)
    done_players = load_set(DONE_FILE)

    queue = deque([(SEED_NAME, SEED_TAG)])
    queued = {player_key(SEED_NAME, SEED_TAG)}   # prevent duplicate enqueues

    print(f"[collector] seed      : {SEED_NAME}#{SEED_TAG}")
    print(f"[collector] maps      : {MAPS}")
    print(f"[collector] per player: {PER_MAP}/map * {len(MAPS)} maps = {EXPECTED} games")
    print(f"[collector] resuming  : {len(done_players)} players done, "
          f"{len(seen_matches)} matches seen\n")

    players_processed = 0

    while queue:
        name, tag = queue.popleft()

        if player_key(name, tag) in done_players:
            continue

        # == Download all 50 games for this player before touching opponents ==
        opponents = process_player(name, tag, seen_matches, done_players)
        players_processed += 1

        # == Now enqueue unique new opponents =================================
        new_enqueued = 0
        for opp_name, opp_tag in opponents:
            okey = player_key(opp_name, opp_tag)
            if okey not in done_players and okey not in queued:
                queue.append((opp_name, opp_tag))
                queued.add(okey)
                new_enqueued += 1

        print(f"    → enqueued {new_enqueued} opponents  |  queue: {len(queue)}")

    print(f"\n[collector] complete")
    print(f"  players processed : {players_processed}")
    print(f"  unique matches    : {len(seen_matches)}")
    print(f"  output folder     : {os.path.abspath(JSON_ROOT)}/")

if __name__ == "__main__":
    collect()

