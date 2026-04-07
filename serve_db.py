"""
serve_db.py
-----------
Serves the Valorant map viewer and exposes a small JSON API backed by
valorant.db so the frontend reads from the SQLite database instead of an
embedded RAW JavaScript object.

Run:
    python serve_db.py
Then open:
    http://localhost:8765/valorant_map_viewer_db.html
"""

from __future__ import annotations

import http.server
import json
import os
import sqlite3
import webbrowser
from pathlib import Path
from urllib.parse import urlparse

from map_adjacency_graph.map_zones import ZONE_POLYGONS

PORT = 8765
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "valorant.db"
VIEWER_FILE = ROOT / "valorant_map_viewer_db.html"
SUPPORTED_MAPS = ["Bind", "Split", "Breeze", "Fracture", "Pearl"]

# Conversion factors from the Valorant API (dash.valorant-api.com/endpoints/maps).
# Formula: normalized = multiplier * ingame_coord + scalarToAdd  →  result in [0, 1]
MAP_SCALARS: dict[str, dict[str, float]] = {
    "Bind":     {"xMultiplier": 5.9e-05,  "xScalarToAdd": 0.576941, "yMultiplier": -5.9e-05,  "yScalarToAdd": 0.967566},
    "Split":    {"xMultiplier": 7.8e-05,  "xScalarToAdd": 0.842188, "yMultiplier": -7.8e-05,  "yScalarToAdd": 0.697578},
    "Breeze":   {"xMultiplier": 7.0e-05,  "xScalarToAdd": 0.465123, "yMultiplier": -7.0e-05,  "yScalarToAdd": 0.833078},
    "Fracture": {"xMultiplier": 7.8e-05,  "xScalarToAdd": 0.556952, "yMultiplier": -7.8e-05,  "yScalarToAdd": 1.155886},
    "Pearl":    {"xMultiplier": 7.8e-05,  "xScalarToAdd": 0.480469, "yMultiplier": -7.8e-05,  "yScalarToAdd": 0.916016},
}

os.chdir(ROOT)


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row[1] for row in rows}


def pick_player_location_join(pl_cols: set[str]) -> tuple[str, str] | None:
    if {"source_type", "source_id"}.issubset(pl_cols):
        return "pl.source_type = 'kill' AND pl.source_id = k.kill_id", "pl.source_id"
    if "kill_id" in pl_cols:
        return "pl.kill_id = k.kill_id", "pl.kill_id"
    return None


def normalize_coord(x: float, y: float, scalars: dict) -> tuple[float, float]:
    """
    Convert in-game Valorant coordinates to normalized [0, 1] minimap coordinates
    using the official Valorant API multipliers and scalars.

    Formula (from dash.valorant-api.com/endpoints/maps):
        norm_x = xMultiplier * x + xScalarToAdd
        norm_y = yMultiplier * y + yScalarToAdd
    """
    nx = scalars["xMultiplier"] * x + scalars["xScalarToAdd"]
    ny = scalars["yMultiplier"] * y + scalars["yScalarToAdd"]
    return nx, ny


def load_player_predictions(db_path: Path = DB_PATH) -> dict:
    """
    Load player position predictions for time buckets 0-15 and 15-30 seconds.
    Returns predictions organized by map, match, round, and time bucket.
    Gets the actual agent from the players table using match_id and player_name.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        # Query player predictions for 0-15 and 15-30 time buckets
        # Join with players table to get the actual agent the player was using
        query = """
            SELECT
                pp.match_id,
                m.map,
                pp.round_num,
                pp.player_name,
                COALESCE(p.agent, 'Unknown') AS agent,
                pp.team,
                pp.time_bucket,
                pp.predicted_x,
                pp.predicted_y,
                pp.confidence
            FROM player_predictions pp
            JOIN matches m ON pp.match_id = m.match_id
            LEFT JOIN players p ON p.match_id = pp.match_id AND p.name = pp.player_name
            WHERE m.map IN ({})
            AND pp.time_bucket IN ('0-15', '15-30')
            ORDER BY m.map, pp.match_id, pp.round_num, pp.time_bucket, pp.player_name
        """.format(','.join('?' for _ in SUPPORTED_MAPS))

        predictions = {}
        
        for row in conn.execute(query, SUPPORTED_MAPS):
            map_name = row["map"]
            match_id = row["match_id"]
            round_num = str(row["round_num"])
            time_bucket = row["time_bucket"]
            
            scalars = MAP_SCALARS.get(map_name)
            if scalars is None:
                continue

            # Prediction coordinates are stored in raw in-game space in the DB,
            # so normalize them here before returning them to the viewer.
            nx, ny = normalize_coord(row["predicted_x"], row["predicted_y"], scalars)

            # Build nested structure
            if map_name not in predictions:
                predictions[map_name] = {"matches": {}}
            if match_id not in predictions[map_name]["matches"]:
                predictions[map_name]["matches"][match_id] = {"rounds": {}}
            if round_num not in predictions[map_name]["matches"][match_id]["rounds"]:
                predictions[map_name]["matches"][match_id]["rounds"][round_num] = {"buckets": {}}
            if time_bucket not in predictions[map_name]["matches"][match_id]["rounds"][round_num]["buckets"]:
                predictions[map_name]["matches"][match_id]["rounds"][round_num]["buckets"][time_bucket] = []
            
            # Add player prediction with actual agent from players table
            predictions[map_name]["matches"][match_id]["rounds"][round_num]["buckets"][time_bucket].append({
                "player": row["player_name"],
                "agent": row["agent"],
                "team": row["team"],
                "x": nx,
                "y": ny,
                "confidence": row["confidence"]
            })
        
        return predictions
    finally:
        conn.close()


def load_raw_from_db(db_path: Path = DB_PATH) -> dict:
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        required = {"matches", "rounds", "kills", "player_locations"}
        if not required.issubset(tables):
            missing = sorted(required - tables)
            raise RuntimeError(f"Database is missing required tables: {', '.join(missing)}")

        pl_cols = get_table_columns(conn, "player_locations")
        join_info = pick_player_location_join(pl_cols)
        if join_info is None:
            raise RuntimeError("player_locations table does not contain a supported join key")
        join_clause, _ = join_info

        player_cols = get_table_columns(conn, "players") if "players" in tables else set()
        has_players_table = bool(player_cols)

        kill_cols = get_table_columns(conn, "kills")
        site_col = "site" if "site" in kill_cols else None

        if has_players_table and {"puuid", "match_id", "agent"}.issubset(player_cols):
            player_join = "LEFT JOIN players p ON p.puuid = pl.puuid AND p.match_id = k.match_id"
            agent_expr = "COALESCE(p.agent, '')"
            team_expr = "COALESCE(pl.team, p.team_id, '')" if "team" in pl_cols else "COALESCE(p.team_id, '')"
            name_expr = "COALESCE(pl.player_name, p.name)"
        else:
            player_join = ""
            agent_expr = "COALESCE(pl.agent, '')" if "agent" in pl_cols else "''"
            team_expr = "COALESCE(pl.team, '')" if "team" in pl_cols else "''"
            name_expr = "pl.player_name" if "player_name" in pl_cols else "NULL"

        site_expr = f"COALESCE(k.{site_col}, '')" if site_col else "''"

        # Resolve victim agent/team from the players table when available
        if has_players_table and {"puuid", "match_id", "agent"}.issubset(player_cols):
            victim_agent_expr = "COALESCE(pv.agent, k.victim_agent, '')"
            victim_team_expr  = "COALESCE(pv.team_id, '')"
            victim_player_join = "LEFT JOIN players pv ON pv.puuid = k.victim_puuid AND pv.match_id = k.match_id"
        else:
            victim_agent_expr  = "COALESCE(k.victim_agent, '')" if "victim_agent" in kill_cols else "''"
            victim_team_expr   = "''"
            victim_player_join = ""

        kills_sql = f"""
            SELECT
                k.kill_id,
                k.match_id,
                m.map AS map_name,
                m.date AS match_date,
                r.round_num,
                k.killer,
                k.victim,
                k.weapon,
                {site_expr} AS site,
                COALESCE(k.timestamp_ms, CAST(k.timestamp_sec * 1000 AS INTEGER), 0) AS timestamp_ms,
                {name_expr} AS player_name,
                {agent_expr} AS agent,
                {team_expr} AS team,
                pl.location_x,
                pl.location_y,
                pl.view_radians,
                k.victim_location_x,
                k.victim_location_y,
                {victim_agent_expr} AS victim_agent,
                {victim_team_expr}  AS victim_team
            FROM kills k
            JOIN rounds r ON r.round_id = k.round_id
            JOIN matches m ON m.match_id = k.match_id
            LEFT JOIN player_locations pl ON {join_clause}
            {player_join}
            {victim_player_join}
            WHERE m.map IN ({','.join('?' for _ in SUPPORTED_MAPS)})
            ORDER BY m.map, k.match_id, r.round_num, timestamp_ms, k.kill_id, player_name
        """

        raw = {}
        kill_lookup = {}

        for row in conn.execute(kills_sql, SUPPORTED_MAPS):
            map_name = row["map_name"]
            match_id = row["match_id"]
            round_num = str(row["round_num"])
            scalars = MAP_SCALARS.get(map_name)

            if scalars is None:
                continue

            raw.setdefault(map_name, {"matches": {}})
            raw[map_name]["matches"].setdefault(match_id, {
                "rounds": {}
            })
            raw[map_name]["matches"][match_id]["rounds"].setdefault(round_num, [])

            round_kills = raw[map_name]["matches"][match_id]["rounds"][round_num]

            key = (map_name, match_id, row["round_num"], row["kill_id"])
            kill_obj = kill_lookup.get(key)

            if kill_obj is None:
                vic_entry = None
                if row["victim_location_x"] is not None and row["victim_location_y"] is not None:
                    vx, vy = normalize_coord(row["victim_location_x"], row["victim_location_y"], scalars)
                    vic_entry = {
                        "n": row["victim"],
                        "a": row["victim_agent"] or "Unknown",
                        "t": row["victim_team"] or "",
                        "x": vx,
                        "y": vy,
                        "r": None,
                    }

                kill_obj = {
                    "i": len(round_kills),
                    "k": row["killer"],
                    "v": row["victim"],
                    "ms": int(row["timestamp_ms"] or 0),
                    "s": row["site"] or "",
                    "w": row["weapon"],
                    "p": [],
                    "vic": vic_entry,
                }
                round_kills.append(kill_obj)
                kill_lookup[key] = kill_obj

            if (
                row["player_name"] is not None
                and row["location_x"] is not None
                and row["location_y"] is not None
            ):
                nx, ny = normalize_coord(row["location_x"], row["location_y"], scalars)
                kill_obj["p"].append({
                    "n": row["player_name"],
                    "a": row["agent"] or "Unknown",
                    "t": row["team"] or "",
                    "x": nx,
                    "y": ny,
                    "r": row["view_radians"],
                })

        raw = {
            map_name: {
                "matches": {
                    match_id: match_data
                    for match_id, match_data in map_data["matches"].items()
                    if match_data["rounds"]
                }
            }
            for map_name, map_data in raw.items()
            if map_data["matches"]
        }

        if not raw:
            raise RuntimeError("No kill/location data was found in valorant.db")

        return raw
    finally:
        conn.close()


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/raw":
            try:
                payload = load_raw_from_db()
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            return
        elif parsed.path == "/api/predictions":
            try:
                payload = load_player_predictions()
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            return
        elif parsed.path == "/api/zones":
            try:
                payload = {
                    map_name: zones
                    for map_name, zones in ZONE_POLYGONS.items()
                    if map_name in SUPPORTED_MAPS
                }
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            return
        elif parsed.path == "/api/debug-coords":
            # Debug endpoint showing 5 specific prediction coordinates
            payload = {
                "Bind": {
                    "matches": {
                        "debug-match": {
                            "rounds": {
                                "0": {
                                    "buckets": {
                                        "0-15": [
                                            {"player": "ookiemaru", "agent": "Neon", "team": "Blue",        "x": 0.7688, "y": 0.7688,"confidence": 0.95},
                                            {"player": "clover", "agent": "Sage", "team": "Blue",           "x": 1.2688, "y": 1.2688,"confidence": 0.95},
                                            {"player": "Water", "agent": "Brimstone", "team": "Blue",       "x": 1.3688, "y": 1.3688, "confidence": 0.95},
                                            {"player": "MasterMike", "agent": "Phoenix", "team": "Blue",    "x": 1.4688, "y": 1.4688, "confidence": 0.95},
                                            {"player": "clover2", "agent": "Skye", "team": "Blue",          "x": 1.5688, "y": 1.5688, "confidence": 0.95},
                                        ]
                                    }
                                }
                            }
                        }
                    }
                }
            }
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    def log_message(self, format: str, *args):
        if "/api/raw" in format or "404" in str(args) or "map_images" in str(args):
            super().log_message(format, *args)


if __name__ == "__main__":
    print(f"Starting Valorant Map Viewer at http://localhost:{PORT}")
    print(f"Serving from: {ROOT}")
    print(f"Reading match data from: {DB_PATH}")
    print(f"Open: http://localhost:{PORT}/{VIEWER_FILE.name}")
    print("Press Ctrl+C to stop.\n")

    webbrowser.open(f"http://localhost:{PORT}/{VIEWER_FILE.name}")

    with http.server.ThreadingHTTPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
