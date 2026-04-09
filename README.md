# Valorant Execute Prediction and Map Viewer

This project analyzes Valorant match data, builds machine learning features, trains prediction models, and provides a browser-based map viewer for exploring round events and player position predictions.

## Project Overview

The repository includes tools for:

- loading and processing Valorant match data
- generating static and sequential features for modeling
- training machine learning models
- labeling rounds and storing results
- serving a local HTML map viewer backed by a SQLite database
- visualizing kills, player locations, zone polygons, and predicted player positions on supported maps

The viewer is served by `serve_db.py` and reads data from `valorant.db` through local JSON API endpoints.

## Main Files

- `serve_db.py` - local HTTP server for the map viewer and JSON API
- `valorant_map_viewer_db.html` - frontend map viewer UI
- `load_data.py` - data loading pipeline
- `data_loader.py` - helper utilities for loading data
- `features.py` - feature generation logic
- `label_rounds.py` - round labeling logic
- `train.py` - model training entry point
- `models.py` - model definitions / training helpers
- `attacker_predictions.py` - prediction-related utilities
- `map_adjacency_graph/map_zones.py` - map zone polygon definitions
- `data_collection/collect_matches.py` - match collection script

## Data and Assets

- `agent_icons/` - agent portraits used in the viewer
- `map_images/` - map images used as the viewer background
- `map_labels/` - zone label JSON files
- `features/` - saved NumPy feature arrays
- `results/` - trained model files and evaluation outputs

## Supported Maps in the Viewer

The current viewer is configured for:

- Bind
- Split
- Breeze
- Fracture
- Pearl

## Requirements

Python dependencies listed in `requirements.txt`:

- `pandas>=2.0`
- `numpy>=1.24`
- `scikit-learn>=1.3`
- `joblib>=1.3`

Install them with:

```bash
pip install -r requirements.txt
```

## Running the Map Viewer

Start the local server:

```bash
python serve_db.py
```

Then open:

```text
http://localhost:8765/valorant_map_viewer_db.html
```

The server exposes several local endpoints used by the frontend:

- `/api/raw` - kill and player location data from the SQLite database
- `/api/predictions` - player position predictions by map, match, round, and time bucket
- `/api/predict` - round-level execute prediction
- `/api/zones` - zone polygons for supported maps

## Viewer Features

The HTML viewer includes:

- map, match, round, and kill selection
- normal player view
- heatmap mode
- predicted player position mode
- debug coordinate mode
- map opacity control
- prediction time bucket switching (`0-15` and `15-30` seconds)
- hover coordinate inspection
- execute prediction summary card

## Expected Project Files

To use the viewer successfully, make sure these resources are present:

- `valorant.db`
- `valorant_map_viewer_db.html`
- `map_images/`
- `agent_icons/`

If map images are missing, the viewer will still run but will show a warning instead of the background map.

## Training / ML Workflow

A typical workflow for the ML side of the project is:

1. collect or load match data
2. generate labels and features
3. save processed arrays into `features/`
4. train models with `train.py`
5. inspect outputs in `results/`
6. serve predictions and explore them in the viewer

The repository already contains generated feature arrays and trained model artifacts in `features/` and `results/`.

## Notes

- The viewer is intended for local development and analysis.
- `serve_db.py` changes the working directory to the project root so static assets can be served correctly.
- Prediction coordinates used by the viewer are normalized on the backend before rendering in the frontend.

## Repository Structure

```text
project-group-56/
├── agent_icons/
├── data_collection/
├── features/
├── map_adjacency_graph/
├── map_images/
├── map_labels/
├── results/
├── attacker_predictions.py
├── data_loader.py
├── features.py
├── label_rounds.py
├── load_data.py
├── models.py
├── requirements.txt
├── serve_db.py
├── train.py
└── valorant_map_viewer_db.html
```

