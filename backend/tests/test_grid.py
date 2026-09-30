import json
import sys
from pathlib import Path

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow imports from backend/
sys.path.insert(
    0,
    str(PROJECT_ROOT / "backend")
)

from app.services.grid import generate_grid


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "input"
    / "test1_Garoth.geojson"
)

JOB_ID = "garoth_test"

JOB_DIR = (
    PROJECT_ROOT
    / "data"
    / "jobs"
    / JOB_ID
)

TILES_DIR = (
    PROJECT_ROOT
    / "data"
    / "tiles"
    / JOB_ID
)


# ---------------------------------------------------------
# Create directories
# ---------------------------------------------------------

JOB_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TILES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Generate grid
# ---------------------------------------------------------

grid = generate_grid(
    str(INPUT_FILE),
    tile_size_m=500,
)

print(f"Total tiles: {len(grid)}")


# ---------------------------------------------------------
# Save grid GeoJSON
# ---------------------------------------------------------

grid_path = JOB_DIR / "grid.geojson"

grid.to_file(
    grid_path,
    driver="GeoJSON",
)

print(f"Grid saved: {grid_path}")


# ---------------------------------------------------------
# Create tile metadata
# ---------------------------------------------------------

tiles = []

for _, tile in grid.iterrows():

    geometry = tile.geometry

    minx, miny, maxx, maxy = geometry.bounds

    center = geometry.centroid

    tile_id = tile["tile_id"]

    tile_dir = TILES_DIR / tile_id

    tile_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    metadata = {
        "tile_id": tile_id,

        "row": int(tile["row"]),
        "column": int(tile["column"]),

        "center": {
            "lat": center.y,
            "lon": center.x,
        },

        "bounds": {
            "north": maxy,
            "south": miny,
            "east": maxx,
            "west": minx,
        },

        "area": {
            "tile_area_m2": float(
                tile["tile_area_m2"]
            ),
            "district_area_m2": float(
                tile["district_area_m2"]
            ),
            "coverage_ratio": float(
                tile["coverage_ratio"]
            ),
        },

        "status": "pending",

        "files": {
            "screenshot": None,
            "input": None,
            "probability": None,
            "mask": None,
            "mask_refined": None,
            "polygon_raw": None,
            "polygon_refined": None,
            "prediction": None,
        },
    }

    metadata_path = (
        tile_dir / "metadata.json"
    )

    with open(
        metadata_path,
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    tiles.append(metadata)


# ---------------------------------------------------------
# Save all tile metadata
# ---------------------------------------------------------

tiles_path = JOB_DIR / "tiles.json"

with open(
    tiles_path,
    "w"
) as f:
    json.dump(
        tiles,
        f,
        indent=2
    )


print(
    f"Tile metadata saved: {tiles_path}"
)

print(
    f"Tile directories: {TILES_DIR}"
)