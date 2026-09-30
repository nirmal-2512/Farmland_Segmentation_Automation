import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "backend")
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

DISTRICT_FILE = (
    PROJECT_ROOT
    / "data"
    / "input"
    / "test1_Garoth.geojson"
)

GRID_FILE = (
    PROJECT_ROOT
    / "data"
    / "jobs"
    / "garoth_test"
    / "grid.geojson"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "jobs"
    / "garoth_test"
    / "grid_preview.png"
)


# ---------------------------------------------------------
# Load
# ---------------------------------------------------------

district = gpd.read_file(
    DISTRICT_FILE
)

grid = gpd.read_file(
    GRID_FILE
)

grid = grid.to_crs(
    district.crs
)


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(12, 12)
)

grid.plot(
    ax=ax,
    column="coverage_ratio",
    cmap="viridis",
    edgecolor="black",
    linewidth=0.25,
    legend=True,
    legend_kwds={
        "label": "District Coverage Ratio"
    },
)

district.boundary.plot(
    ax=ax,
    linewidth=2,
)


# ---------------------------------------------------------
# Tile IDs
# ---------------------------------------------------------

for _, tile in grid.iterrows():

    center = tile.geometry.centroid

    ax.text(
        center.x,
        center.y,
        str(tile["tile_id"])
        .replace("tile_", ""),
        fontsize=2.5,
        ha="center",
        va="center",
    )


# ---------------------------------------------------------
# Formatting
# ---------------------------------------------------------

ax.set_title(
    f"Garoth Coverage Grid — {len(grid)} Tiles"
)

ax.set_xlabel(
    "Longitude"
)

ax.set_ylabel(
    "Latitude"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
)

plt.close()

print(
    f"Grid preview saved: {OUTPUT_FILE}"
)