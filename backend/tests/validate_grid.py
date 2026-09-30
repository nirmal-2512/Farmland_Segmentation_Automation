import sys
from pathlib import Path

import geopandas as gpd
from shapely.ops import unary_union


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


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

district = gpd.read_file(DISTRICT_FILE)
grid = gpd.read_file(GRID_FILE)

grid = grid.to_crs(district.crs)

district_geometry = unary_union(
    district.geometry
)


# ---------------------------------------------------------
# Basic information
# ---------------------------------------------------------

print()
print("========================================")
print("        GRID VALIDATION")
print("========================================")
print()

print("District CRS       :", district.crs)
print("District features  :", len(district))
print("Grid tiles         :", len(grid))


# ---------------------------------------------------------
# Geometry validation
# ---------------------------------------------------------

invalid_tiles = (
    ~grid.geometry.is_valid
).sum()

empty_tiles = (
    grid.geometry.is_empty
).sum()

print()
print("Invalid geometries :", invalid_tiles)
print("Empty geometries   :", empty_tiles)


# ---------------------------------------------------------
# Tile ID validation
# ---------------------------------------------------------

duplicate_ids = (
    grid["tile_id"].duplicated()
).sum()

print(
    "Duplicate tile IDs :",
    duplicate_ids
)


# ---------------------------------------------------------
# District coverage
# ---------------------------------------------------------

grid_geometry = unary_union(
    grid.geometry
)

intersection = grid_geometry.intersection(
    district_geometry
)

coverage_percentage = (
    intersection.area /
    district_geometry.area
) * 100

print()
print(
    "District coverage  :",
    f"{coverage_percentage:.4f}%"
)


# ---------------------------------------------------------
# Outside tiles
# ---------------------------------------------------------

outside_tiles = (
    ~grid.geometry.intersects(
        district_geometry
    )
).sum()

print(
    "Outside tiles      :",
    outside_tiles
)


# ---------------------------------------------------------
# Coverage ratio
# ---------------------------------------------------------

if "coverage_ratio" in grid.columns:

    print()
    print("Coverage ratio:")
    print(
        grid["coverage_ratio"].describe()
    )

    full_tiles = (
        grid["coverage_ratio"] >= 0.999
    ).sum()

    partial_tiles = (
        grid["coverage_ratio"] < 0.999
    ).sum()

    print()
    print("Full tiles         :", full_tiles)
    print("Partial tiles      :", partial_tiles)


# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------

print()
print("========================================")

if (
    invalid_tiles == 0
    and empty_tiles == 0
    and duplicate_ids == 0
    and outside_tiles == 0
    and coverage_percentage >= 99.99
):
    print("✓ GRID VALIDATION PASSED")
else:
    print("✗ GRID VALIDATION FAILED")

print("========================================")
print()