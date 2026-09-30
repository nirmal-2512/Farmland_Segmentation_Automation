import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union
from pyproj import CRS


def generate_grid(
    geojson_path: str,
    tile_size_m: float = 500.0,
):
    """
    Generate a square coverage grid over a district boundary.

    The grid is generated in a local UTM CRS and returned in EPSG:4326.
    Only cells intersecting the district are retained.
    """

    # ---------------------------------------------------------
    # 1. Read district
    # ---------------------------------------------------------
    district = gpd.read_file(geojson_path)

    if district.empty:
        raise ValueError("District GeoJSON contains no features.")

    if district.geometry.is_empty.any():
        raise ValueError("District GeoJSON contains empty geometries.")

    if district.crs is None:
        raise ValueError("District GeoJSON has no CRS information.")

    # ---------------------------------------------------------
    # 2. Convert to WGS84
    # ---------------------------------------------------------
    district = district.to_crs("EPSG:4326")

    # Combine all district features
    district_geometry = unary_union(district.geometry)

    # ---------------------------------------------------------
    # 3. Select local UTM CRS
    # ---------------------------------------------------------
    centroid = district_geometry.centroid

    utm_zone = int((centroid.x + 180) / 6) + 1

    if centroid.y >= 0:
        epsg = 32600 + utm_zone
    else:
        epsg = 32700 + utm_zone

    projected_crs = CRS.from_epsg(epsg)

    # ---------------------------------------------------------
    # 4. Project district to meters
    # ---------------------------------------------------------
    district_projected = district.to_crs(projected_crs)

    district_geometry_projected = unary_union(
        district_projected.geometry
    )

    # ---------------------------------------------------------
    # 5. District bounds
    # ---------------------------------------------------------
    minx, miny, maxx, maxy = district_geometry_projected.bounds

    # ---------------------------------------------------------
    # 6. Generate grid
    # ---------------------------------------------------------
    cells = []

    x = minx
    column = 0

    while x < maxx:

        y = miny
        row = 0

        while y < maxy:

            cell = box(
                x,
                y,
                x + tile_size_m,
                y + tile_size_m,
            )

            # -------------------------------------------------
            # Only keep cells intersecting district
            # -------------------------------------------------
            if cell.intersects(district_geometry_projected):

                intersection = cell.intersection(
                    district_geometry_projected
                )

                cell_area = cell.area
                intersection_area = intersection.area

                coverage_ratio = (
                    intersection_area / cell_area
                )

                cells.append(
                    {
                        "tile_id": f"tile_{row:04d}_{column:04d}",
                        "row": row,
                        "column": column,
                        "tile_area_m2": cell_area,
                        "district_area_m2": intersection_area,
                        "coverage_ratio": coverage_ratio,
                        "geometry": cell,
                    }
                )

            y += tile_size_m
            row += 1

        x += tile_size_m
        column += 1

    # ---------------------------------------------------------
    # 7. Create GeoDataFrame
    # ---------------------------------------------------------
    grid = gpd.GeoDataFrame(
        cells,
        geometry="geometry",
        crs=projected_crs,
    )

    # ---------------------------------------------------------
    # 8. Convert to WGS84
    # ---------------------------------------------------------
    grid = grid.to_crs("EPSG:4326")

    return grid