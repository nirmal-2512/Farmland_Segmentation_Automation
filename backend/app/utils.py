# ============================================================
# UTILITY FUNCTIONS FOR CONTOUR & GEOJSON
# ============================================================

import cv2
import numpy as np
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# ============================================================
# MASK CLEANING AND POSTPROCESSING
# ============================================================

def clean_boundary_mask(mask: np.ndarray) -> np.ndarray:
    """
    Notebook Cell 4
    Morphological refinement
    """

    if mask.dtype != np.uint8:
        mask = (mask * 255).astype(np.uint8)

    _, mask = cv2.threshold(
        mask,
        127,
        255,
        cv2.THRESH_BINARY
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5,5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    return mask


def draw_contours_overlay(image: np.ndarray, contours: List[np.ndarray], color=(0, 0, 255), thickness=2) -> np.ndarray:
    """
    Draw contours on a copy of the original image for debug overlay.
    """
    overlay = image.copy()
    cv2.drawContours(overlay, contours, -1, color, thickness)
    return overlay

def refine_field_regions(mask: np.ndarray):
    
    """
    Notebook Cell 5 + Cell 6 + Cell 7
    """

    mask = clean_boundary_mask(mask)

    # ---------------------------------------
    # Boundary -> Region
    # ---------------------------------------

    region = cv2.bitwise_not(mask)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        region,
        connectivity=8
    )

    output = np.zeros_like(region)

    MIN_AREA = 1000
    MAX_AREA = 500000

    current = 1

    filtered_labels = np.zeros_like(labels,dtype=np.int32)

    for label in range(1,num_labels):

        area = stats[label,cv2.CC_STAT_AREA]

        if area < MIN_AREA:
            continue

        if area > MAX_AREA:
            continue

        filtered_labels[labels==label]=current

        current+=1

    output = np.zeros_like(region)

    for label in np.unique(filtered_labels):

        if label==0:
            continue

        output[filtered_labels==label]=255

    logger.info(
        f"Detected {current-1} field regions"
    )

    return output


boundary_mask_to_field_regions = refine_field_regions# ============================================================
# CONTOUR EXTRACTION (Notebook Cell 8)
# ============================================================

def extract_contours(mask, min_area=1000):
    """
    Extract one contour for every connected field region.
    """

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Connected Components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    contours_list = []

    EPSILON_FACTOR = 0.002
    MAX_AREA = 500000

    for label in range(1, num_labels):

        area = stats[label, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        if area > MAX_AREA:
            continue

        region = np.zeros_like(mask)

        region[labels == label] = 255

        contours, _ = cv2.findContours(
            region,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if len(contours) == 0:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        epsilon = (
            EPSILON_FACTOR *
            cv2.arcLength(contour, True)
        )

        contour = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        contours_list.append(contour)

    logger.info(
        f"Extracted {len(contours_list)} polygons"
    )

    return contours_list


# ============================================================
# CONTOUR SIMPLIFICATION
# ============================================================

def simplify_contour(contour, epsilon_ratio=0.002):

    epsilon = epsilon_ratio * cv2.arcLength(
        contour,
        True
    )

    return cv2.approxPolyDP(
        contour,
        epsilon,
        True
    )


# ============================================================
# CONTOUR -> POLYGON
# ============================================================

def contour_to_polygon(contour):

    contour = simplify_contour(contour)

    polygon = []

    for p in contour:

        x = int(p[0][0])
        y = int(p[0][1])

        polygon.append([x, y])

    if len(polygon) < 3:
        return []

    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])

    return polygon

# ============================================================
# GEOJSON GENERATION (Notebook Cell 9)
# ============================================================

def contours_to_geojson(
    contours,
    image_width,
    image_height,
    crs="EPSG:4326"
):

    features = []

    field_id = 1

    for contour in contours:

        try:

            contour = simplify_contour(contour)

            polygon = []

            for p in contour:

                x = int(p[0][0])
                y = int(p[0][1])

                polygon.append([float(x), float(y)])

            if len(polygon) < 3:
                continue

            # Close polygon
            if polygon[0] != polygon[-1]:
                polygon.append(polygon[0])

            area = cv2.contourArea(
                np.array(polygon).astype(np.float32)
            )

            perimeter = cv2.arcLength(
                contour,
                True
            )

            x, y, w, h = cv2.boundingRect(contour)

            centroid_x = float(
                np.mean(
                    [p[0] for p in polygon[:-1]]
                )
            )

            centroid_y = float(
                np.mean(
                    [p[1] for p in polygon[:-1]]
                )
            )

            feature = {

                "type":"Feature",

                "id":field_id,

                "properties":{

                    "id":field_id,

                    "field_id":field_id,

                    "index":field_id,

                    "area_pixels":float(area),

                    "area":float(area),

                    "perimeter":float(perimeter),

                    "centroid":{

                        "x":centroid_x,

                        "y":centroid_y

                    },

                    "bbox":{

                        "x":int(x),

                        "y":int(y),

                        "width":int(w),

                        "height":int(h)

                    },

                    "image_dimensions":{

                        "width":image_width,

                        "height":image_height

                    }

                },

                "geometry":{

                    "type":"Polygon",

                    "coordinates":[polygon]

                }

            }

            features.append(feature)

            field_id += 1

        except Exception as e:

            logger.error(e)

    return {

        "type":"FeatureCollection",

        "crs":{

            "type":"name",

            "properties":{

                "name":crs

            }

        },

        "features":features

    }

# ============================================================
# PIXEL TO GEOGRAPHIC COORDINATES
# ============================================================

def pixel_to_geo(
    pixel_x: float,
    pixel_y: float,
    image_width: int,
    image_height: int,
    bounds: Dict[str, float]
) -> tuple:
    """
    Convert pixel coordinates to geographic coordinates
    using Web Mercator projection to match Google Maps exactly.
    """
    try:
        # Longitude is linear — safe to interpolate directly
        norm_x = pixel_x / image_width
        lon = bounds["west"] + norm_x * (bounds["east"] - bounds["west"])

        # Latitude is NOT linear on Mercator maps
        # Must convert bounds to Mercator Y, interpolate, then convert back

        def lat_to_mercator_y(lat_deg):
            lat_rad = np.radians(lat_deg)
            return np.log(np.tan(np.pi / 4 + lat_rad / 2))

        def mercator_y_to_lat(y):
            return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)

        north_y = lat_to_mercator_y(bounds["north"])
        south_y = lat_to_mercator_y(bounds["south"])

        norm_y = pixel_y / image_height
        mercator_y = north_y - norm_y * (north_y - south_y)
        lat = mercator_y_to_lat(mercator_y)

        return (lon, lat)

    except Exception as e:
        logger.error(f"Pixel to geo conversion error: {e}")
        return (0.0, 0.0)

# ============================================================
# GEOGRAPHIC DISTANCE AND AREA HELPERS
# ============================================================

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in meters between two points."""
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def polygon_area_geo(coords: List[List[float]]) -> float:
    """Approximate polygon area in square meters using spherical coordinates."""
    if len(coords) < 3:
        return 0.0

    total = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        total += np.radians(lon2 - lon1) * (
            np.sin(np.radians(lat1)) + np.sin(np.radians(lat2))
        )

    return abs(total) * (6371000.0 ** 2) / 2.0


def polygon_perimeter_geo(coords: List[List[float]]) -> float:
    """Approximate perimeter in meters for geographic polygon coordinates."""
    if len(coords) < 2:
        return 0.0

    perimeter = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        perimeter += haversine_distance(lat1, lon1, lat2, lon2)

    return perimeter

# ============================================================
# GEOJSON WITH GEOGRAPHIC COORDINATES
# ============================================================
def contours_to_geojson_geographic(
        contours,
        image_width,
        image_height,
        bounds,
        crs="EPSG:4326"
):

    features=[]

    field_id=1

    for contour in contours:

        contour=simplify_contour(contour)

        polygon=[]

        for p in contour:

            px=int(p[0][0])
            py=int(p[0][1])

            lon,lat=pixel_to_geo(

                px,

                py,

                image_width,

                image_height,

                bounds

            )

            polygon.append([lon,lat])

        if len(polygon)<3:
            continue

        if polygon[0]!=polygon[-1]:
            polygon.append(polygon[0])

        area=cv2.contourArea(contour)

        perimeter=cv2.arcLength(contour,True)

        feature={

            "type":"Feature",

            "id":field_id,

            "properties":{

                "id":field_id,

                "field_id":field_id,

                "area_pixels":float(area),

                "perimeter_pixels":float(perimeter)

            },

            "geometry":{

                "type":"Polygon",

                "coordinates":[polygon]

            }

        }

        features.append(feature)

        field_id+=1

    return{

        "type":"FeatureCollection",

        "crs":{

            "type":"name",

            "properties":{

                "name":crs

            }

        },

        "features":features

    }