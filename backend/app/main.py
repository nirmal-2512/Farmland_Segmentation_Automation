# ============================================================
# FASTAPI SERVER FOR SEGFORMER-B2 FARMLAND SEGMENTATION
# ============================================================

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2

from io import BytesIO
import logging
import os
import json
import re

from pathlib import Path
from datetime import datetime

import config
from .inference import ONNXInference

from .utils import (
    clean_boundary_mask,
    refine_field_regions,
    boundary_mask_to_field_regions,
    draw_contours_overlay,
    extract_contours,
    contours_to_geojson,
    contours_to_geojson_geographic
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="SegFormer-B2 Farmland Segmentation API",
    description="Satellite image farmland boundary detection and GeoJSON export",
    version="2.1.0"
)


# ============================================================
# CORS MIDDLEWARE
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# INITIALIZE MODEL
# ============================================================

ONNX_PATH = config.ONNX_MODEL_PATH

try:
    inference = ONNXInference(
        ONNX_PATH,
        providers=config.INFERENCE_PROVIDERS
    )

    logger.info("ONNX Model Loaded Successfully")

except Exception as e:
    logger.error(f"Failed to load ONNX model: {e}")
    inference = None


# ============================================================
# PROJECT PATHS
# ============================================================

# main.py is located at:
#
# Farmland_Segmentation_Automation/
# └── backend/
#     └── app/
#         └── main.py
#
# Therefore:
# parents[0] = app
# parents[1] = backend
# parents[2] = project root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = PROJECT_ROOT / "data"

TILES_ROOT = DATA_ROOT / "tiles"

JOBS_ROOT = DATA_ROOT / "jobs"

LOGS_ROOT = DATA_ROOT / "logs"


# Make sure required directories exist
TILES_ROOT.mkdir(parents=True, exist_ok=True)
JOBS_ROOT.mkdir(parents=True, exist_ok=True)
LOGS_ROOT.mkdir(parents=True, exist_ok=True)


logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Data root: {DATA_ROOT}")
logger.info(f"Tiles root: {TILES_ROOT}")


# ============================================================
# DEBUG HELPERS
# ============================================================

def _create_debug_dir() -> Path:
    """
    Create a timestamped debug output directory for each request.
    """

    base_dir = Path(__file__).resolve().parent

    debug_root = base_dir / "debug_outputs"

    debug_root.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = datetime.utcnow().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    run_dir = debug_root / f"run_{timestamp}"

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return run_dir


def _save_debug_images(
    run_dir: Path,
    original: np.ndarray,
    raw_mask: np.ndarray,
    mask_binary: np.ndarray,
    field_regions: np.ndarray
):
    """
    Save all intermediate images to the debug folder.

    Files:

    original.png
        Raw image received from Chrome Extension.

    raw_mask.png
        Raw model output scaled to 0-255.

    mask_binary.png
        Thresholded model mask.

    field_regions.png
        Field-region mask used for contour extraction.
    """

    # --------------------------------------------------------
    # 1. Original image
    # --------------------------------------------------------

    cv2.imwrite(
        str(run_dir / "original.png"),
        original
    )

    logger.info(
        "  Saved: original.png"
    )


    # --------------------------------------------------------
    # 2. Raw model output
    # --------------------------------------------------------

    raw_visual = (
        raw_mask * 255
    ).astype(np.uint8)

    cv2.imwrite(
        str(run_dir / "raw_mask.png"),
        raw_visual
    )

    logger.info(
        f"  Saved: raw_mask.png "
        f"[min={raw_mask.min():.3f} "
        f"max={raw_mask.max():.3f} "
        f"mean={raw_mask.mean():.3f}]"
    )


    # --------------------------------------------------------
    # 3. Binary mask
    # --------------------------------------------------------

    cv2.imwrite(
        str(run_dir / "mask_binary.png"),
        mask_binary
    )

    logger.info(
        "  Saved: mask_binary.png"
    )


    # --------------------------------------------------------
    # 4. Field regions
    # --------------------------------------------------------

    cv2.imwrite(
        str(run_dir / "field_regions.png"),
        field_regions
    )

    logger.info(
        "  Saved: field_regions.png"
    )


    logger.info(
        f"  Debug folder: {run_dir}"
    )


# ============================================================
# SAFE PATH HELPER
# ============================================================

def _safe_path_component(value: str, field_name: str) -> str:
    """
    Validate job_id / tile_id before using them as directory names.

    This prevents path traversal such as:

        ../../something

    or:

        ../

    """

    if value is None:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is required"
        )

    value = str(value).strip()

    if not value:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} cannot be empty"
        )

    if not re.fullmatch(
        r"[A-Za-z0-9_.-]+",
        value
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid {field_name}. "
                "Only letters, numbers, "
                "underscore, hyphen and dot are allowed."
            )
        )

    if value in {".", ".."}:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field_name}"
        )

    return value


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health_check():
    """
    Check server health and model availability.
    """

    if inference is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_path": ONNX_PATH
    }


# ============================================================
# AUTOMATION SCREENSHOT SAVE ENDPOINT
# ============================================================

@app.post("/automation/save-screenshot")
async def save_automation_screenshot(
    file: UploadFile = File(...),

    job_id: str = Form(...),

    tile_id: str = Form(...),

    requested_lat: float = Form(...),

    requested_lng: float = Form(...),

    actual_lat: float = Form(...),

    actual_lng: float = Form(...),

    zoom: float = Form(...),

    tile_north: float = Form(...),

    tile_south: float = Form(...),

    tile_east: float = Form(...),

    tile_west: float = Form(...),

    map_width: int = Form(0),

    map_height: int = Form(0)
):
    """
    Save a screenshot captured by the Chrome automation.

    The screenshot is stored inside:

        data/tiles/<job_id>/<tile_id>/screenshot.png

    Additional capture information is stored in:

        screenshot_metadata.json

    Existing tile metadata.json is updated when available.

    IMPORTANT:
    This endpoint only stores the screenshot and capture metadata.

    It does NOT perform model inference.

    Model inference will be connected after the
    screenshot-navigation stage is verified.
    """

    logger.info(
        "============================================================"
    )

    logger.info(
        "AUTOMATION SCREENSHOT REQUEST"
    )

    logger.info(
        f"job_id={job_id}"
    )

    logger.info(
        f"tile_id={tile_id}"
    )

    logger.info(
        f"requested center=({requested_lat}, {requested_lng})"
    )

    logger.info(
        f"actual center=({actual_lat}, {actual_lng})"
    )

    logger.info(
        f"zoom={zoom}"
    )


    # --------------------------------------------------------
    # Validate identifiers
    # --------------------------------------------------------

    job_id = _safe_path_component(
        job_id,
        "job_id"
    )

    tile_id = _safe_path_component(
        tile_id,
        "tile_id"
    )


    # --------------------------------------------------------
    # Validate tile bounds
    # --------------------------------------------------------

    if tile_north <= tile_south:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid tile bounds: "
                "north must be greater than south"
            )
        )

    if tile_east <= tile_west:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid tile bounds: "
                "east must be greater than west"
            )
        )


    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    try:

        contents = await file.read()

    except Exception as e:

        logger.error(
            f"Failed to read screenshot upload: {e}"
        )

        raise HTTPException(
            status_code=400,
            detail="Failed to read screenshot upload"
        )


    if not contents:

        raise HTTPException(
            status_code=400,
            detail="Screenshot file is empty"
        )


    # Decode image to verify that the uploaded file
    # is actually a valid image.

    nparr = np.frombuffer(
        contents,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        nparr,
        cv2.IMREAD_COLOR
    )


    if image is None:

        logger.error(
            "Uploaded screenshot could not be decoded"
        )

        raise HTTPException(
            status_code=400,
            detail="Invalid screenshot image"
        )


    image_height_actual, image_width_actual = image.shape[:2]


    logger.info(
        f"Screenshot dimensions: "
        f"{image_width_actual}x{image_height_actual}"
    )


    # --------------------------------------------------------
    # Create tile directory
    # --------------------------------------------------------

    tile_dir = (
        TILES_ROOT
        / job_id
        / tile_id
    )

    tile_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Screenshot path
    # --------------------------------------------------------

    screenshot_path = (
        tile_dir
        / "screenshot.png"
    )


    # --------------------------------------------------------
    # Save screenshot
    # --------------------------------------------------------

    with open(
        screenshot_path,
        "wb"
    ) as f:

        f.write(contents)


    logger.info(
        f"Screenshot saved: {screenshot_path}"
    )


    # --------------------------------------------------------
    # Calculate center error
    # --------------------------------------------------------

    latitude_error = (
        actual_lat
        - requested_lat
    )

    longitude_error = (
        actual_lng
        - requested_lng
    )


    center_error = float(
        (
            latitude_error ** 2
            +
            longitude_error ** 2
        ) ** 0.5
    )


    # --------------------------------------------------------
    # Capture metadata
    # --------------------------------------------------------

    capture_time = (
        datetime.utcnow()
        .isoformat()
        + "Z"
    )


    screenshot_metadata = {

        "job_id": job_id,

        "tile_id": tile_id,

        "captured_at_utc": capture_time,

        "screenshot": {
            "filename": "screenshot.png",
            "path": str(
                screenshot_path.relative_to(
                    PROJECT_ROOT
                )
            ),
            "width": image_width_actual,
            "height": image_height_actual,
            "uploaded_filename": file.filename
        },

        "requested_map_state": {

            "latitude": requested_lat,
            "longitude": requested_lng,
            "zoom": zoom
        },

        "actual_map_state": {

            "latitude": actual_lat,
            "longitude": actual_lng,
            "zoom": zoom
        },

        "center_error": {

            "latitude_error": latitude_error,

            "longitude_error": longitude_error,

            "euclidean_degree_error": center_error
        },

        "tile_bounds": {

            "north": tile_north,

            "south": tile_south,

            "east": tile_east,

            "west": tile_west
        },

        "map_viewport": {

            "width": map_width,

            "height": map_height
        },

        "status": "captured"
    }


    # --------------------------------------------------------
    # Save screenshot_metadata.json
    # --------------------------------------------------------

    screenshot_metadata_path = (
        tile_dir
        / "screenshot_metadata.json"
    )


    with open(
        screenshot_metadata_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            screenshot_metadata,
            f,
            indent=2
        )


    logger.info(
        "Screenshot metadata saved: "
        f"{screenshot_metadata_path}"
    )


    # --------------------------------------------------------
    # Update existing tile metadata.json
    # --------------------------------------------------------

    tile_metadata_path = (
        tile_dir
        / "metadata.json"
    )


    tile_metadata = {}


    if tile_metadata_path.exists():

        try:

            with open(
                tile_metadata_path,
                "r",
                encoding="utf-8"
            ) as f:

                tile_metadata = json.load(f)

        except Exception as e:

            logger.warning(
                "Could not read existing metadata.json: "
                f"{e}"
            )

            tile_metadata = {}


    # Update capture information

    tile_metadata["status"] = "captured"

    tile_metadata["capture"] = {

        "captured_at_utc": capture_time,

        "requested_center": {

            "lat": requested_lat,

            "lng": requested_lng
        },

        "actual_center": {

            "lat": actual_lat,

            "lng": actual_lng
        },

        "zoom": zoom,

        "center_error": center_error,

        "image_width": image_width_actual,

        "image_height": image_height_actual
    }


    # Preserve the existing structure if available.

    if "files" not in tile_metadata:

        tile_metadata["files"] = {}


    tile_metadata["files"]["screenshot"] = (
        "screenshot.png"
    )


    tile_metadata["files"][
        "screenshot_metadata"
    ] = "screenshot_metadata.json"


    # Save metadata

    try:

        with open(
            tile_metadata_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                tile_metadata,
                f,
                indent=2
            )

        logger.info(
            f"Tile metadata updated: "
            f"{tile_metadata_path}"
        )

    except Exception as e:

        logger.warning(
            "Failed to update metadata.json: "
            f"{e}"
        )


    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    logger.info(
        "AUTOMATION SCREENSHOT SAVED SUCCESSFULLY"
    )

    logger.info(
        "============================================================"
    )


    return JSONResponse(
        content={

            "success": True,

            "message": (
                "Screenshot saved successfully"
            ),

            "job_id": job_id,

            "tile_id": tile_id,

            "screenshot": str(
                screenshot_path.relative_to(
                    PROJECT_ROOT
                )
            ),

            "metadata": str(
                screenshot_metadata_path.relative_to(
                    PROJECT_ROOT
                )
            ),

            "dimensions": {

                "width": image_width_actual,

                "height": image_height_actual
            },

            "requested_center": {

                "lat": requested_lat,

                "lng": requested_lng
            },

            "actual_center": {

                "lat": actual_lat,

                "lng": actual_lng
            },

            "center_error": center_error
        }
    )


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = 0.25,
    return_mask: bool = False
):
    """
    Predict boundary mask and extract contours as GeoJSON.

    Parameters:
    - file: Satellite tile image (JPG, PNG)
    - threshold: Confidence threshold for mask (0-1)
    - return_mask: Whether to return base64 encoded mask image

    Returns:
    - geojson: GeoJSON FeatureCollection of boundary polygons
    - metadata: Image dimensions and processing info
    - mask (optional): Base64 encoded mask image
    """

    if inference is None:

        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )


    try:

        contents = await file.read()

        nparr = np.frombuffer(
            contents,
            np.uint8
        )

        image = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )


        if image is None:

            raise ValueError(
                "Invalid image file"
            )


        h, w = image.shape[:2]


        logger.info(
            f"Processing image: {w}x{h}"
        )


        mask, debug_info = inference.predict(
            image
        )


        logger.info(
            f"Prediction shape: {mask.shape}"
        )

        logger.info(
            f"Prediction stats: "
            f"min={mask.min():.4f}, "
            f"max={mask.max():.4f}, "
            f"mean={mask.mean():.4f}"
        )


        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        mask_binary = (
            mask > threshold
        ).astype(np.uint8) * 255


        # ----------------------------------------------------
        # Convert boundary mask to field regions
        # ----------------------------------------------------

        field_mask = (
            boundary_mask_to_field_regions(
                mask_binary
            )
        )


        # ----------------------------------------------------
        # Extract contours
        # ----------------------------------------------------

        contours = extract_contours(
            field_mask
        )


        logger.info(
            f"Found {len(contours)} contours"
        )


        # ----------------------------------------------------
        # Convert to pixel-space GeoJSON
        # ----------------------------------------------------

        geojson_data = contours_to_geojson(
            contours,
            image_width=w,
            image_height=h
        )


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        response_data = {

            "geojson": geojson_data,

            "metadata": {

                "image_width": w,

                "image_height": h,

                "num_contours": len(contours),

                "threshold": threshold,

                "filename": file.filename
            }
        }


        # ----------------------------------------------------
        # Optional mask
        # ----------------------------------------------------

        if return_mask:

            import base64

            _, buffer = cv2.imencode(
                ".png",
                field_mask
            )

            mask_b64 = base64.b64encode(
                buffer
            ).decode("utf-8")

            response_data["mask"] = mask_b64


        return JSONResponse(
            content=response_data
        )


    except Exception as e:

        logger.error(
            f"Prediction error: {str(e)}"
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Prediction failed: {str(e)}"
            )
        )


# ============================================================
# BATCH PREDICTION ENDPOINT
# ============================================================

@app.post("/predict-batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    threshold: float = 0.25
):
    """
    Predict on multiple images.

    Parameters:
    - files: List of satellite tile images
    - threshold: Confidence threshold for mask

    Returns:
    - results: List of prediction results for each image
    """

    if inference is None:

        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )


    results = []


    try:

        for file in files:

            contents = await file.read()

            nparr = np.frombuffer(
                contents,
                np.uint8
            )

            image = cv2.imdecode(
                nparr,
                cv2.IMREAD_COLOR
            )


            if image is None:

                results.append({

                    "filename": file.filename,

                    "status": "error",

                    "message": "Invalid image"
                })

                continue


            h, w = image.shape[:2]


            # ------------------------------------------------
            # Model inference
            # ------------------------------------------------

            mask, _ = inference.predict(
                image
            )


            # ------------------------------------------------
            # Threshold
            # ------------------------------------------------

            mask_binary = (
                mask > threshold
            ).astype(np.uint8) * 255


            # ------------------------------------------------
            # Clean mask
            # ------------------------------------------------

            mask_clean = clean_boundary_mask(
                mask_binary
            )


            # ------------------------------------------------
            # Extract contours
            # ------------------------------------------------

            contours = extract_contours(
                mask_clean
            )


            # ------------------------------------------------
            # Convert to GeoJSON
            # ------------------------------------------------

            geojson_data = contours_to_geojson(
                contours,
                image_width=w,
                image_height=h
            )


            results.append({

                "filename": file.filename,

                "status": "success",

                "geojson": geojson_data,

                "num_contours": len(contours),

                "dimensions": {

                    "width": w,

                    "height": h
                }
            })


        return JSONResponse(
            content={
                "results": results
            }
        )


    except Exception as e:

        logger.error(
            f"Batch prediction error: {str(e)}"
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Batch prediction failed: "
                f"{str(e)}"
            )
        )


# ============================================================
# GEOREFERENCED PREDICTION ENDPOINT
# ============================================================

@app.post("/predict-georef")
async def predict_georef(
    file: UploadFile = File(...),

    north: float = Form(...),

    south: float = Form(...),

    east: float = Form(...),

    west: float = Form(...),

    image_width: int = Form(...),

    image_height: int = Form(...),

    threshold: float = Form(0.25),

    debug: bool = Form(False),

    return_mask: bool = Form(False)
):
    """
    Predict boundaries and georeference polygons
    using map bounds.

    Always saves debug images to:

        backend/app/debug_outputs/

    """

    if inference is None:

        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )


    # --------------------------------------------------------
    # Validate geographic bounds
    # --------------------------------------------------------

    if north <= south:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid map bounds provided"
            )
        )


    if east <= west:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid map bounds provided"
            )
        )


    bounds = {

        "north": north,

        "south": south,

        "east": east,

        "west": west
    }


    try:

        # ====================================================
        # Read image
        # ====================================================

        contents = await file.read()


        nparr = np.frombuffer(
            contents,
            np.uint8
        )


        image = cv2.imdecode(
            nparr,
            cv2.IMREAD_COLOR
        )


        if image is None:

            raise ValueError(
                "Invalid image file"
            )


        h, w = image.shape[:2]


        logger.info(
            f"Processing image: {w}x{h}"
        )


        # ====================================================
        # Create debug folder
        # ====================================================

        run_dir = _create_debug_dir()


        logger.info(
            f"Debug folder created: {run_dir}"
        )


        # ====================================================
        # Model inference
        # ====================================================

        mask, debug_info = inference.predict(
            image
        )


        logger.info(
            f"Prediction shape: {mask.shape}"
        )


        logger.info(
            f"Prediction stats: "
            f"min={mask.min():.4f}, "
            f"max={mask.max():.4f}, "
            f"mean={mask.mean():.4f}"
        )


        # ====================================================
        # Apply threshold
        # ====================================================

        mask_binary = (
            mask > threshold
        ).astype(np.uint8) * 255


        # ====================================================
        # Extract field regions
        # ====================================================

        field_mask = (
            boundary_mask_to_field_regions(
                mask_binary
            )
        )


        # ====================================================
        # Save debug images
        # ====================================================

        _save_debug_images(

            run_dir=run_dir,

            original=image,

            raw_mask=mask,

            mask_binary=mask_binary,

            field_regions=field_mask
        )


        # ====================================================
        # Threshold variants
        # ====================================================

        if debug:

            for t in [
                0.35,
                0.45,
                0.50,
                0.60
            ]:

                t_mask = (
                    mask > t
                ).astype(np.uint8) * 255


                cv2.imwrite(

                    str(
                        run_dir
                        / f"threshold_{int(t * 100)}.png"
                    ),

                    t_mask
                )


            logger.info(
                "Saved threshold variants "
                "(debug=True)"
            )


        # ====================================================
        # Extract contours
        # ====================================================

        contours = extract_contours(
            field_mask
        )


        logger.info(
            f"Found {len(contours)} contours"
        )


        # ====================================================
        # Draw overlay
        # ====================================================

        overlay = draw_contours_overlay(

            image,

            contours,

            color=(0, 0, 255),

            thickness=2
        )


        cv2.imwrite(

            str(
                run_dir
                / "contours_overlay.png"
            ),

            overlay
        )


        logger.info(
            "Saved: contours_overlay.png"
        )


        # ====================================================
        # Convert contours to geographic GeoJSON
        # ====================================================

        geojson_data = (
            contours_to_geojson_geographic(

                contours,

                image_width=image_width,

                image_height=image_height,

                bounds=bounds
            )
        )


        # ====================================================
        # Response
        # ====================================================

        response_data = {

            "geojson": geojson_data,

            "metadata": {

                "image_width": w,

                "image_height": h,

                "map_width": image_width,

                "map_height": image_height,

                "bounds": bounds,

                "num_contours": len(contours),

                "threshold": threshold,

                "filename": file.filename,

                "debug_folder": str(run_dir)
            }
        }


        # ====================================================
        # Optional field mask
        # ====================================================

        if return_mask:

            import base64


            _, buffer = cv2.imencode(

                ".png",

                field_mask
            )


            mask_b64 = (
                base64.b64encode(
                    buffer
                ).decode("utf-8")
            )


            response_data["mask"] = (
                mask_b64
            )


        return JSONResponse(
            content=response_data
        )


    except Exception as e:

        logger.error(
            "Georeference prediction error: "
            f"{str(e)}"
        )


        raise HTTPException(

            status_code=400,

            detail=(
                f"Prediction failed: "
                f"{str(e)}"
            )
        )


# ============================================================
# MODEL INFO ENDPOINT
# ============================================================

@app.get("/info")
async def model_info():
    """
    Get model information.
    """

    if inference is None:

        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )


    return {

        "model_type": "SegFormer-B2",

        "input_format": "ONNX",

        "input_shape": [
            1,
            3,
            512,
            512
        ],

        "output_shape": [
            1,
            2,
            128,
            128
        ],

        "task": (
            "Farmland field region segmentation"
        ),

        "model_path": ONNX_PATH
    }


# ============================================================
# RUN SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn


    uvicorn.run(

        app,

        host=config.HOST,

        port=config.PORT,

        log_level=config.LOG_LEVEL.lower()
    )