# ============================================================
# FASTAPI SERVER FOR SEGFORMER-B2 FARMLAND SEGMENTATION
# ============================================================

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2

import logging
import os
import json
import base64

from pathlib import Path
from datetime import datetime

from . import config
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="SegFormer-B2 Farmland Segmentation API",
    description=(
        "Satellite image farmland boundary detection "
        "and GeoJSON export"
    ),
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

    logger.info(
        "ONNX Model Loaded Successfully"
    )

except Exception as e:

    logger.error(
        f"Failed to load ONNX model: {e}"
    )

    inference = None


# ============================================================
# PROJECT PATHS
# ============================================================

# main.py:
#
# Farmland_Segmentation_Automation/
# └── backend/
#     └── app/
#         └── main.py
#
# parents[0] = app
# parents[1] = backend
# parents[2] = project root

APP_DIR = Path(__file__).resolve().parent

BACKEND_DIR = APP_DIR.parent

PROJECT_ROOT = BACKEND_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"

JOBS_DIR = DATA_DIR / "jobs"

TILES_DIR = DATA_DIR / "tiles"

MERGED_DIR = DATA_DIR / "merged"

OUTPUT_DIR = DATA_DIR / "output"

LOGS_DIR = DATA_DIR / "logs"


# Create important directories.
for directory in [
    DATA_DIR,
    JOBS_DIR,
    TILES_DIR,
    MERGED_DIR,
    OUTPUT_DIR,
    LOGS_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# DEBUG HELPERS
# ============================================================

def _create_debug_dir() -> Path:
    """
    Create a timestamped debug output directory.

    Example:

    backend/app/debug_outputs/
        run_20261001_123456_123456/
    """

    debug_root = (
        APP_DIR /
        "debug_outputs"
    )

    debug_root.mkdir(
        parents=True,
        exist_ok=True
    )

    timestamp = (
        datetime.utcnow()
        .strftime("%Y%m%d_%H%M%S_%f")
    )

    run_dir = (
        debug_root /
        f"run_{timestamp}"
    )

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
    Save intermediate inference images.

    Files:

    original.png
    raw_mask.png
    mask_binary.png
    field_regions.png
    """

    # --------------------------------------------------------
    # 1. Original image
    # --------------------------------------------------------

    cv2.imwrite(
        str(
            run_dir /
            "original.png"
        ),
        original
    )

    logger.info(
        "Saved debug image: original.png"
    )


    # --------------------------------------------------------
    # 2. Raw probability mask
    # --------------------------------------------------------

    raw_visual = (
        np.clip(
            raw_mask,
            0.0,
            1.0
        ) * 255
    ).astype(np.uint8)

    cv2.imwrite(
        str(
            run_dir /
            "raw_mask.png"
        ),
        raw_visual
    )

    logger.info(
        "Saved debug image: raw_mask.png "
        f"[min={raw_mask.min():.3f}, "
        f"max={raw_mask.max():.3f}, "
        f"mean={raw_mask.mean():.3f}]"
    )


    # --------------------------------------------------------
    # 3. Binary mask
    # --------------------------------------------------------

    cv2.imwrite(
        str(
            run_dir /
            "mask_binary.png"
        ),
        mask_binary
    )

    logger.info(
        "Saved debug image: mask_binary.png"
    )


    # --------------------------------------------------------
    # 4. Field regions
    # --------------------------------------------------------

    cv2.imwrite(
        str(
            run_dir /
            "field_regions.png"
        ),
        field_regions
    )

    logger.info(
        "Saved debug image: field_regions.png"
    )


# ============================================================
# AUTOMATION TILE DIRECTORY
# ============================================================

def _get_tile_dir(
    job_id: str | None,
    tile_id: str | None
) -> Path | None:
    """
    Return automation tile directory.

    Example:

    data/
        tiles/
            garoth_test/
                tile_0000_0000/
    """

    if not job_id or not tile_id:
        return None


    # Prevent accidental path traversal.
    safe_job_id = Path(
        job_id
    ).name

    safe_tile_id = Path(
        tile_id
    ).name


    tile_dir = (
        TILES_DIR /
        safe_job_id /
        safe_tile_id
    )


    tile_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    return tile_dir


# ============================================================
# SAVE TILE METADATA
# ============================================================

def _save_json(
    path: Path,
    data
):
    """
    Save JSON with readable formatting.
    """

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# SAVE AUTOMATION PREDICTION ARTIFACTS
# ============================================================

def _save_tile_prediction_artifacts(
    tile_dir: Path,
    image: np.ndarray,
    probability_mask: np.ndarray,
    binary_mask: np.ndarray,
    field_mask: np.ndarray,
    overlay: np.ndarray,
    geojson_data: dict,
    metadata: dict
):
    """
    Save all prediction artifacts for one automation tile.

    Expected directory:

    data/tiles/<job_id>/<tile_id>/

        input.png
        probability.png
        mask.png
        mask_refined.png
        contours_overlay.png
        prediction.geojson
        prediction_metadata.json
    """

    # --------------------------------------------------------
    # input.png
    # --------------------------------------------------------

    cv2.imwrite(
        str(
            tile_dir /
            "input.png"
        ),
        image
    )


    # --------------------------------------------------------
    # probability.png
    # --------------------------------------------------------

    probability_visual = (
        np.clip(
            probability_mask,
            0.0,
            1.0
        ) * 255
    ).astype(np.uint8)

    cv2.imwrite(
        str(
            tile_dir /
            "probability.png"
        ),
        probability_visual
    )


    # --------------------------------------------------------
    # mask.png
    # --------------------------------------------------------

    cv2.imwrite(
        str(
            tile_dir /
            "mask.png"
        ),
        binary_mask
    )


    # --------------------------------------------------------
    # mask_refined.png
    # --------------------------------------------------------

    cv2.imwrite(
        str(
            tile_dir /
            "mask_refined.png"
        ),
        field_mask
    )


    # --------------------------------------------------------
    # contours_overlay.png
    # --------------------------------------------------------

    cv2.imwrite(
        str(
            tile_dir /
            "contours_overlay.png"
        ),
        overlay
    )


    # --------------------------------------------------------
    # prediction.geojson
    # --------------------------------------------------------

    _save_json(
        tile_dir /
        "prediction.geojson",
        geojson_data
    )


    # --------------------------------------------------------
    # prediction_metadata.json
    # --------------------------------------------------------

    _save_json(
        tile_dir /
        "prediction_metadata.json",
        metadata
    )


    logger.info(
        f"Saved automation artifacts: {tile_dir}"
    )


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
# BASIC PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    threshold: float = 0.25,
    return_mask: bool = False
):
    """
    Predict farmland field regions.

    This endpoint preserves the existing non-georeferenced
    prediction behavior.
    """

    if inference is None:

        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )


    try:

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

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
            f"/predict image: {w}x{h}"
        )


        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        mask, debug_info = (
            inference.predict(
                image
            )
        )


        logger.info(
            f"Prediction shape: {mask.shape}"
        )

        logger.info(
            "Prediction stats: "
            f"min={mask.min():.4f}, "
            f"max={mask.max():.4f}, "
            f"mean={mask.mean():.4f}"
        )


        # ----------------------------------------------------
        # Threshold
        # ----------------------------------------------------

        mask_binary = (
            mask > threshold
        ).astype(
            np.uint8
        ) * 255


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

        contours = (
            extract_contours(
                field_mask
            )
        )


        logger.info(
            f"Found {len(contours)} contours"
        )


        # ----------------------------------------------------
        # Convert to pixel-space GeoJSON
        # ----------------------------------------------------

        geojson_data = (
            contours_to_geojson(
                contours,
                image_width=w,
                image_height=h
            )
        )


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        response_data = {

            "geojson":
                geojson_data,

            "metadata": {

                "image_width":
                    w,

                "image_height":
                    h,

                "num_contours":
                    len(contours),

                "threshold":
                    threshold,

                "filename":
                    file.filename
            }
        }


        # ----------------------------------------------------
        # Optional mask
        # ----------------------------------------------------

        if return_mask:

            _, buffer = cv2.imencode(
                ".png",
                field_mask
            )

            mask_b64 = (
                base64
                .b64encode(buffer)
                .decode("utf-8")
            )

            response_data["mask"] = (
                mask_b64
            )


        return JSONResponse(
            content=response_data
        )


    except Exception as e:

        logger.exception(
            "Prediction error"
        )

        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {str(e)}"
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
    """

    if inference is None:

        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )


    results = []


    try:

        for file in files:

            # ------------------------------------------------
            # Read image
            # ------------------------------------------------

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

                    "filename":
                        file.filename,

                    "status":
                        "error",

                    "message":
                        "Invalid image"
                })

                continue


            h, w = image.shape[:2]


            # ------------------------------------------------
            # Inference
            # ------------------------------------------------

            mask, _ = (
                inference.predict(
                    image
                )
            )


            mask_binary = (
                mask > threshold
            ).astype(
                np.uint8
            ) * 255


            mask_clean = (
                clean_boundary_mask(
                    mask_binary
                )
            )


            contours = (
                extract_contours(
                    mask_clean
                )
            )


            geojson_data = (
                contours_to_geojson(
                    contours,
                    image_width=w,
                    image_height=h
                )
            )


            results.append({

                "filename":
                    file.filename,

                "status":
                    "success",

                "geojson":
                    geojson_data,

                "num_contours":
                    len(contours),

                "dimensions": {

                    "width":
                        w,

                    "height":
                        h
                }
            })


        return JSONResponse(
            content={
                "results":
                    results
            }
        )


    except Exception as e:

        logger.exception(
            "Batch prediction error"
        )

        raise HTTPException(
            status_code=400,
            detail=(
                f"Batch prediction failed: "
                f"{str(e)}"
            )
        )


# ============================================================
# SAVE SCREENSHOT FOR AUTOMATION
# ============================================================

@app.post("/automation/save-screenshot")
async def automation_save_screenshot(
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
    Save the full browser screenshot and tile metadata.

    This endpoint is used by the Chrome automation.

    The full screenshot is saved as:

        data/tiles/<job_id>/<tile_id>/screenshot.png

    The actual map viewport crop is saved later as:

        input.png
    """

    try:

        # ----------------------------------------------------
        # Get tile directory
        # ----------------------------------------------------

        tile_dir = _get_tile_dir(
            job_id,
            tile_id
        )


        if tile_dir is None:

            raise ValueError(
                "Invalid job_id or tile_id"
            )


        # ----------------------------------------------------
        # Read screenshot
        # ----------------------------------------------------

        contents = await file.read()


        if not contents:

            raise ValueError(
                "Uploaded screenshot is empty"
            )


        # ----------------------------------------------------
        # Decode screenshot
        # ----------------------------------------------------

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
                "Uploaded file is not a valid image"
            )


        image_height, image_width = (
            image.shape[:2]
        )


        # ----------------------------------------------------
        # Save full screenshot
        # ----------------------------------------------------

        screenshot_path = (
            tile_dir /
            "screenshot.png"
        )


        success = cv2.imwrite(
            str(screenshot_path),
            image
        )


        if not success:

            raise IOError(
                "Failed to save screenshot.png"
            )


        # ----------------------------------------------------
        # Build metadata
        # ----------------------------------------------------

        metadata = {

            "job_id":
                job_id,

            "tile_id":
                tile_id,

            "requested_center": {

                "lat":
                    requested_lat,

                "lng":
                    requested_lng
            },

            "actual_center": {

                "lat":
                    actual_lat,

                "lng":
                    actual_lng
            },

            "zoom":
                zoom,

            "tile_bounds": {

                "north":
                    tile_north,

                "south":
                    tile_south,

                "east":
                    tile_east,

                "west":
                    tile_west
            },

            "map_dimensions": {

                "width":
                    map_width,

                "height":
                    map_height
            },

            "screenshot_dimensions": {

                "width":
                    image_width,

                "height":
                    image_height
            },

            "saved_at":
                datetime.utcnow()
                .isoformat()
        }


        # ----------------------------------------------------
        # Save metadata
        # ----------------------------------------------------

        metadata_path = (
            tile_dir /
            "metadata.json"
        )


        _save_json(
            metadata_path,
            metadata
        )


        logger.info(
            f"Saved screenshot: "
            f"{screenshot_path}"
        )


        return {

            "success":
                True,

            "job_id":
                job_id,

            "tile_id":
                tile_id,

            "screenshot":
                str(
                    screenshot_path
                ),

            "metadata":
                str(
                    metadata_path
                ),

            "dimensions": {

                "width":
                    image_width,

                "height":
                    image_height
            }
        }


    except Exception as e:

        logger.exception(
            "Automation screenshot save failed"
        )

        raise HTTPException(
            status_code=400,
            detail=(
                "Screenshot save failed: "
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

    return_mask: bool = Form(False),

    # --------------------------------------------------------
    # Automation metadata
    # --------------------------------------------------------

    job_id: str | None = Form(None),

    tile_id: str | None = Form(None)
):
    """
    Predict farmland boundaries and georeference them.

    The supplied geographic bounds MUST correspond to the
    actual image being processed.

    For Chrome automation:

        job_id = garoth_test
        tile_id = tile_0000_0000

    The endpoint saves normal FastAPI debug artifacts and,
    when job_id/tile_id are supplied, also saves artifacts
    directly under:

        data/tiles/<job_id>/<tile_id>/
    """

    # ========================================================
    # MODEL CHECK
    # ========================================================

    if inference is None:

        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )


    # ========================================================
    # VALIDATE BOUNDS
    # ========================================================

    if north <= south:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid map bounds: "
                "north must be greater than south"
            )
        )


    if east <= west:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid map bounds: "
                "east must be greater than west"
            )
        )


    # ========================================================
    # VALIDATE IMAGE DIMENSIONS
    # ========================================================

    if image_width <= 0:

        raise HTTPException(
            status_code=400,
            detail="image_width must be greater than 0"
        )


    if image_height <= 0:

        raise HTTPException(
            status_code=400,
            detail="image_height must be greater than 0"
        )


    # ========================================================
    # BUILD BOUNDS
    # ========================================================

    bounds = {

        "north":
            north,

        "south":
            south,

        "east":
            east,

        "west":
            west
    }


    # ========================================================
    # TILE DIRECTORY
    # ========================================================

    tile_dir = _get_tile_dir(
        job_id,
        tile_id
    )


    try:

        # ====================================================
        # READ IMAGE
        # ====================================================

        contents = await file.read()


        if not contents:

            raise ValueError(
                "Uploaded image is empty"
            )


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
            "========================================"
        )

        logger.info(
            "GEOREFERENCED PREDICTION"
        )

        logger.info(
            "========================================"
        )

        logger.info(
            f"Image dimensions: {w}x{h}"
        )

        logger.info(
            f"Requested map dimensions: "
            f"{image_width}x{image_height}"
        )

        logger.info(
            f"Bounds: {bounds}"
        )

        logger.info(
            f"job_id: {job_id}"
        )

        logger.info(
            f"tile_id: {tile_id}"
        )


        # ====================================================
        # SAVE INPUT TO AUTOMATION TILE DIRECTORY
        # ====================================================

        if tile_dir is not None:

            input_path = (
                tile_dir /
                "input.png"
            )


            cv2.imwrite(
                str(input_path),
                image
            )


            logger.info(
                f"Saved automation input: "
                f"{input_path}"
            )


        # ====================================================
        # CREATE DEBUG DIRECTORY
        # ====================================================

        run_dir = _create_debug_dir()


        logger.info(
            f"Debug folder: {run_dir}"
        )


        # ====================================================
        # MODEL INFERENCE
        # ====================================================

        mask, debug_info = (
            inference.predict(
                image
            )
        )


        logger.info(
            f"Prediction shape: "
            f"{mask.shape}"
        )


        logger.info(
            "Prediction stats: "
            f"min={mask.min():.4f}, "
            f"max={mask.max():.4f}, "
            f"mean={mask.mean():.4f}"
        )


        # ====================================================
        # THRESHOLD
        # ====================================================

        mask_binary = (
            mask > threshold
        ).astype(
            np.uint8
        ) * 255


        # ====================================================
        # FIELD REGIONS
        # ====================================================

        field_mask = (
            boundary_mask_to_field_regions(
                mask_binary
            )
        )


        # ====================================================
        # SAVE NORMAL DEBUG IMAGES
        # ====================================================

        _save_debug_images(
            run_dir=run_dir,

            original=image,

            raw_mask=mask,

            mask_binary=mask_binary,

            field_regions=field_mask
        )


        # ====================================================
        # OPTIONAL THRESHOLD VARIANTS
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
                ).astype(
                    np.uint8
                ) * 255


                cv2.imwrite(
                    str(
                        run_dir /
                        f"threshold_{int(t * 100)}.png"
                    ),
                    t_mask
                )


            logger.info(
                "Saved threshold variants"
            )


        # ====================================================
        # EXTRACT CONTOURS
        # ====================================================

        contours = (
            extract_contours(
                field_mask
            )
        )


        logger.info(
            f"Found {len(contours)} contours"
        )


        # ====================================================
        # DRAW CONTOUR OVERLAY
        # ====================================================

        overlay = (
            draw_contours_overlay(
                image,
                contours,
                color=(0, 0, 255),
                thickness=2
            )
        )


        overlay_path = (
            run_dir /
            "contours_overlay.png"
        )


        cv2.imwrite(
            str(overlay_path),
            overlay
        )


        logger.info(
            "Saved: contours_overlay.png"
        )


        # ====================================================
        # SAVE OVERLAY TO AUTOMATION TILE
        # ====================================================

        if tile_dir is not None:

            tile_overlay_path = (
                tile_dir /
                "contours_overlay.png"
            )


            cv2.imwrite(
                str(tile_overlay_path),
                overlay
            )


        # ====================================================
        # CONVERT TO GEOGRAPHIC GEOJSON
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
        # AUTOMATION ARTIFACTS
        # ====================================================

        prediction_metadata = {

            "job_id":
                job_id,

            "tile_id":
                tile_id,

            "image_width":
                w,

            "image_height":
                h,

            "map_width":
                image_width,

            "map_height":
                image_height,

            "bounds":
                bounds,

            "threshold":
                threshold,

            "num_contours":
                len(contours),

            "filename":
                file.filename,

            "debug":
                debug,

            "debug_folder":
                str(run_dir),

            "created_at":
                datetime.utcnow()
                .isoformat()
        }


        if tile_dir is not None:

            # -----------------------------------------------
            # prediction.geojson
            # -----------------------------------------------

            _save_json(
                tile_dir /
                "prediction.geojson",
                geojson_data
            )


            # -----------------------------------------------
            # probability.png
            # -----------------------------------------------

            probability_visual = (
                np.clip(
                    mask,
                    0.0,
                    1.0
                ) * 255
            ).astype(np.uint8)


            cv2.imwrite(
                str(
                    tile_dir /
                    "probability.png"
                ),
                probability_visual
            )


            # -----------------------------------------------
            # mask.png
            # -----------------------------------------------

            cv2.imwrite(
                str(
                    tile_dir /
                    "mask.png"
                ),
                mask_binary
            )


            # -----------------------------------------------
            # mask_refined.png
            # -----------------------------------------------

            cv2.imwrite(
                str(
                    tile_dir /
                    "mask_refined.png"
                ),
                field_mask
            )


            # -----------------------------------------------
            # prediction_metadata.json
            # -----------------------------------------------

            _save_json(
                tile_dir /
                "prediction_metadata.json",
                prediction_metadata
            )


            logger.info(
                "Saved all automation prediction "
                f"artifacts to: {tile_dir}"
            )


        # ====================================================
        # RESPONSE
        # ====================================================

        response_data = {

            "geojson":
                geojson_data,

            "metadata":
                prediction_metadata
        }


        # ====================================================
        # OPTIONAL MASK IN RESPONSE
        # ====================================================

        if return_mask:

            _, buffer = cv2.imencode(
                ".png",
                field_mask
            )


            mask_b64 = (
                base64
                .b64encode(
                    buffer
                )
                .decode(
                    "utf-8"
                )
            )


            response_data[
                "mask"
            ] = mask_b64


        return JSONResponse(
            content=response_data
        )


    except Exception as e:

        logger.exception(
            "Georeference prediction error"
        )


        raise HTTPException(
            status_code=400,
            detail=(
                "Prediction failed: "
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

        "model_type":
            "SegFormer-B2",

        "input_format":
            "ONNX",

        "input_shape":
            [1, 3, 512, 512],

        "output_shape":
            [1, 2, 128, 128],

        "task":
            "Farmland field region segmentation",

        "model_path":
            ONNX_PATH
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