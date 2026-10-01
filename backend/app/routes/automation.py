from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, File, UploadFile, Form, HTTPException


router = APIRouter(
    prefix="/automation",
    tags=["automation"]
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_ROOT = PROJECT_ROOT / "data"

TILES_ROOT = DATA_ROOT / "tiles"


@router.post("/save-screenshot")
async def save_screenshot(
    file: UploadFile = File(...),
    job_id: str = Form(...),
    tile_id: str = Form(...),

    center_lat: float = Form(...),
    center_lon: float = Form(...),

    map_lat: float = Form(...),
    map_lon: float = Form(...),

    zoom: float = Form(...),

    north: float = Form(...),
    south: float = Form(...),
    east: float = Form(...),
    west: float = Form(...),

    map_width: int = Form(...),
    map_height: int = Form(...)
):
    """
    Save an automatically captured Google Maps screenshot.

    The screenshot is stored at:

    data/
        tiles/
            <job_id>/
                <tile_id>/
                    screenshot.png
                    screenshot_metadata.json
    """

    try:

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Screenshot filename is missing."
            )


        if not tile_id:
            raise HTTPException(
                status_code=400,
                detail="tile_id is required."
            )


        if not job_id:
            raise HTTPException(
                status_code=400,
                detail="job_id is required."
            )


        if north <= south:
            raise HTTPException(
                status_code=400,
                detail="Invalid north/south bounds."
            )


        if east <= west:
            raise HTTPException(
                status_code=400,
                detail="Invalid east/west bounds."
            )


        tile_dir = (
            TILES_ROOT /
            job_id /
            tile_id
        )


        tile_dir.mkdir(
            parents=True,
            exist_ok=True
        )


        screenshot_path = tile_dir / "screenshot.png"


        metadata_path = tile_dir / "screenshot_metadata.json"


        contents = await file.read()


        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Received empty screenshot."
            )


        screenshot_path.write_bytes(
            contents
        )


        metadata = {

            "job_id":
                job_id,

            "tile_id":
                tile_id,

            "captured_at":
                datetime.utcnow().isoformat() +
                "Z",

            "requested_center": {

                "lat":
                    center_lat,

                "lon":
                    center_lon
            },

            "actual_map_center": {

                "lat":
                    map_lat,

                "lon":
                    map_lon
            },

            "zoom":
                zoom,

            "bounds": {

                "north":
                    north,

                "south":
                    south,

                "east":
                    east,

                "west":
                    west
            },

            "image": {

                "width":
                    map_width,

                "height":
                    map_height,

                "filename":
                    file.filename,

                "size_bytes":
                    len(contents)
            },

            "files": {

                "screenshot":
                    "screenshot.png"
            },

            "status":
                "captured"
        }


        import json


        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2
            ),
            encoding="utf-8"
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

            "size_bytes":
                len(contents),

            "status":
                "captured"
        }


    except HTTPException:
        raise


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save screenshot: "
                f"{error}"
            )
        )