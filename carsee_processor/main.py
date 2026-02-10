"""
CarSee processing API: run ONNX damage/reference models in the cloud.
Deploy to Render.com; Flutter app sends photos and receives detections.
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import (
    download_models,
    load_sessions,
    run_inference_for_image,
)

logger = logging.getLogger(__name__)
sessions_cache = {}
input_names_cache = {}
output_names_cache = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ONNX models once at startup."""
    global sessions_cache, input_names_cache, output_names_cache
    try:
        models_dir = download_models()
        sessions_cache, input_names_cache, output_names_cache = load_sessions(models_dir)
        logger.info("Loaded %d ONNX models", len(sessions_cache))
    except Exception as e:
        logger.exception("Failed to load models: %s", e)
        raise
    yield
    sessions_cache.clear()
    input_names_cache.clear()
    output_names_cache.clear()


app = FastAPI(title="CarSee Processor", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": len(sessions_cache)}


@app.post("/process")
async def process(
    images: list[UploadFile] = File(...),
    tire_diameter: float = Form(57.47),
    handle_width: float = Form(20.6),
    license_plate_width: float = Form(32.0),
):
    """
    Process one or more car photos: run all ONNX models and return
    detections per model + reference scale per photo.
    """
    if not sessions_cache:
        raise HTTPException(status_code=503, detail="Models not loaded")
    if not images:
        raise HTTPException(status_code=400, detail="At least one image required")

    results = []
    for img_file in images:
        content = await img_file.read()
        if len(content) == 0:
            results.append({
                "detectionsByModel": {},
                "referenceResult": {"type": "fallback", "scale": 0.01, "realSizeCm": 100.0},
            })
            continue
        try:
            detections_by_model, reference_result = run_inference_for_image(
                content,
                sessions_cache,
                input_names_cache,
                output_names_cache,
                tire_diameter=tire_diameter,
                handle_width=handle_width,
                license_plate_width=license_plate_width,
            )
            results.append({
                "detectionsByModel": detections_by_model,
                "referenceResult": reference_result,
            })
        except Exception as e:
            logger.exception("Inference failed for %s: %s", img_file.filename, e)
            raise HTTPException(status_code=500, detail=str(e))

    return {"photos": results}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
