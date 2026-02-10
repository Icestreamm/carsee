"""
CarSee processing API: run ONNX damage/reference models in the cloud.
Deploy to Render.com; Flutter app sends photos and receives detections.
Uses one-model-at-a-time loading to stay under 512 MB RAM (Render free tier).
"""
import gc
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from inference import (
    get_models_dir,
    download_single_model,
    load_single_session,
    run_single_model_for_image,
    compute_reference,
    MODELS,
)

logger = logging.getLogger(__name__)
models_dir = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """No download at startup to avoid 502/OOM. Models are downloaded on first use per model."""
    global models_dir
    try:
        models_dir = get_models_dir()
        logger.info("Models dir ready: %s (models download on first request)", models_dir)
    except Exception as e:
        logger.exception("Failed to init: %s", e)
        raise
    yield
    models_dir = None


app = FastAPI(title="CarSee Processor", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"service": "CarSee Processor", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "models_ready": models_dir is not None}


# Allowed model keys for single-model endpoint (one at a time to avoid 502/OOM)
VALID_MODEL_KEYS = [m["key"] for m in MODELS]


@app.post("/run_single_model")
async def run_single_model(
    model_key: str = Form(...),
    images: list[UploadFile] = File(...),
    tire_diameter: float = Form(57.47),
    handle_width: float = Form(20.6),
    license_plate_width: float = Form(32.0),
):
    """
    Run exactly one model on all images. Call this endpoint once per model (7 times total)
    to avoid loading multiple models and reduce memory (fixes 502).
    Returns detections per image for this model only.
    """
    if models_dir is None:
        raise HTTPException(status_code=503, detail="Models not ready")
    if model_key not in VALID_MODEL_KEYS:
        raise HTTPException(status_code=400, detail=f"Invalid model_key. Use one of: {VALID_MODEL_KEYS}")
    if not images:
        raise HTTPException(status_code=400, detail="At least one image required")

    image_bytes_list = []
    for img_file in images:
        content = await img_file.read()
        image_bytes_list.append(content if len(content) > 0 else None)

    # Download this model only if missing (lazy; avoids startup OOM)
    try:
        download_single_model(model_key)
    except Exception as e:
        logger.warning("Download %s failed: %s", model_key, e)
        return {
            "model_key": model_key,
            "photos": [
                {"detections": [], "orig_w": 640.0, "orig_h": 480.0}
                for _ in image_bytes_list
            ],
        }

    loaded = load_single_session(models_dir, model_key)
    if loaded is None:
        return {
            "model_key": model_key,
            "photos": [
                {"detections": [], "orig_w": 640.0, "orig_h": 480.0}
                for _ in image_bytes_list
            ],
        }

    session, config, inp_name, out_name = loaded
    photos_result = []
    try:
        for img_bytes in image_bytes_list:
            if img_bytes is None:
                photos_result.append({"detections": [], "orig_w": 640.0, "orig_h": 480.0})
                continue
            try:
                dets, orig_w, orig_h = run_single_model_for_image(
                    session, config, inp_name, out_name, img_bytes, model_key
                )
                photos_result.append({
                    "detections": dets,
                    "orig_w": orig_w,
                    "orig_h": orig_h,
                })
            except Exception as e:
                logger.warning("run_single_model %s: %s", model_key, e)
                photos_result.append({"detections": [], "orig_w": 640.0, "orig_h": 480.0})
    finally:
        del session
    gc.collect()

    return {"model_key": model_key, "photos": photos_result}


@app.post("/process")
async def process(
    images: list[UploadFile] = File(...),
    tire_diameter: float = Form(57.47),
    handle_width: float = Form(20.6),
    license_plate_width: float = Form(32.0),
):
    """
    Process one or more car photos: run all ONNX models (one at a time to save RAM)
    and return detections per model + reference scale per photo.
    """
    if models_dir is None:
        raise HTTPException(status_code=503, detail="Models not ready")
    if not images:
        raise HTTPException(status_code=400, detail="At least one image required")

    # Read all image bytes
    image_bytes_list = []
    for img_file in images:
        content = await img_file.read()
        if len(content) == 0:
            image_bytes_list.append(None)
        else:
            image_bytes_list.append(content)

    # One result per image; we'll fill detectionsByModel and referenceResult
    results = [{"detectionsByModel": {}, "_orig_w": 640.0} for _ in image_bytes_list]

    # Run one model at a time, then drop it to free memory
    for m in MODELS:
        model_key = m["key"]
        try:
            download_single_model(model_key)
        except Exception as e:
            logger.warning("Download %s failed: %s", model_key, e)
        loaded = load_single_session(models_dir, model_key)
        if loaded is None:
            for r in results:
                r["detectionsByModel"][model_key] = []
            continue
        session, config, inp_name, out_name = loaded
        try:
            for i, img_bytes in enumerate(image_bytes_list):
                if img_bytes is None:
                    results[i]["detectionsByModel"][model_key] = []
                    continue
                try:
                    dets, orig_w, _ = run_single_model_for_image(
                        session, config, inp_name, out_name, img_bytes, model_key
                    )
                    results[i]["detectionsByModel"][model_key] = dets
                    results[i]["_orig_w"] = orig_w
                except Exception as e:
                    logger.warning("Model %s image %s: %s", model_key, i, e)
                    results[i]["detectionsByModel"][model_key] = []
        finally:
            del session
        gc.collect()

    # Compute reference for each image and remove temp key
    for i in range(len(results)):
        ref = compute_reference(
            results[i]["detectionsByModel"],
            tire_diameter,
            handle_width,
            license_plate_width,
            results[i]["_orig_w"],
        )
        del results[i]["_orig_w"]
        results[i]["referenceResult"] = ref

    return {"photos": results}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
