"""
ONNX inference: letterbox preprocess, YOLO parse (v5/v8), NMS, reference detection.
Matches Flutter multi_model_service.dart logic.
"""
import os
import io
import urllib.request
import numpy as np
from PIL import Image
import onnxruntime as ort

from model_config import (
    MODELS,
    REFERENCE_ORDER,
    INPUT_SIZE,
    NMS_IOU_THRESHOLD,
    GITHUB_RELEASE,
)


def get_models_dir():
    d = os.environ.get("MODELS_DIR", os.path.join(os.path.dirname(__file__), "models"))
    os.makedirs(d, exist_ok=True)
    return d


def download_models():
    base = get_models_dir()
    for m in MODELS:
        path = os.path.join(base, m["key"])
        if os.path.isfile(path):
            continue
        url = f"{GITHUB_RELEASE}/{m['key']}"
        print(f"Downloading {m['key']} from GitHub...")
        urllib.request.urlretrieve(url, path)
        print(f"  -> {path}")
    return base


def letterbox(image: Image.Image, target_w: int, target_h: int):
    """Letterbox resize; returns (numpy CHW tensor, scale, pad_x, pad_y, orig_w, orig_h)."""
    w, h = image.size
    aspect = w / h
    target_aspect = target_w / target_h
    if aspect > target_aspect:
        new_w = target_w
        new_h = int(target_w / aspect)
    else:
        new_h = target_h
        new_w = int(target_h * aspect)
    scale = new_w / w
    pad_x = (target_w - new_w) / 2.0
    pad_y = (target_h - new_h) / 2.0
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), (114, 114, 114))
    canvas.paste(resized, (int(pad_x), int(pad_y)))
    arr = np.array(canvas, dtype=np.float32) / 255.0
    # HWC -> CHW
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)
    return arr, scale, pad_x, pad_y, float(w), float(h)


def load_sessions(models_dir: str):
    """Load ONNX sessions for all models."""
    sessions = {}
    input_names = {}
    output_names = {}
    for m in MODELS:
        path = os.path.join(models_dir, m["key"])
        if not os.path.isfile(path):
            continue
        sess = ort.InferenceSession(
            path,
            providers=["CPUExecutionProvider"],
            sess_options=ort.SessionOptions(),
        )
        iname = sess.get_inputs()[0].name
        oname = sess.get_outputs()[0].name
        sessions[m["key"]] = {"session": sess, "config": m}
        input_names[m["key"]] = iname
        output_names[m["key"]] = oname
    return sessions, input_names, output_names


def iou(box1, box2):
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2, w2, h2 = box2
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter = inter_w * inter_h
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def nms(detections, iou_threshold=0.45):
    """Non-max suppression by confidence, then IoU."""
    if not detections:
        return []
    dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep = []
    for i, d in enumerate(dets):
        suppressed = False
        for k in keep:
            if iou(d["boundingBox"], dets[k]["boundingBox"]) > iou_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(i)
    return [dets[i] for i in keep]


def _parse_yolo_single(
    flat,
    shape,
    class_names,
    confidence_threshold,
    source_key,
    detection_type,
    scale,
    pad_x,
    pad_y,
    orig_w,
    orig_h,
):
    """Parse YOLO output (v5 or v8 format)."""
    raw = []
    shape = list(shape)
    if len(shape) == 3:
        if shape[1] < shape[2]:
            values_per_pred = shape[1]
            num_predictions = shape[2]
            num_classes = len(class_names) if class_names else (values_per_pred - 4)
            is_v8 = True
        else:
            num_predictions = shape[1]
            values_per_pred = shape[2]
            num_classes = len(class_names) if class_names else (values_per_pred - 5)
            is_v8 = False
    else:
        num_classes = len(class_names) if class_names else 1
        values_per_pred = 5 + num_classes
        num_predictions = len(flat) // values_per_pred
        is_v8 = False

    for i in range(num_predictions):
        if is_v8:
            xc = flat[0 * num_predictions + i]
            yc = flat[1 * num_predictions + i]
            w = flat[2 * num_predictions + i]
            h = flat[3 * num_predictions + i]
            max_p = 0
            ci = 0
            for c in range(num_classes):
                idx = (4 + c) * num_predictions + i
                if idx < len(flat) and flat[idx] > max_p:
                    max_p = flat[idx]
                    ci = c
            conf = max_p
        else:
            off = i * values_per_pred
            if off + 4 >= len(flat):
                continue
            xc = flat[off + 0]
            yc = flat[off + 1]
            w = flat[off + 2]
            h = flat[off + 3]
            obj = flat[off + 4]
            max_p = 0
            ci = 0
            for c in range(num_classes):
                idx = off + 5 + c
                if idx < len(flat) and flat[idx] > max_p:
                    max_p = flat[idx]
                    ci = c
            conf = obj * max_p

        if conf < confidence_threshold:
            continue
        x1 = (xc - w / 2 - pad_x) / scale
        y1 = (yc - h / 2 - pad_y) / scale
        ww = w / scale
        hh = h / scale
        cx = max(0, min(x1, orig_w))
        cy = max(0, min(y1, orig_h))
        cw = max(0, min(ww, orig_w - cx))
        ch = max(0, min(hh, orig_h - cy))
        name = (class_names[ci] if class_names and ci < len(class_names) else f"class_{ci}")
        raw.append({
            "boundingBox": [float(cx), float(cy), float(cw), float(ch)],
            "confidence": float(conf),
            "classIndex": ci,
            "className": name,
            "source": source_key,
            "type": detection_type,
        })
    return nms(raw, NMS_IOU_THRESHOLD)


def run_inference_for_image(
    image_bytes: bytes,
    sessions,
    input_names,
    output_names,
    tire_diameter: float = 57.47,
    handle_width: float = 20.6,
    license_plate_width: float = 32.0,
):
    """Run all models on one image; return detections_by_model and reference_result."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor, scale, pad_x, pad_y, orig_w, orig_h = letterbox(
        image, INPUT_SIZE, INPUT_SIZE
    )
    detections_by_model = {}
    for key, data in sessions.items():
        sess = data["session"]
        config = data["config"]
        inp_name = input_names[key]
        out_name = output_names[key]
        out = sess.run([out_name], {inp_name: tensor})[0]
        flat = out.flatten().tolist()
        shape = list(out.shape)
        thresh = config["confidence"]
        class_names = config.get("class_names")
        dtype = config["type"]
        dets = _parse_yolo_single(
            flat, shape, class_names, thresh, key, dtype,
            scale, pad_x, pad_y, orig_w, orig_h,
        )
        detections_by_model[key] = dets

    reference_result = compute_reference(
        detections_by_model,
        tire_diameter,
        handle_width,
        license_plate_width,
        orig_w,
    )
    return detections_by_model, reference_result


def compute_reference(
    detections_by_model,
    tire_diameter,
    handle_width,
    license_plate_width,
    orig_w,
):
    """Compute reference scale from detections (priority: tire > handle > plate > headlight > fallback)."""
    def get_largest(detections):
        if not detections:
            return None
        return max(detections, key=lambda d: max(d["boundingBox"][2], d["boundingBox"][3]))

    def class_contains(d, words):
        cn = d["className"].lower()
        return any(w in cn for w in words)

    # Component model first
    comp = detections_by_model.get("car_component_detector.onnx") or []
    tires = [d for d in comp if class_contains(d, ["wheel", "tire"])]
    if tires:
        largest = get_largest(tires)
        if largest:
            px = max(largest["boundingBox"][2], largest["boundingBox"][3])
            return {"type": "tire", "scale": tire_diameter / px, "realSizeCm": tire_diameter}
    plates = [d for d in comp if class_contains(d, ["license", "plate"]) and d["confidence"] >= 0.65]
    if plates:
        largest = get_largest(plates)
        if largest:
            px = largest["boundingBox"][2]
            return {"type": "license_plate", "scale": license_plate_width / px, "realSizeCm": license_plate_width}
    headlights = [d for d in comp if class_contains(d, ["headlight"]) and d["confidence"] >= 0.30]
    if headlights:
        largest = get_largest(headlights)
        if largest:
            px = largest["boundingBox"][2]
            return {"type": "headlight", "scale": 33.0 / px, "realSizeCm": 33.0}

    # Handle
    handle = detections_by_model.get("handle_best.onnx") or []
    if handle:
        largest = get_largest(handle)
        if largest:
            px = largest["boundingBox"][2]
            return {"type": "handle", "scale": handle_width / px, "realSizeCm": handle_width}

    return {"type": "fallback", "scale": 100.0 / orig_w, "realSizeCm": 100.0}
