# Model config: must match Flutter app (multi_model_service.dart, asset_download_service.dart)

GITHUB_RELEASE = "https://github.com/Icestreamm/carsee/releases/download/v1.0.0"

MODELS = [
    {"key": "car_component_detector.onnx", "confidence": 0.30, "type": "reference",
     "class_names": ["wheel", "tire", "license", "plate", "headlight", "taillight",
                    "door", "mirror", "bumper", "hood", "trunk", "window", "grille"]},
    {"key": "damage_detector_capstone.onnx", "confidence": 0.50, "type": "damage",
     "class_names": ["damage", "scratch", "dent", "crack", "broken_part"]},
    {"key": "damage_detector_CDDCE.onnx", "confidence": 0.50, "type": "damage",
     "class_names": ["damage", "front_damage", "rear_damage", "side_damage",
                    "windshield_damage", "light_damage", "tire_damage", "body_damage"]},
    {"key": "damage_detector_sindhu.onnx", "confidence": 0.50, "type": "damage",
     "class_names": ["dent", "scratch", "crack", "broken", "missing", "paint_damage",
                    "rust", "bent", "displaced", "shattered", "chip", "gouge",
                    "tear", "hole", "deformation", "scuff", "abrasion", "corrosion"]},
    {"key": "handle_best.onnx", "confidence": 0.50, "type": "reference",
     "class_names": ["handle"]},
    {"key": "side_detector_hunter.onnx", "confidence": 0.35, "type": "reference",
     "class_names": None},
    {"key": "side_detector_kulas.onnx", "confidence": 0.35, "type": "reference",
     "class_names": None},
]

# Order for reference detection priority (component first for tire/plate/headlight, then handle)
REFERENCE_ORDER = [
    "car_component_detector.onnx",
    "handle_best.onnx",
    "side_detector_hunter.onnx",
    "side_detector_kulas.onnx",
]

INPUT_SIZE = 640
NMS_IOU_THRESHOLD = 0.45
