from ultralytics import YOLO
from pathlib import Path

_yolo_model = None

APPLIANCE_COCO_CLASSES = {"microwave", "oven", "toaster", "refrigerator", "sink"}
SOURCE_MAP = {
    "microwave": "microwave_manual.md",
    "oven": "oven_manual.md",
    "toaster": "toaster_manual.md",
    "refrigerator": "refrigerator_manual.md",
}


def load_yolo_model():
    """Load the pretrained YOLOv8n model once at startup."""
    global _yolo_model
    _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def detect_appliance(image_path: str):
    """Run detection and return (appliance_name, confidence, manual_source) for the
    highest-confidence appliance detected, or (None, None, None) if none found."""
    if _yolo_model is None:
        raise RuntimeError("YOLO model not loaded. Call load_yolo_model() at startup.")

    results = _yolo_model(image_path, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_name = _yolo_model.names[int(box.cls)]
            conf = float(box.conf)
            if cls_name in APPLIANCE_COCO_CLASSES:
                detections.append((cls_name, conf))

    if not detections:
        return None, None, None

    detections.sort(key=lambda x: -x[1])
    top_class, top_conf = detections[0]
    return top_class, top_conf, SOURCE_MAP.get(top_class)
