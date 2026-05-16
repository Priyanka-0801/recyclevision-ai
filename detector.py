from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# COCO class index for bottle is 39
BOTTLE_CLASS_ID = 39

def load_model():
    """Load YOLOv8 nano model (lightest, good for CPU/Colab)"""
    model = YOLO("yolov8n.pt")  # auto-downloads on first run
    return model

def detect_bottles(image: np.ndarray, model, conf_threshold: float = 0.3):
    """
    Run YOLOv8 on the image and return bottle detections.

    Args:
        image: numpy array (BGR or RGB)
        model: loaded YOLO model
        conf_threshold: minimum confidence to count as detection

    Returns:
        dict with count, detections, annotated image
    """
    results = model(image, conf=conf_threshold, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        if cls_id != BOTTLE_CLASS_ID:
            continue  # only bottles

        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # Crop the bottle region for downstream classification
        crop = image[y1:y2, x1:x2]

        detections.append({
            "confidence": round(conf, 3),
            "bbox": (x1, y1, x2, y2),
            "crop": crop,
            "needs_review": conf < 0.5  # flag low-confidence detections
        })

    # Annotated image for display
    annotated = result.plot()

    return {
        "bottle_count": len(detections),
        "detections": detections,
        "annotated_image": annotated,
        "raw_result": result
    }

def analyze_color(crop: np.ndarray) -> dict:
    """
    Analyze the dominant color of a cropped bottle region.
    Uses HSV color space for better color separation.

    Returns a color label and distribution.
    """
    if crop is None or crop.size == 0:
        return {"label": "unknown", "distribution": {}}

    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Low saturation = clear/white/grey
    clear_mask = s < 50
    clear_ratio = float(np.mean(clear_mask))

    # Color buckets by hue
    blue_mask = ((h >= 100) & (h <= 130)) & ~clear_mask
    green_mask = ((h >= 40) & (h <= 80)) & ~clear_mask
    brown_mask = ((h >= 10) & (h <= 30)) & ~clear_mask

    blue_ratio = float(np.mean(blue_mask))
    green_ratio = float(np.mean(green_mask))
    brown_ratio = float(np.mean(brown_mask))
    other_ratio = max(0.0, 1.0 - clear_ratio - blue_ratio - green_ratio - brown_ratio)

    distribution = {
        "clear": round(clear_ratio, 2),
        "blue": round(blue_ratio, 2),
        "green": round(green_ratio, 2),
        "brown": round(brown_ratio, 2),
        "other": round(other_ratio, 2)
    }

    # Dominant color label
    label = max(distribution, key=distribution.get)
    return {"label": label, "distribution": distribution}
