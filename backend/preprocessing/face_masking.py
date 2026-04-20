"""Face detection and blurring using OpenCV's DNN face detector.

Falls back to Haar-cascade if the DNN weights are not available locally.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# DNN face detector files – download once and cache locally
DNN_PROTO = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
DNN_MODEL = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

_CACHE_DIR = Path("./model_cache/face_detector")
_PROTO_PATH = _CACHE_DIR / "deploy.prototxt"
_MODEL_PATH = _CACHE_DIR / "res10_300x300_ssd_iter_140000.caffemodel"


def _ensure_dnn_weights() -> bool:
    """Download DNN face detector files if absent. Returns True on success."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if _PROTO_PATH.exists() and _MODEL_PATH.exists():
        return True
    try:
        import urllib.request

        logger.info("Downloading face detector prototxt …")
        urllib.request.urlretrieve(DNN_PROTO, _PROTO_PATH)
        logger.info("Downloading face detector model weights …")
        urllib.request.urlretrieve(DNN_MODEL, _MODEL_PATH)
        return True
    except Exception as exc:
        logger.warning("Could not download DNN face detector: %s. Falling back to Haar.", exc)
        return False


def detect_faces_dnn(
    img_bgr: np.ndarray, confidence_threshold: float = 0.5
) -> List[Tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) bounding boxes using SSD face detector."""
    _ensure_dnn_weights()
    if not (_PROTO_PATH.exists() and _MODEL_PATH.exists()):
        return detect_faces_haar(img_bgr)

    net = cv2.dnn.readNetFromCaffe(str(_PROTO_PATH), str(_MODEL_PATH))
    h, w = img_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img_bgr, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    boxes = []
    for i in range(detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf < confidence_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)
        boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


def detect_faces_haar(img_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Haar-cascade fallback (faster but less accurate)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return []
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def blur_faces(
    image: Image.Image,
    blur_strength: int = 25,
    use_dnn: bool = True,
) -> Tuple[Image.Image, int]:
    """Detect and blur all faces in a PIL image.

    Returns
    -------
    (masked_image, face_count)
    """
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    boxes = detect_faces_dnn(img_bgr) if use_dnn else detect_faces_haar(img_bgr)

    if not boxes:
        return image, 0

    k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1  # must be odd

    for x, y, w, h in boxes:
        x, y = max(0, x), max(0, y)
        roi = img_bgr[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        img_bgr[y : y + h, x : x + w] = blurred

    result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb), len(boxes)
