"""Text region detection and masking using OpenCV EAST text detector.

The EAST model weights are downloaded on first use.  If unavailable the
module gracefully skips text masking and returns the original image.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

EAST_MODEL_URL = "https://github.com/oyyd/frozen_east_text_detection.pb/raw/master/frozen_east_text_detection.pb"
_CACHE_DIR = Path("./model_cache/east")
_MODEL_PATH = _CACHE_DIR / "frozen_east_text_detection.pb"


def _ensure_east_weights() -> bool:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if _MODEL_PATH.exists():
        return True
    try:
        import urllib.request

        logger.info("Downloading EAST text detector weights …")
        urllib.request.urlretrieve(EAST_MODEL_URL, _MODEL_PATH)
        return True
    except Exception as exc:
        logger.warning("Could not download EAST model: %s. Text masking disabled.", exc)
        return False


def detect_text_regions(
    img_bgr: np.ndarray,
    confidence_threshold: float = 0.5,
    size: int = 320,
) -> List[Tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) bounding boxes for text regions."""
    if not _ensure_east_weights():
        return []

    h, w = img_bgr.shape[:2]
    new_h = (h // 32 + 1) * 32 if h % 32 != 0 else h
    new_w = (w // 32 + 1) * 32 if w % 32 != 0 else w
    resized = cv2.resize(img_bgr, (new_w, new_h))
    ratio_h = h / new_h
    ratio_w = w / new_w

    net = cv2.dnn.readNet(str(_MODEL_PATH))
    blob = cv2.dnn.blobFromImage(
        resized, 1.0, (new_w, new_h), (123.68, 116.78, 103.94), swapRB=True, crop=False
    )
    net.setInput(blob)
    scores, geometry = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    rows, cols = scores.shape[2], scores.shape[3]
    boxes = []

    for y in range(rows):
        sc = scores[0, 0, y]
        xdata0 = geometry[0, 0, y]
        xdata1 = geometry[0, 1, y]
        xdata2 = geometry[0, 2, y]
        xdata3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]

        for x in range(cols):
            if sc[x] < confidence_threshold:
                continue
            offset_x = x * 4.0
            offset_y = y * 4.0
            angle = angles[x]
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            bh = xdata0[x] + xdata2[x]
            bw = xdata1[x] + xdata3[x]
            end_x = int(offset_x + cos_a * xdata1[x] + sin_a * xdata2[x])
            end_y = int(offset_y - sin_a * xdata1[x] + cos_a * xdata2[x])
            start_x = int(end_x - bw)
            start_y = int(end_y - bh)

            x1 = int(start_x * ratio_w)
            y1 = int(start_y * ratio_h)
            x2 = int(end_x * ratio_w)
            y2 = int(end_y * ratio_h)
            boxes.append((x1, y1, x2 - x1, y2 - y1))

    return boxes


def mask_text_regions(
    image: Image.Image,
    fill_color: Tuple[int, int, int] = (0, 0, 0),
) -> Tuple[Image.Image, int]:
    """Black-box all detected text regions in a PIL image.

    Returns
    -------
    (masked_image, region_count)
    """
    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    boxes = detect_text_regions(img_bgr)

    if not boxes:
        return image, 0

    fill_bgr = (fill_color[2], fill_color[1], fill_color[0])
    for x, y, w, h in boxes:
        x, y = max(0, x), max(0, y)
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), fill_bgr, -1)

    result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb), len(boxes)
