#!/usr/bin/env python3
"""Generate synthetic sample images for testing when real photos aren't available.

Creates simple geometric images that can be used to test the full pipeline
without needing actual photos.

Usage:
    python scripts/generate_sample_images.py
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent.parent))

OUTPUT_DIR = Path("./data/images")


def _text_image(text: str, bg: tuple, w: int = 400, h: int = 300) -> Image.Image:
    img = Image.new("RGB", (w, h), color=bg)
    draw = ImageDraw.Draw(img)
    draw.text((w // 4, h // 3), text, fill=(255, 255, 255))
    return img


def _gradient_image(color1: tuple, color2: tuple, w: int = 400, h: int = 300) -> Image.Image:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(w):
        t = i / w
        arr[:, i] = tuple(int(c1 * (1 - t) + c2 * t) for c1, c2 in zip(color1, color2))
    return Image.fromarray(arr, "RGB")


def _noisy_image(w: int = 400, h: int = 300, blur: bool = False) -> Image.Image:
    arr = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    if blur:
        import cv2
        img_cv = np.array(img)
        img_cv = cv2.GaussianBlur(img_cv, (15, 15), 0)
        img = Image.fromarray(img_cv)
    return img


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    burst_dir = OUTPUT_DIR / "sample_burst"
    burst_dir.mkdir(exist_ok=True)

    samples = [
        # (filename, image)
        ("beach_sunset.jpg", _gradient_image((255, 165, 0), (70, 130, 180))),
        ("beach_sunset2.jpg", _gradient_image((255, 140, 0), (60, 120, 170))),
        ("beach_sunset3.jpg", _gradient_image((230, 150, 0), (80, 140, 190))),
        ("dog_park.jpg", _gradient_image((34, 139, 34), (135, 206, 235))),
        ("dog_park2.jpg", _gradient_image((30, 130, 30), (140, 210, 240))),
        ("office_meeting.jpg", _text_image("Meeting Room", (50, 50, 100))),
        ("street_sign.jpg", _text_image("STOP\nMain St", (30, 30, 30))),
        ("family_photo.jpg", _text_image("Family Portrait", (100, 80, 60))),
        ("food_plate.jpg", _gradient_image((200, 100, 50), (220, 120, 80))),
        ("city_skyline.jpg", _gradient_image((10, 10, 40), (60, 60, 100))),
    ]

    # Burst group: sharp + blurry versions
    burst_samples = [
        ("burst/burst_sharp.jpg", _noisy_image(blur=False)),
        ("burst/burst_medium.jpg", _noisy_image(blur=True)),
        ("burst/burst_blurry.jpg", _noisy_image(blur=True)),
    ]

    for filename, img in samples + burst_samples:
        path = OUTPUT_DIR / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), "JPEG", quality=90)
        print(f"  Created {path}")

    print(f"\nGenerated {len(samples) + len(burst_samples)} sample images in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
