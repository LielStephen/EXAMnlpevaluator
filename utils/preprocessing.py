from __future__ import annotations

from io import BytesIO
import re

import cv2
import numpy as np
from PIL import Image

from config import Settings


def load_image_from_bytes(file_bytes: bytes) -> Image.Image:
    image = Image.open(BytesIO(file_bytes))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def validate_image_size(file_size_bytes: int) -> None:
    size_mb = file_size_bytes / (1024 * 1024)
    if size_mb > Settings.max_image_size_mb:
        raise ValueError(
            f"Image is {size_mb:.2f} MB. Upload an image under {Settings.max_image_size_mb} MB."
        )


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    image_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=18)
    thresholded = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )

    height, width = thresholded.shape
    if max(height, width) < 1400:
        scale = 1400 / max(height, width)
        thresholded = cv2.resize(
            thresholded,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC,
        )

    return Image.fromarray(thresholded).convert("RGB")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def highlight_keywords(text: str, keywords: list[str], color: str) -> str:
    highlighted = text
    for keyword in sorted(keywords, key=len, reverse=True):
        if not keyword.strip():
            continue
        pattern = re.compile(rf"(?i)\b{re.escape(keyword)}\b")
        highlighted = pattern.sub(
            lambda match: f"<span style='background-color:{color};padding:0 4px;border-radius:4px;'>{match.group(0)}</span>",
            highlighted,
        )
    return highlighted
