from __future__ import annotations

from functools import lru_cache

import easyocr
import numpy as np
from PIL import Image

from config import Settings
from utils.preprocessing import preprocess_for_ocr


@lru_cache(maxsize=1)
def get_reader() -> easyocr.Reader:
    return easyocr.Reader(
        Settings.easyocr_languages,
        gpu=Settings.use_gpu_for_ocr,
        verbose=False,
    )


class OCRExtractor:
    def __init__(self) -> None:
        self.reader = get_reader()

    def extract_text(self, image: Image.Image) -> str:
        processed = preprocess_for_ocr(image)
        image_array = np.array(processed)
        results = self.reader.readtext(image_array, detail=0, paragraph=True)
        return " ".join(segment.strip() for segment in results if segment.strip()).strip()
