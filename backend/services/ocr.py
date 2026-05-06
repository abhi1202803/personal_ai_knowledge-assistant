"""OCR helpers for extracting text from uploaded chat images."""

from __future__ import annotations

import base64
import io
import logging
from functools import lru_cache

import numpy as np
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_ocr_engine() -> RapidOCR:
    """Create the OCR engine lazily because model loading is relatively expensive."""

    return RapidOCR()


def extract_text_from_base64_image(image_data: str) -> str:
    """Extract text from a base64-encoded image.

    The frontend sends raw base64, but this also accepts a full data URL.
    Returns an empty string when OCR finds no text.
    """

    if not image_data:
        return ""

    if "," in image_data and image_data.lstrip().startswith("data:"):
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_data, validate=False)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        logger.warning("Could not decode image for OCR: %s", exc)
        return ""

    try:
        result, _ = _get_ocr_engine()(np.array(image))
    except Exception as exc:
        logger.error("OCR failed: %s", exc, exc_info=True)
        return ""

    if not result:
        return ""

    lines = []
    for item in result:
        if len(item) >= 2 and item[1]:
            lines.append(str(item[1]))

    text = "\n".join(lines).strip()
    logger.info("OCR extracted %d characters from uploaded image", len(text))
    return text
