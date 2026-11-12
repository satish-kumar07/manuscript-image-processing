"""Image helpers: reading uploads, the two classic filters we compare against, PNG/JPEG encoding, optional OCR."""
from __future__ import annotations

import base64
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import threshold_sauvola


def decode_image(raw: bytes) -> np.ndarray:
    """Bytes -> RGB uint8 array. Raises ValueError with a friendly message if it is not an image."""
    data = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("This file could not be read as an image. Please upload a PNG, JPG, BMP, TIFF or WEBP picture.")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def limit_size(rgb: np.ndarray, max_side: int):
    """Shrink very large pictures (never enlarge). Returns (image, scale_used)."""
    h, w = rgb.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return rgb, 1.0
    scale = max_side / longest
    small = cv2.resize(rgb, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return small, scale


def otsu_ink(rgb: np.ndarray) -> np.ndarray:
    """'Simple filter': one brightness threshold for the whole page. True = ink."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary > 0


def sauvola_ink(rgb: np.ndarray, window: int = 51, k: float = 0.2) -> np.ndarray:
    """'Local filter': a separate threshold for every neighbourhood. True = ink."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray < threshold_sauvola(gray, window_size=window, k=k)


def png_bytes(ink: np.ndarray) -> bytes:
    """Ink mask -> black text on a white page, as PNG bytes."""
    page = np.where(ink, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", page, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise RuntimeError("Could not encode the result image.")
    return buf.tobytes()


def jpeg_bytes(rgb: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Could not encode the preview image.")
    return buf.tobytes()


def data_url(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


# ---------------------------------------------------------------- optional text reading (OCR) ----
def find_tesseract():
    """Path to tesseract.exe / tesseract, or None if it is not installed."""
    candidates = [
        os.environ.get("TESSERACT_CMD"),
        shutil.which("tesseract"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        str(Path.home() / r"AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


class Reader:
    """Reads the text on a page with Tesseract. `available` is False when Tesseract or pytesseract is missing."""

    def __init__(self):
        self.available = False
        self._pt = None
        path = find_tesseract()
        if not path:
            return
        try:
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = path
            pytesseract.get_tesseract_version()
            self._pt, self.available = pytesseract, True
        except Exception:
            self.available = False

    def read(self, page_gray: np.ndarray, timeout: int = 60) -> str:
        if not self.available:
            return ""
        try:
            return " ".join(self._pt.image_to_string(page_gray, lang="eng", timeout=timeout).split())
        except Exception:
            return ""
