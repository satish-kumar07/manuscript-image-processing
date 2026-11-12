"""Manuscript Cleaner - web backend (FastAPI).

Run:   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
Open:  http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from .imaging import Reader, data_url, decode_image, jpeg_bytes, limit_size, otsu_ink, png_bytes, sauvola_ink
from .tiling import predict_probability

BASE = Path(__file__).resolve().parent.parent
STATIC = BASE / "static"
MAX_MB = int(os.environ.get("MAX_UPLOAD_MB", "20"))      # biggest upload accepted
MAX_SIDE = int(os.environ.get("MAX_SIDE", "3000"))       # longer side is shrunk to this (keeps it fast)
MAX_PIXELS = 80_000_000                                  # refuse absurdly large pictures

STATE = SimpleNamespace(engine=None, error=None, reader=None, model_path=None)
GPU_LOCK = threading.Lock()                              # one page at a time on the GPU


def find_model():
    """Look for the trained model: MODEL_PATH, ./models, ../outputs_v2, then the author's default folder."""
    candidates = []
    if os.environ.get("MODEL_PATH"):
        candidates.append(Path(os.environ["MODEL_PATH"]))
    candidates += [
        BASE / "models" / "best_unet_v2.pth",
        BASE.parent / "outputs_v2" / "best_unet_v2.pth",
        Path(r"D:\mbrepo\outputs_v2\best_unet_v2.pth"),
    ]
    return next((c for c in candidates if c.exists()), None)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    STATE.reader = Reader()
    path = find_model()
    if path is None:
        STATE.error = "The model file best_unet_v2.pth was not found. Copy it into the 'models' folder and restart the app."
    else:
        try:
            from .engine import TorchEngine              # imported here so the site still opens without PyTorch

            STATE.engine, STATE.model_path = TorchEngine(path), str(path)
        except Exception as exc:                        # missing PyTorch, wrong file, ...
            STATE.error = f"The model could not be loaded: {exc}"
    yield


app = FastAPI(title="Manuscript Cleaner", version="1.0", lifespan=lifespan)


# ------------------------------------------------------------------------------------ small endpoints
@app.get("/api/health")
def health():
    eng = STATE.engine
    return {
        "model_loaded": eng is not None,
        "error": STATE.error,
        "device": eng.device.type if eng else None,
        "gpu_name": eng.gpu_name if eng else None,
        "device_note": eng.device_note if eng else None,
        "parameters": eng.params if eng else None,
        "ocr_available": bool(STATE.reader and STATE.reader.available),
        "max_upload_mb": MAX_MB,
        "max_side_px": MAX_SIDE,
    }


def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


@app.get("/api/results")
def results():
    return _read_json(STATIC / "data" / "results.json", {})


@app.get("/api/samples")
def samples():
    items = _read_json(STATIC / "samples" / "samples.json", [])
    out = []
    for it in items:
        if (STATIC / it.get("image", "")).exists():
            out.append({
                "id": it["id"], "title": it.get("title", it["id"]), "note": it.get("note", ""),
                "image": "/" + it["image"],
                "truth": ("/" + it["truth"]) if it.get("truth") and (STATIC / it["truth"]).exists() else None,
            })
    return out


# ------------------------------------------------------------------------------------ the main job
@app.post("/api/clean")
def clean(file: UploadFile = File(...), ocr: bool = Form(False)):
    """Clean one uploaded page. Nothing is written to disk: the picture only lives in memory."""
    if STATE.engine is None:
        raise HTTPException(503, STATE.error or "The model is not loaded.")

    raw = file.file.read(MAX_MB * 1024 * 1024 + 1)
    if not raw:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(raw) > MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"This file is larger than {MAX_MB} MB. Please upload a smaller picture.")

    t_start = time.perf_counter()
    try:
        rgb = decode_image(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    h0, w0 = rgb.shape[:2]
    if h0 * w0 > MAX_PIXELS:
        raise HTTPException(413, "This picture has too many pixels. Please upload a smaller scan.")
    if min(h0, w0) < 16:
        raise HTTPException(400, "This picture is too small to contain readable text.")

    rgb, scale = limit_size(rgb, MAX_SIDE)
    h, w = rgb.shape[:2]

    with GPU_LOCK:
        t_ai = time.perf_counter()
        prob = predict_probability(STATE.engine.infer, rgb, patch=STATE.engine.patch)
        seconds_ai = time.perf_counter() - t_ai
    ink_ai = prob > 0.5
    ink_otsu, ink_sauvola = otsu_ink(rgb), sauvola_ink(rgb)

    pct = lambda m: round(float(m.mean()) * 100, 2)
    stats = {"ink_ai_pct": pct(ink_ai), "ink_otsu_pct": pct(ink_otsu), "ink_sauvola_pct": pct(ink_sauvola)}

    notes = []
    if scale < 1:
        notes.append(f"Your picture was very large, so it was shrunk from {w0}\u00d7{h0} to {w}\u00d7{h} pixels before cleaning.")
    if stats["ink_ai_pct"] < 0.2:
        notes.append("Almost no text was found. The page may be blank, or the writing may be too faint or too small for the model.")
    if stats["ink_ai_pct"] > 35:
        notes.append("More than a third of the page was marked as ink. This is unusual: the picture may be very dark, upside down in tone, or not a text page.")
    if min(h, w) < 200:
        notes.append("This is a small picture. Results are usually better on full-size scans.")

    ocr_result = None
    if ocr:
        reader = STATE.reader
        if reader and reader.available:
            import cv2
            import numpy as np

            cleaned = np.where(ink_ai, 0, 255).astype(np.uint8)
            ocr_result = {
                "available": True,
                "original": reader.read(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)),
                "cleaned": reader.read(cleaned),
            }
        else:
            ocr_result = {"available": False, "original": "", "cleaned": ""}

    return {
        "width": w, "height": h, "resized": scale < 1, "original_size": [w0, h0],
        "images": {
            "original": data_url(jpeg_bytes(rgb), "image/jpeg"),
            "ai": data_url(png_bytes(ink_ai), "image/png"),
            "otsu": data_url(png_bytes(ink_otsu), "image/png"),
            "sauvola": data_url(png_bytes(ink_sauvola), "image/png"),
        },
        "stats": {**stats, "seconds_ai": round(seconds_ai, 2), "seconds_total": round(time.perf_counter() - t_start, 2),
                  "device": STATE.engine.device.type, "gpu_name": STATE.engine.gpu_name},
        "notes": notes,
        "ocr": ocr_result,
    }


# the website itself (must be mounted last so the /api routes win)
app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="site")
