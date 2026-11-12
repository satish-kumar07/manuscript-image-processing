r"""One-minute self-test: loads your trained model and cleans a random page, so you know the GPU path works.

Run from the manuscript_app folder:   python tools\check_model.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.engine import TorchEngine
from app.main import find_model
from app.tiling import predict_probability

path = find_model()
if path is None:
    raise SystemExit("Model not found. Copy best_unet_v2.pth into the 'models' folder (or set MODEL_PATH).")
print("Model file :", path)

t0 = time.perf_counter()
engine = TorchEngine(path)
print(f"Loaded in {time.perf_counter() - t0:.1f} s | device: {engine.device} {engine.gpu_name or ''} | parameters: {engine.params:,} | tile: {engine.patch}px")
if engine.device_note:
    print("Note       :", engine.device_note)

page = np.random.default_rng(0).integers(120, 255, size=(1200, 1600, 3), dtype=np.uint8)
t0 = time.perf_counter()
prob = predict_probability(engine.infer, page, patch=engine.patch)
print(f"Cleaned a 1600x1200 test page in {time.perf_counter() - t0:.2f} s | output {prob.shape}, values {prob.min():.2f}..{prob.max():.2f}")
print("OK: the model runs. Start the website with run_app.bat")
