"""Cuts a big page into overlapping 256x256 tiles, asks the model about each tile, and blends the answers.

This file uses only numpy, so it works with any engine that has an `infer(batch) -> probabilities` method.
"""
from __future__ import annotations

import numpy as np


def predict_probability(infer, rgb, patch=256, overlap=0.5, batch_size=8):
    """Return an (H, W) float32 map: the probability that each pixel is ink.

    Tiles overlap by 50% and are blended with a smooth window, so no seams are visible.
    """
    h, w = rgb.shape[:2]
    stride = int(patch * (1 - overlap))
    big_h, big_w = max(h, patch), max(w, patch)
    if (big_h - patch) % stride:
        big_h += stride - (big_h - patch) % stride
    if (big_w - patch) % stride:
        big_w += stride - (big_w - patch) % stride
    padded = np.pad(rgb, ((0, big_h - h), (0, big_w - w), (0, 0)), mode="edge")

    window_1d = np.hanning(patch + 2)[1:-1].astype(np.float32)
    window = np.outer(window_1d, window_1d)
    total = np.zeros((big_h, big_w), np.float32)
    weight = np.zeros((big_h, big_w), np.float32)

    corners = [(y, x) for y in range(0, big_h - patch + 1, stride) for x in range(0, big_w - patch + 1, stride)]
    for start in range(0, len(corners), batch_size):
        chunk = corners[start:start + batch_size]
        tiles = np.stack([padded[y:y + patch, x:x + patch] for y, x in chunk])
        probs = infer(tiles)
        for (y, x), p in zip(chunk, probs):
            total[y:y + patch, x:x + patch] += p * window
            weight[y:y + patch, x:x + patch] += window
    return (total / np.maximum(weight, 1e-8))[:h, :w]
