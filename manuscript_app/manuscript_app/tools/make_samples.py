r"""Create the sample pages for the website from your held-out TEST pages (pages the AI never saw while learning).

Usage (PowerShell, from the manuscript_app folder):
    python tools\make_samples.py --outputs D:\mbrepo\outputs_v2 --count 6

It reads outputs_v2\final_pages.csv, picks test pages from different years, cuts a real-size window that
contains plenty of handwriting, and saves  static\samples\sample_XX.jpg  (+ the expert answer as sample_XX_truth.png)
and static\samples\samples.json.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent.parent
SAMPLES = HERE / "static" / "samples"


def read_img(path, gray=False):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read {path}")
    return img if gray else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def gt_to_mask(gt):
    """Ground-truth picture -> mask where 1 = ink (same rule as in the training notebook)."""
    if gt.min() == gt.max():
        return np.zeros(gt.shape, np.uint8)
    if len(np.unique(gt)) <= 2:
        mask = (gt == gt.min()).astype(np.uint8)
    else:
        t, _ = cv2.threshold(gt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (gt < t).astype(np.uint8)
    return 1 - mask if mask.mean() > 0.5 else mask


def best_window(mask, win_h, win_w, stride=40):
    """Top-left corner of the window that contains the most ink (uses an integral image)."""
    h, w = mask.shape
    win_h, win_w = min(win_h, h), min(win_w, w)
    integral = cv2.integral(mask.astype(np.float32))
    best, where = -1.0, (0, 0)
    for y in range(0, h - win_h + 1, stride):
        for x in range(0, w - win_w + 1, stride):
            s = integral[y + win_h, x + win_w] - integral[y, x + win_w] - integral[y + win_h, x] + integral[y, x]
            if s > best:
                best, where = s, (y, x)
    return where, win_h, win_w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default=r"D:\mbrepo\outputs_v2", help="folder that contains final_pages.csv")
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args()

    pages = pd.read_csv(Path(args.outputs) / "final_pages.csv")
    pages = pages[pages.split == "test"].sort_values(["year", "image_path"]).reset_index(drop=True)
    if pages.empty:
        raise SystemExit("No test pages found in final_pages.csv")
    picks = np.unique(np.linspace(0, len(pages) - 1, min(args.count, len(pages))).round().astype(int))

    SAMPLES.mkdir(parents=True, exist_ok=True)
    for old in SAMPLES.glob("sample_*"):
        old.unlink()
    items = []
    for n, i in enumerate(picks, start=1):
        row = pages.iloc[i]
        rgb = read_img(row.image_path)
        mask = gt_to_mask(read_img(row.gt_path, gray=True))
        (y, x), wh, ww = best_window(mask, args.height, args.width)
        crop, truth = rgb[y:y + wh, x:x + ww], mask[y:y + wh, x:x + ww]
        name = f"sample_{n:02d}"
        cv2.imwrite(str(SAMPLES / f"{name}.jpg"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(SAMPLES / f"{name}_truth.png"), np.where(truth > 0, 0, 255).astype(np.uint8))
        items.append({
            "id": name,
            "title": f"Manuscript page from {row.year}",
            "note": "A page from the DIBCO benchmark that the AI never saw while learning.",
            "image": f"samples/{name}.jpg",
            "truth": f"samples/{name}_truth.png",
        })
        print("saved", name, "<-", Path(row.image_path).name, f"({ww}x{wh})")
    (SAMPLES / "samples.json").write_text(json.dumps(items, indent=2), encoding="utf-8")
    print(f"Done: {len(items)} samples in {SAMPLES}")


if __name__ == "__main__":
    main()
