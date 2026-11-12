"""Loads the trained U-Net and runs it on a batch of image tiles (GPU if possible, otherwise CPU).

The U-Net classes below are copied unchanged from the training notebook (cell T3), so the saved
weights in best_unet_v2.pth load without any key mismatch.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
            nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]
        self.inc = DoubleConv(in_ch, c[0])
        self.d1, self.d2 = DoubleConv(c[0], c[1]), DoubleConv(c[1], c[2])
        self.d3, self.d4 = DoubleConv(c[2], c[3]), DoubleConv(c[3], c[4])
        self.pool = nn.MaxPool2d(2)
        self.u1, self.c1 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2), DoubleConv(c[3] * 2, c[3])
        self.u2, self.c2 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2), DoubleConv(c[2] * 2, c[2])
        self.u3, self.c3 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2), DoubleConv(c[1] * 2, c[1])
        self.u4, self.c4 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2), DoubleConv(c[0] * 2, c[0])
        self.outc = nn.Conv2d(c[0], out_ch, 1)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(self.pool(x1)); x3 = self.d2(self.pool(x2))
        x4 = self.d3(self.pool(x3)); x5 = self.d4(self.pool(x4))
        x = self.c1(torch.cat([self.u1(x5), x4], 1))
        x = self.c2(torch.cat([self.u2(x), x3], 1))
        x = self.c3(torch.cat([self.u3(x), x2], 1))
        x = self.c4(torch.cat([self.u4(x), x1], 1))
        return self.outc(x)


def pick_device():
    """Return (device, note). Uses CUDA only if a test allocation really works."""
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")          # fails here if the GPU is in a bad state
            return torch.device("cuda"), None
        return torch.device("cpu"), "CUDA is not available, using the CPU (slower)."
    except Exception as exc:                        # sticky CUDA errors, driver problems, ...
        return torch.device("cpu"), f"CUDA problem ({str(exc).splitlines()[0]}), using the CPU (slower)."


class TorchEngine:
    """Wraps the U-Net. `infer` takes tiles (N, H, W, 3) uint8 and returns ink probabilities (N, H, W)."""

    def __init__(self, model_path):
        self.device, self.device_note = pick_device()
        ckpt = torch.load(str(model_path), map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state, base, patch = ckpt["model"], int(ckpt.get("base") or 0), int(ckpt.get("patch") or 256)
        else:                                       # a plain state_dict (unet_v2_weights_only.pth)
            state, base, patch = ckpt, 0, 256
        if not base:
            base = int(state["inc.net.0.weight"].shape[0])
        self.patch = patch
        self.model = UNet(base=base).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.use_amp = self.device.type == "cuda"
        self.params = sum(p.numel() for p in self.model.parameters())
        self.gpu_name = torch.cuda.get_device_name(0) if self.device.type == "cuda" else None

    def infer(self, batch):
        try:
            return self._infer(batch)
        except torch.cuda.OutOfMemoryError:          # too big for the GPU: split the batch and retry
            torch.cuda.empty_cache()
            if len(batch) == 1:
                raise
            half = len(batch) // 2
            return np.concatenate([self.infer(batch[:half]), self.infer(batch[half:])])

    @torch.inference_mode()
    def _infer(self, batch):
        x = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).float().div_(255).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.use_amp):
            probs = torch.sigmoid(self.model(x))
        return probs.float().cpu().numpy()[:, 0]
