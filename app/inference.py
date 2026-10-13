"""
Inference pipeline, identical to the Colab evaluation code:

  1. The uploaded image is treated as the low-resolution (LR) input.
  2. RGB -> YCbCr (full-range BT.601, as PIL does).
  3. Y goes through the network (SRCNN/VDSR on the bicubic-upscaled Y,
     SRGAN on the LR Y). Cb/Cr are always upscaled with plain bicubic.
  4. Merge Y' + Cb + Cr and convert back to RGB.
"""

import io

import numpy as np
import torch
from PIL import Image

SCALE = 4


def bicubic_upscale(img: Image.Image, scale: int = SCALE) -> Image.Image:
    return img.resize((img.width * scale, img.height * scale), Image.BICUBIC)


def prepare_input(image_bytes: bytes, max_dim: int) -> Image.Image:
    img = Image.open(io.BytesIO(image_bytes))
    img.load()
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        r = max_dim / max(w, h)
        img = img.resize((max(1, round(w * r)), max(1, round(h * r))), Image.BICUBIC)
    return img


@torch.no_grad()
def run_model(entry: dict, lr_img: Image.Image, device: torch.device) -> Image.Image:
    y, cb, cr = lr_img.convert("YCbCr").split()
    size_hr = (lr_img.width * SCALE, lr_img.height * SCALE)

    if entry["kind"] == "bicubic":
        net_in = y.resize(size_hr, Image.BICUBIC)
    else:
        net_in = y

    x = torch.from_numpy(np.array(net_in)).float().div(255)[None, None].to(device)
    out = entry["net"](x)[0, 0].cpu().numpy()
    out_y = Image.fromarray(np.clip(out * 255.0, 0, 255).round().astype(np.uint8))

    cb_up = cb.resize(out_y.size, Image.BICUBIC)
    cr_up = cr.resize(out_y.size, Image.BICUBIC)
    return Image.merge("YCbCr", [out_y, cb_up, cr_up]).convert("RGB")


def image_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()
