"""
Network definitions — each class must match, layer for layer, the architecture
that was trained in Colab, otherwise load_state_dict fails.

  SRCNN-Y : 9-1-5, 1 channel in / 1 channel out, takes the BICUBIC-upscaled Y
  VDSR-Y  : 20 conv layers, global residual,    takes the BICUBIC-upscaled Y
  SRGAN-Y : SRResNet-style generator (8 residual blocks, 2x PixelShuffle),
            takes the LOW-RESOLUTION Y and upscales x4 inside the network
"""

import os

import torch
import torch.nn as nn


class SRCNN(nn.Module):
    def __init__(self, num_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, 9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, 1)
        self.conv3 = nn.Conv2d(32, num_channels, 5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv3(self.relu(self.conv2(self.relu(self.conv1(x)))))


class VDSR(nn.Module):
    def __init__(self, num_layers: int = 20, num_filters: int = 64):
        super().__init__()
        layers = [nn.Conv2d(1, num_filters, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [nn.Conv2d(num_filters, num_filters, 3, padding=1), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(num_filters, 1, 3, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, c: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.PReLU(),
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c),
        )

    def forward(self, x):
        return x + self.body(x)


class Generator(nn.Module):
    def __init__(self, nb: int = 8, c: int = 64):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(1, c, 9, padding=4), nn.PReLU())
        self.blocks = nn.Sequential(*[ResBlock(c) for _ in range(nb)])
        self.mid = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c))
        self.up = nn.Sequential(*[
            m for _ in range(2)
            for m in (nn.Conv2d(c, c * 4, 3, padding=1), nn.PixelShuffle(2), nn.PReLU())
        ])
        self.tail = nn.Conv2d(c, 1, 9, padding=4)

    def forward(self, x):
        h = self.head(x)
        h = h + self.mid(self.blocks(h))
        return self.tail(self.up(h))


# key -> (label, checkpoint file, constructor, input kind)
#   input kind "bicubic": network refines a Y plane that is already at HR size
#   input kind "lr"     : network takes the LR Y plane and upscales it itself
MODEL_SPECS = {
    "srcnn": ("SRCNN-Y", "srcnn_y_model.pth", SRCNN, "bicubic"),
    "vdsr": ("VDSR-Y", "vdsr_y_model_300_fixed.pth", VDSR, "bicubic"),
    "srgan": ("SRGAN-Y", "srgan_y_G.pth", Generator, "lr"),
}


def load_available_models(models_dir: str, device: torch.device) -> dict:
    """Load every checkpoint that is present in models_dir; skip the rest."""
    loaded = {}
    for key, (label, fname, ctor, kind) in MODEL_SPECS.items():
        path = os.path.join(models_dir, fname)
        if not os.path.isfile(path):
            print(f"[models] {label}: {fname} not found, skipping")
            continue
        net = ctor().to(device)
        net.load_state_dict(torch.load(path, map_location=device))
        net.eval()
        loaded[key] = {"label": label, "net": net, "kind": kind}
        print(f"[models] {label}: loaded")
    return loaded
