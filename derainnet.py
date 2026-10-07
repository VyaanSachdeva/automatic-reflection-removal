"""
DeRaindrop's attentive generator (Qian et al., "Attentive Generative Adversarial
Network for Raindrop Removal from A Single Image", CVPR 2018,
https://github.com/rui1996/DeRaindrop), ported to current PyTorch and CPU.

The layers are copied unchanged from the official models/generator.py so the
released weights/gen.pkl loads as-is; only the hard-coded .cuda() calls and the
deprecated Variable/F.tanh are gone. Its recurrent attention branch outputs a
per-pixel raindrop map before any restoration happens, which makes it usable on
its own as a raindrop *detector* -- the part UnReflectAnything's highlight
detector gets wrong on this degradation.

The network was trained on OpenCV-loaded (BGR) images in [0, 1], with sides a
multiple of 4; derain() handles both and works in RGB.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from raindrop_paths import RAINDROP_ROOT

ITERATION = 4
WEIGHTS = Path(os.environ.get(
    "DERAINDROP_WEIGHTS",
    RAINDROP_ROOT / "derainnet_code" / "weights" / "gen.pkl",
))


def _conv_relu(cin, cout, k=3, s=1, p=1, d=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, p, dilation=d), nn.ReLU())


def _res_pair():
    return nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.ReLU())


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.det_conv0 = _conv_relu(4, 32)
        self.det_conv1 = _res_pair()
        self.det_conv2 = _res_pair()
        self.det_conv3 = _res_pair()
        self.det_conv4 = _res_pair()
        self.det_conv5 = _res_pair()
        self.conv_i = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_f = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Sigmoid())
        self.conv_g = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Tanh())
        self.conv_o = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.Sigmoid())
        self.det_conv_mask = nn.Sequential(nn.Conv2d(32, 1, 3, 1, 1))
        self.conv1 = _conv_relu(4, 64, 5, 1, 2)
        self.conv2 = _conv_relu(64, 128, 3, 2, 1)
        self.conv3 = _conv_relu(128, 128)
        self.conv4 = _conv_relu(128, 256, 3, 2, 1)
        self.conv5 = _conv_relu(256, 256)
        self.conv6 = _conv_relu(256, 256)
        self.diconv1 = _conv_relu(256, 256, 3, 1, 2, 2)
        self.diconv2 = _conv_relu(256, 256, 3, 1, 4, 4)
        self.diconv3 = _conv_relu(256, 256, 3, 1, 8, 8)
        self.diconv4 = _conv_relu(256, 256, 3, 1, 16, 16)
        self.conv7 = _conv_relu(256, 256)
        self.conv8 = _conv_relu(256, 256)
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReflectionPad2d((1, 0, 1, 0)),
                                     nn.AvgPool2d(2, stride=1), nn.ReLU())
        self.conv9 = _conv_relu(128, 128)
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReflectionPad2d((1, 0, 1, 0)),
                                     nn.AvgPool2d(2, stride=1), nn.ReLU())
        self.conv10 = _conv_relu(64, 32)
        self.outframe1 = _conv_relu(256, 3)
        self.outframe2 = _conv_relu(128, 3)
        self.output = nn.Sequential(nn.Conv2d(32, 3, 3, 1, 1))

    def attention(self, input):
        """The recurrent raindrop-attention branch alone: list of ITERATION maps."""
        b, _, row, col = input.shape
        mask = input.new_full((b, 1, row, col), 0.5)
        h = input.new_zeros((b, 32, row, col))
        c = input.new_zeros((b, 32, row, col))
        mask_list = []
        for _ in range(ITERATION):
            x = self.det_conv0(torch.cat((input, mask), 1))
            for block in (self.det_conv1, self.det_conv2, self.det_conv3, self.det_conv4, self.det_conv5):
                x = F.relu(block(x) + x)
            x = torch.cat((x, h), 1)
            i, f, g, o = self.conv_i(x), self.conv_f(x), self.conv_g(x), self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)
            mask = self.det_conv_mask(h)
            mask_list.append(mask)
        return mask_list

    def restore(self, input, mask):
        x = self.conv1(torch.cat((input, mask), 1))
        res1 = x
        x = self.conv3(self.conv2(x))
        res2 = x
        x = self.conv6(self.conv5(self.conv4(x)))
        x = self.diconv4(self.diconv3(self.diconv2(self.diconv1(x))))
        x = self.conv8(self.conv7(x))
        x = self.deconv1(x) + res2
        x = self.conv9(x)
        x = self.deconv2(x) + res1
        return self.output(self.conv10(x))

    def forward(self, input):
        mask_list = self.attention(input)
        return mask_list, self.restore(input, mask_list[-1])


FINETUNED = Path(os.environ.get(
    "DERAINDROP_VRGS_WEIGHTS",
    RAINDROP_ROOT / "checkpoints" / "deraindrop_vrgs.pkl",
))
_models: dict[Path, Generator] = {}


def load(weights: Path = WEIGHTS) -> Generator:
    """weights: WEIGHTS (the released model) or FINETUNED (finetune_deraindrop.py)."""
    if weights not in _models:
        m = Generator()
        m.load_state_dict(torch.load(weights, map_location="cpu"))
        _models[weights] = m.eval()
    return _models[weights]


def _prep(img: Image.Image):
    rgb = img.convert("RGB")
    w, h = rgb.size
    w4, h4 = w // 4 * 4, h // 4 * 4
    a = np.asarray(rgb, dtype=np.float32)[:h4, :w4, ::-1] / 255.0  # crop to /4, RGB -> BGR
    return rgb, w, h, w4, h4, torch.from_numpy(a.transpose(2, 0, 1).copy()).unsqueeze(0)


@torch.no_grad()
def attention_map(img: Image.Image) -> torch.Tensor:
    """Just the raindrop detector: [1,1,H,W] in [0,1] at the input's size.
    About a third of a full derain() on CPU."""
    _, w, h, w4, h4, x = _prep(img)
    att = load().attention(x)[-1].clamp(0, 1)
    return F.pad(att, (0, w - w4, 0, h - h4))


@torch.no_grad()
def derain(img: Image.Image, weights: Path = WEIGHTS) -> tuple[Image.Image, torch.Tensor]:
    """Returns (restored image, raindrop attention map [1,1,H,W] in [0,1]),
    both at the input's own size."""
    model = load(weights)
    rgb, w, h, w4, h4, x = _prep(img)
    masks, out = model(x)
    out = out.clamp(0, 1)[:, [2, 1, 0]]  # BGR -> RGB
    att = masks[-1].clamp(0, 1)
    if (w4, h4) != (w, h):  # restore the cropped border from the input, so sizes match
        full = torch.from_numpy(np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0)
        full[:, :, :h4, :w4] = out
        out = full
        att = F.pad(att, (0, w - w4, 0, h - h4))
    out_img = Image.fromarray((out[0].permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8))
    return out_img, att
