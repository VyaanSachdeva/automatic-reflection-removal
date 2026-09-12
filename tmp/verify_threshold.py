"""Verify that the only output difference between gradio (thr=0.2) and notebook (thr=0.3)
is the Highlight Threshold. Saves diffuse outputs for both thresholds side by side."""

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF
from huggingface_hub import hf_hub_download, snapshot_download

from unreflectanything import model

HF_REPO = "AlbeRota/UnReflectAnything"
REV = "a0d3c7bff5ddb2c430e74b5f8ee67be7cc28fcbe"
OUT = Path(__file__).parent

weights_path = hf_hub_download(repo_id=HF_REPO, revision=REV, filename="weights/full_model_weights.pt")
snapshot_download(repo_id=HF_REPO, revision=REV)  # ensure sibling decoder weights present

device = "cuda" if torch.cuda.is_available() else "cpu"
ura = model(pretrained=True, weights_path=weights_path, device=device, verbose=False)

image = np.array(Image.open(OUT.parent / "mgmt/sample_images/sampleimage_1.png").convert("RGB"))  # (H,W,3)
h, w = image.shape[:2]
target = ura.image_size

tensor = TF.to_tensor(image).unsqueeze(0)                      # [1,3,H,W]
tensor = TF.resize(tensor, [target, target], antialias=True)  # [1,3,896,896]
tensor = tensor.to(device, dtype=torch.float32)

for thr in (0.2, 0.3):
    with torch.no_grad():
        t0 = time.time()
        out = ura(images=tensor, threshold=thr, dilation=40, return_dict=True)
        dt = (time.time() - t0) * 1000
    diffuse = TF.resize(out["diffuse"].cpu(), [h, w], antialias=True)        # [1,3,h,w]
    arr = (diffuse[0].numpy().transpose(1, 2, 0).clip(0, 1) * 255).astype(np.uint8)  # (h,w,3)
    # fraction of patches sent to the token inpainter
    pm = out["patch_mask"]  # [B,1,gh,gw]
    frac = (pm > 0.5).float().mean().item()
    Image.fromarray(arr).save(OUT / f"diffuse_thr_{thr}.png")
    print(f"threshold={thr}: {dt:.0f} ms, inpainted-patch fraction = {frac:.3f} -> diffuse_thr_{thr}.png")

print("done")
