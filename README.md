# Rain on Glass

**Removing raindrops, streaks and glare from dashcam and CCTV footage.**

When it rains, water sits on the camera's own glass: the windshield in front of a dashcam, or the dome or housing
of a CCTV camera. Droplets refract and blur the scene behind them, sliding drops leave streaks, and headlights and
streetlights flare through the wet glass. This project restores a single rainy frame.

This is different from removing *falling* rain (the "rain streak removal" problem). Here the water is on the lens: it
is close to the camera, out of focus, and stays put while the scene moves.

## Approach

Real pairs of rainy and clean footage from the same camera barely exist, so the model learns from synthetic rain.

1. **VRGS (Virtual Raindrop-Glare Synthesis)** renders rain on glass onto ordinary clean driving footage (KITTI):
   - **Droplets:** each drop is a tiny lens showing a shrunken, upside-down view of the scene, drawn as the glass
     really looks a few cm from the lens: out of focus, mostly see-through, irregular (rotated ellipses with a wobbly
     edge), numerous and mostly small, cut off at the frame edge. Every drop draws its own opacity, rim, blur and
     edge softness, so a frame mixes faint and clearly visible drops.
   - **Streaks:** trails sharing one wind direction per frame, angled for a moving dashcam and straight down for a
     static CCTV camera.
   - **Glare:** real light-streak shapes from Flare7K, placed where the scene already has bright lights. Only the
     bright part counts as degraded in the mask.
   - **Haze:** a thin wet-glass veil over the whole frame.

   Every rendered frame comes with an exact mask of the pixels it changed, which is the detector's training target.
   VRGS is a procedural renderer with no learned weights ([raindrop_render.py](raindrop_render.py)).

2. **The restoration model** adapts the architecture of UnReflectAnything (Rota et al., CVPR 2026), a
   specular-highlight remover that also learns from synthetic data:

   ```
   rainy frame → raindrop detector → DINOv3-L encoder (frozen) → choose patches
               → repair them in feature space → DPT decoder → restored frame
   ```

   Pixels outside the repaired patches are kept from the original frame.

## Status

The synthesis, data pipeline, training engine and a live demo are built. The full training run at 896 px needs a
GPU, so the current model still uses pretrained parts:

| Stage | Currently | After VRGS training |
|---|---|---|
| Raindrop detector | DeRaindrop's attention network (Qian et al., CVPR 2018) | Trained on VRGS masks |
| Encoder | DINOv3 ViT-L/16, frozen | Unchanged |
| Repair step | **Neighbour fill**: rebuilds each patch from the clean patches around it, with no learned weights; or **LaMa** (big-lama, pretrained), which inpaints the marked pixels directly at full resolution | Inpainter trained on VRGS pairs |
| Decoder | UnReflectAnything's released DPT decoder | Fine-tuned on VRGS pairs |

**Why neighbour fill:** UnReflectAnything's released token inpainter was trained with an inverted mask. At each hole
it receives that hole's own features and learned to correct a highlight in place, so on a raindrop it keeps the drop
and darkens it. Neighbour fill discards the drop's features and is about 9× faster.

### Results so far

8 real rainy photos from the DeRaindrop test set, scored against their clean versions (higher is better):

| Method | PSNR (dB) | SSIM | Better than the untouched photo |
|---|---|---|---|
| Rainy input, left alone | 25.08 | 0.863 | – |
| UnReflectAnything, stock | 23.58 | 0.786 | 0 / 8 |
| + raindrop detector, released inpainter | 22.86 | 0.819 | 0 / 8 |
| + raindrop detector, neighbour fill | 23.38 | 0.817 | 0 / 8 |
| Raindrop detector + LaMa fill (pixel space) | **25.30** | 0.842 | 0 / 8 |
| *True droplet mask + LaMa (upper bound on detection)* | *26.86* | *0.870* | *3 / 8* |
| DeRaindrop | 31.32 | 0.922 | 7 / 8 |

**LaMa fill** ([eval_lama.py](eval_lama.py), [eval_results/lama_fill.json](eval_results/lama_fill.json)) is the first
repair step to score above the untouched photo on PSNR (+0.22 dB, and +1.9 dB over neighbour fill); visually it
rebuilds doors, railings and wall tiles where neighbour fill leaves blocky smears. It still loses on SSIM, and the
detector is why: DeRaindrop's raindrop map marks 24% of the pixels where the true droplets cover 13%, so LaMa repaints
clean areas, and it misses faint, out-of-focus droplets. With the true droplet mask (taken from the clean photo, so
only a diagnostic) the same fill reaches 26.86 dB / 0.870, above the untouched photo on both. A VRGS-trained detector
is the next lever. LaMa takes about 0.3 s per photo on the GPU.

DeRaindrop wins on these photos, but they come from its own test set (the glass pane and camera it was trained on).
On VRGS dashcam/CCTV rain it beats the untouched frame on **0 of 8** frames: it leaves the streaks and glare in
place. Fine-tuning it for 30 minutes on the GPU (4,252 steps; current VRGS on three KITTI drives, half real
DeRaindrop pairs) makes it the first version that helps on dashcam/CCTV rain
([eval_results/deraindrop_vrgs_finetune.json](eval_results/deraindrop_vrgs_finetune.json)):

| | VRGS rain, unseen KITTI drive (18 frames) | Real photos, DeRaindrop test_a (58) |
|---|---|---|
| Rainy input | 25.41 dB / 0.928 | 23.81 dB / 0.847 |
| DeRaindrop, released | 25.78 / 0.898, beats input on 0 / 18 | **30.93** / 0.911, 57 / 58 |
| Fine-tuned (earlier: CPU, 20 min, old VRGS) | 25.46 / 0.901, 0 / 18 | 30.66 / 0.912, 58 / 58 |
| **Fine-tuned (GPU, 30 min)** | **26.29 / 0.923, 7 / 18** | 30.11 / **0.912**, **58 / 58** |

It now removes some streaks and glare, but it costs 0.8 dB PSNR on the real photos (SSIM unchanged), and it still
trails the untouched VRGS frame on average SSIM. A lower learning rate or more real pairs in the mix should close
the real-photo gap.

### Efficiency

Measured on an Intel Core i7-1065G7 laptop CPU, batch size 1
([benchmark_efficiency.py](benchmark_efficiency.py), [eval_results/efficiency.json](eval_results/efficiency.json)):

| | Parameters | FLOPs / image | Latency | Throughput |
|---|---|---|---|---|
| Stock UnReflectAnything | 498 M | ≈5.5 T | 78.7 s | 0.76 img/min |
| Current pipeline | 408 M | ≈4.2 T | 52.9 s | 1.13 img/min |
| DeRaindrop | 6.2 M | 0.81 T | 10.8 s | 5.5 img/min |
| VRGS renderer (per 960×290 frame) | none | – | 6.6 s | 9.1 frames/min |

Peak memory is 5.5 GB. The frozen DINOv3 encoder accounts for 61% of the pipeline's latency.

On an NVIDIA RTX 4060 Laptop GPU (8 GB) the current pipeline takes **0.87 s** per image (about 60× faster), with
5.4 GB peak VRAM at its 896 px working size. The demo and DeRaindrop use the GPU automatically; set
`RAIN_DEVICE=cpu` to force the CPU. `benchmark_efficiency.py` always measures on the CPU.

## Getting started

### 1. Install

Windows with Python 3.12:

```bat
git clone https://github.com/VyaanSachdeva/rain-on-glass.git
cd rain-on-glass
py -3.12 -m venv .venv
.venv\Scripts\activate
rem PyTorch first (pick one):
pip install torch==2.9.1 torchvision==0.24.1
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

The pretrained UnReflectAnything weights (3.4 GB) and the DINOv3 encoder (1.2 GB) download automatically on first
use. On Windows, set `PYTHONUTF8=1` so the training logger can print its symbols.

### 2. Data

All data lives in a `raindrop_project` folder **next to** this repo. Set `RAINDROP_ROOT` to put it elsewhere.

```
raindrop_project/
  extracted/
    kitti_2011_09_26_drive_0001/   KITTI raw, drive 0001 (clean driving frames)
    DeRaindrop/{train,test_a,test_b}/{data,gt}   real rain/clean pairs
    Flare7K/Flare7Kpp/             Flare7K++ (glare assets)
  organized/KITTI_Clean/all/rgb/   KITTI frames from image_02/data, used for training
  derainnet_code/                  git clone of https://github.com/rui1996/DeRaindrop (weights/gen.pkl)
  checkpoints/                     fine-tuned weights written by finetune_deraindrop.py
  lama/big-lama.pt                 LaMa weights (TorchScript, 196 MB): github.com/enesmsahin/simple-lama-inpainting releases v0.1.0
  demo_assets/                     demo examples and precomputed outputs
```

| Dataset | Source | License |
|---|---|---|
| KITTI raw | [cvlibs.net/datasets/kitti/raw_data.php](https://www.cvlibs.net/datasets/kitti/raw_data.php) | CC BY-NC-SA 3.0 |
| DeRaindrop | [github.com/rui1996/DeRaindrop](https://github.com/rui1996/DeRaindrop) | See the repository |
| Flare7K++ | [github.com/ykdai/Flare7K](https://github.com/ykdai/Flare7K) | S-Lab License 1.0 (non-commercial) |

For training, set `DATASET_DIR` to `raindrop_project/organized` in a `.env` file in the repo folder.

### 3. Run

| What | Command |
|---|---|
| Live demo (add rain, watch each model stage, compare baselines) | `run_demo.bat`, or `python demo_app.py`, then open http://127.0.0.1:7860 |
| Preview VRGS on KITTI frames | `python raindrop_render.py` |
| Fine-tune DeRaindrop on VRGS (GPU; batch 4 fits 8 GB) | `python finetune_deraindrop.py --minutes 30 --batch 4 --workers 4` |
| Download more KITTI drives (left camera only) | `python download_kitti.py 0002 0020 0079 0113` |
| Pre-render the cached VRGS training set | `python prerender_vrgs.py --size 448 --variants 4` |
| Train the raindrop detector + inpainter (GPU) | `python train_raindrop.py --config config/raindrop_train.yaml` |
| Score every method on the same test sets | `python eval_all.py` |
| Measure efficiency | `python benchmark_efficiency.py` |
| Training smoke test (224 px, 8 images; ~1 min on a GPU) | `python train_raindrop.py --config config/raindrop_smoke_test.yaml` |

`config/train.yaml` holds the original paper's full-scale settings (896 px, batch size and schedule). It is the
reference for the full GPU training config.

## Repository layout

| File | Role |
|---|---|
| [raindrop_render.py](raindrop_render.py) | VRGS: the rain-on-glass renderer |
| [raindrop_engine.py](raindrop_engine.py), [train_raindrop.py](train_raindrop.py) | Training with VRGS as the degradation |
| [model_stages.py](model_stages.py) | Runs the model one stage at a time; neighbour fill and LaMa fill |
| [eval_lama.py](eval_lama.py) | Scores LaMa fill against neighbour fill and the untouched input |
| [derainnet.py](derainnet.py) | DeRaindrop's generator, ported to current PyTorch (GPU when available, else CPU) |
| [finetune_deraindrop.py](finetune_deraindrop.py) | Fine-tunes DeRaindrop on VRGS + real pairs |
| [demo_app.py](demo_app.py), [run_demo.bat](run_demo.bat) | Gradio demo |
| [benchmark_efficiency.py](benchmark_efficiency.py) | Parameters, FLOPs, latency, memory |
| [raindrop_paths.py](raindrop_paths.py) | Where the data folder is |
| `engine.py`, `main.py`, `models.py`, `token_inpainters.py`, `losses.py`, `dataset/`, `utilities/` | Training framework from UnReflectAnything, adapted |

## Roadmap

- [x] VRGS renderer (droplets, streaks, glare, haze, exact masks)
- [x] Data pipeline (KITTI, DeRaindrop, Flare7K++) and training engine (CPU smoke test)
- [x] Baselines, diagnosis of the released inpainter, neighbour fill
- [x] First VRGS fine-tune (DeRaindrop)
- [ ] Faster, vectorised VRGS for large-scale data generation
- [ ] Full VRGS training of the detector and inpainter at 896 px on a GPU
- [ ] Evaluation on the full DeRaindrop test sets and a CCTV source
- [ ] Video: use neighbouring frames, since drops stay put while the scene moves

## Credits

- The model architecture and training framework are adapted from **UnReflectAnything**: A. Rota, M. Kiray,
  M. A. Karaoglu, P. Ruhkamp, E. De Momi, N. Navab, B. Busam, *UnReflectAnything: RGB-Only Highlight Removal by
  Rendering Synthetic Specular Supervision*, CVPR 2026. MIT License, see [LICENSE](LICENSE).
- **DINOv3** (Meta) is used under the DINOv3 License, see [LICENSE_DINO.txt](LICENSE_DINO.txt).
- **DeRaindrop**: R. Qian, R. T. Tan, W. Yang, J. Su, J. Liu, *Attentive Generative Adversarial Network for
  Raindrop Removal from a Single Image*, CVPR 2018.
- Droplet optics follow M. Roser and A. Geiger, *Video-based Raindrop Detection for Improved Image Registration*,
  ICCV Workshops 2009.
- Glare assets: Y. Dai et al., *Flare7K*, NeurIPS 2022.
- **LaMa**: R. Suvorov et al., *Resolution-robust Large Mask Inpainting with Fourier Convolutions*, WACV 2022
  (Apache 2.0); TorchScript export of big-lama from the simple-lama-inpainting project.

**Author:** Vyaan Sachdeva
