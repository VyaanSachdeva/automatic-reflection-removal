# Development notes

Environment details and known pitfalls for working on this code. See [README.md](README.md) for the project itself.

## Setup that works

- All data lives in `raindrop_project/` next to the repo (`raindrop_paths.RAINDROP_ROOT`). `.env` (gitignored) sets
  `DATASET_DIR=<...>/raindrop_project/organized` for training.
- Python 3.12 with `torch 2.9.1`. `RAIN_DEVICE=cpu` forces the CPU where a GPU is present.
- On an RTX 4060 Laptop (8 GB): 0.87 s per image for raindrop detector + neighbour fill, 5.4 GB peak VRAM at 896 px.
  The training smoke test passes in 51 s.
- On a 16 GB laptop, segment-and-repair with SAM 3 needs about 5 GB free. The demo preloads the older models (about
  10 GB) unless `DEMO_PRELOAD=0` is set.

## Known pitfalls

- **Inverted inpainter mask.** UnReflectAnything's released token inpainter was trained with an inverted mask (see
  the comment in `UnReflect_Model_TokenInpainter.forward` in `models.py`). At a hole it gets the hole's own tokens
  and corrects them in place, so on raindrops it darkens drops instead of removing them. When training our own
  inpainter on VRGS, use the documented convention (True = hole) consistently in training and inference, and check
  which one `models.py` actually passes.
- Keep **transformers pinned** to the git commit in `requirements.txt` and **huggingface-hub==0.36.2**. Newer
  versions break loading the checkpoint.
- **DINOv3:** the configs use the open mirror `AlbeRota/DINOv3_mirror`. The official `facebook/dinov3-…` repo is
  gated.
- The installed `unreflectanything` package ships top-level `models.py` / `token_inpainters.py` that this repo's
  same-named files shadow. `model_stages._load()` handles that; keep it in mind if imports behave oddly.
- `main.py` disables PyTorch's MultiheadAttention fast path on CPU (it segfaulted on Windows). It is fine on CUDA.
- At 896 px, full training ran out of RAM on the 16 GB CPU laptop. On the 8 GB GPU, inference alone peaks at 5.4 GB at
  896 px, so training needs a smaller resolution (448–512 px) and/or bf16 and gradient checkpointing.
- **gradio is pinned to 6.17.3**: 6.18+ needs huggingface-hub ≥ 1.16, which conflicts with the 0.36.2 pin.
- First use downloads the UnReflectAnything weights (3.4 GB, to `%LOCALAPPDATA%\unreflectanything`) **and** the DINOv3
  encoder (1.2 GB, to the Hugging Face cache). Windows reports a growing file's size as 0 while it downloads, so check
  the `.incomplete` file before deciding a download has stalled.
- Set `PYTHONUTF8=1` on Windows, or the training logger prints "Logging error" tracebacks for its ✓ symbols (harmless).
- **The saved neighbour-fill outputs (`demo_assets/drnfill_output`) don't match the current code.** Rerunning
  `model_stages.run_stages(detector="deraindrop", fill="neighbours", composite=True)` gives the same result on CPU and
  GPU, but scores lower than the saved images (photo 19: 20.50 dB / 0.715 vs 21.60 / 0.745). The script and settings
  that produced the saved images aren't in the repo, so the README's 23.38 dB comes from an earlier version.
- `config/train.yaml` is the original paper's full-scale config (896 px). `config/raindrop_train.yaml` is the
  8 GB-GPU version (448 px).
- **VRAM spill on Windows.** When a process outgrows the 8 GB, Windows quietly moves memory into system RAM
  ("shared GPU memory") instead of failing, and training slows to a crawl. Signs: GPU at 100%, VRAM near 8 GB, no
  new log lines. Check with `Get-Counter '\GPU Process Memory(*)\Shared Usage'`. DeRaindrop fine-tune: batch 8
  spills, batch 4 fits (5.3 GB). Raindrop training at 448 px: batch 4 spills, batch 2 is the setting.
- **System RAM.** Each DataLoader / process-pool worker loads its own copy of torch. 8 fine-tune workers plus a
  second job exhausted the commit limit (Windows error 1455) and killed the run; run one heavy job at a time.
- **No DataLoader workers for `train_raindrop.py` on Windows:** the dataset holds an `lru_cache` that can't be
  pickled into spawned workers, so `NUM_WORKERS` must be 0 (training batches come from the VRGS cache anyway).
- The engine requires `LR_SCHEDULER` to list exactly two schedulers: `ONPLATEAU` plus one other.
- The released checkpoint's keys start with `module.` (it was trained with DDP); strip `module.decoders.<name>.`
  to get a decoder's weights for `FROM_PRETRAINED`.
- **`find_concepts` confidence.** SAM 3 is run at the lowest prompt threshold, then each prompt's own threshold is
  applied (`SAM3_PROMPTS`).
- **Glare repair.** The whole-frame restore already dims glare. Taking Flare7K++'s removal off it as well removes the
  glare twice and leaves dark patches, so the hybrid takes the darker of the two per pixel instead.

## Ideas not yet tried

- Sharper in-focus VRGS drops, then retrain the detector: it finds VRGS rain well (IoU 0.53) but not real close-up drops.
- DeRaindrop fine-tune with a lower learning rate (1e-5) or more real pairs, to recover the 0.8 dB lost on real photos.
- Fog/veiling and out-of-focus occluders in VRGS: the CCTV test shows no model here handles them.
- Matched clean frames from the same CCTV cameras, so that footage can be scored with PSNR/SSIM.
- Video: drops stay put while the scene moves, so neighbouring frames carry information a single frame doesn't.
