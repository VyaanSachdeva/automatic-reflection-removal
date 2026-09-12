# UnReflectAnything

[![Project](https://img.shields.io/badge/Project-Webpage-ff611b?logo=googlehome&logoColor=ff611b)](https://alberto-rota.github.io/UnReflectAnything/)
[![PyPI](https://img.shields.io/pypi/v/unreflectanything?color=76b1f3&label=pip%20install&logo=python&logoColor=76b1f3)](https://pypi.org/project/unreflectanything/)
[![Paper](https://img.shields.io/badge/Paper-arXiv-B31B1B?logo=arxiv&logoColor=B31B1B)](https://arxiv.org/abs/2512.09583)
[![Demo](https://img.shields.io/badge/Demo-HF%20-FFD21E?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/spaces/AlbeRota/UnReflectAnything-Demo)
[![Modelcard](https://img.shields.io/badge/Model%20Card-HF%20-FFD21E?logo=huggingface&logoColor=FFD21E)](https://huggingface.co/AlbeRota/UnReflectAnything)
[![Oral](https://img.shields.io/badge/Oral-Recording%20-ff0033?logo=youtube&logoColor=ff0033)](https://youtu.be/Uuy9arZwWaM?t=3655)
[![Wiki](https://img.shields.io/badge/API-Wiki-9187FF?logo=wikipedia&logoColor=9187FF)](https://github.com/alberto-rota/UnReflectAnything/wiki)
[![Colab](https://img.shields.io/badge/Examples-Colab-F9AB00?logo=googlecolab&logoColor=F9AB00)](https://colab.research.google.com/#fileId=https%3A//huggingface.co/AlbeRota/UnReflectAnything/blob/main/notebooks/UnReflectAnything.ipynb)
[![Licence](https://img.shields.io/badge/MIT-License-1E811F)](https://mit-license.org/)
### RGB-Only Highlight Removal by Rendering Synthetic Specular Supervision
UnReflectAnything inputs any RGB image and removes specular highlights, returning a clean diffuse-only output. We trained UnReflectAnything by synthetizing specularities and supervising in DINOv3 feature space.

UnReflectAnything works on both natural indoor and surgical/endoscopic domain data. 

---
![examples](https://raw.githubusercontent.com/alberto-rota/UnReflectAnything/refs/heads/main/assets/header.png)

> [!IMPORTANT]
> The maintainers are still working on an official API and Weights release for UnReflectAnything. In v1.0.2 the model is available from the API with `pretrained=False` by deafault, and forecefully setting it `True` will display a warning and weights will not be downloaded. Stay tuned for the official release!

## Installation
```bash
pip install unreflectanything
```
Install UnReflectAnything as a Python Package. 

The minimum required Python version is 3.11, but development and all experiments have been based on **Python 3.12**.

For GPU support, make sure PyTorch comes with CUDA version for your system (see [PyTorch Get Started](https://pytorch.org/get-started/locally/)).


## Setting up
After pip-installing, you can use the `unreflectanything` CLI command, which is also aliased to `unreflect` and `ura`. The three commands are equivalent.

With the CLI you can already download the model weights with
```bash
unreflectanything download --weights
```
and some sample images with 
```bash
unreflectanything download --images
```

Weights are stored by default in `~/.cache/unreflectanything/weights` (or `$XDG_CACHE_HOME/unreflectanything/weights` if set ; `%LOCALAPPDATA%\unreflectanything` for Windows). Use `--output-dir` to choose another location. 

Both the weights and images are stored on the [HuggingFace Model Repo](https://huggingface.co/spaces/AlbeRota/UnReflectAnything-Demo).

## Enable shell completion
Shell completion is available for the `bash` and `zsh` shells. Run
```bash
unreflectanything completion bash
```
and execute the `echo ...` command that gets printed.


## Command Line Interface
Get an overview of the available CLI endpoints with 
```
unreflectanything --help   # alias 'unreflect --help' alias 'ura --help'
```
Refer to the [Wiki](https://github.com/alberto-rota/UnReflectAnything/wiki) to get detailed documentation about each endpoint. We report a summary of the available subcommands. Remember that `ura` is aliased to the `unreflectanything` command

| Subcommand | Description | Command |
|------------|-------------|-------------|
| `inference` | Run inference on image(s) to remove reflections | `ura inference /path/to/images -o /path/to/output` |
| `download` | Download checkpoint weights, sample images, notebooks, configs | `ura download --weights` |
| `cache` | Print cache directory or clear cached assets | `ura cache --dir` or `ura cache --clear` |
| `verify` | Verify weights installation and compatibility, or dataset directory structure | `ura verify --weights` or `ura verify --dataset --path /path/to/dataset` |
| `cite` | Print citation (BibTeX, APA, MLA, IEEE, plain) | `ura cite --bibtex` |
| `completion` | Print or install shell completion (bash/zsh) | `ura completion bash` |

Training, testing, and evaluation are available via the [Python API](https://github.com/alberto-rota/UnReflectAnything/wiki); see the [Wiki](https://github.com/alberto-rota/UnReflectAnything/wiki) for details.

## Python API

The same endpoints above are exposed as a Python API. Refer to the [Wiki](https://github.com/alberto-rota/UnReflectAnything/wiki) to get detailed documentation about each endpoint. A few examples are reported below

```python
import unreflectanything as unreflect
import torch

# Get the model class (e.g. for custom setup or training)
ModelClass = unreflect.model()

# Get a pretrained model (torch.nn.Module) and run on batched RGB
unreflectmodel = unreflect.model(pretrained=True)  # uses cached weights; run 'unreflect download --weights' first
images = torch.rand(2, 3, 448, 448, device="cuda")  # [B, 3, H, W], values in [0, 1]
model_out = unreflectmodel(images)  # [B, 3, H, W] diffuse tensor

# File-based or tensor-based inference (one-shot, no model handle)
unreflect.inference("input.png", output="output.png")
unreflect.inference(images, output="output.png")
result = unreflect.inference(images)

# Cache directory (where weights, images, etc. are stored)
weights_dir = unreflect.cache("weights")
```

## Contributing & Development
If you want to contribute or develop UnReflectAnything:
1. Clone the repository:
   ```bash
   git clone https://github.com/alberto-rota/UnReflectAnything.git
   cd UnReflectAnything
   ```
2. Install dependencies (we recommend a virtual environment with Python 3.12):
   ```bash
   pip install -r requirements.txt
   ```

## Citation
If you include UnReflectAnything in your pipeline or research work, we encourage you cite our work. 
Get the citation entry with 
```bash
unreflectanything cite --bibtex
```
or copy it directly from below
```
@InProceedings{Rota_2026_CVPR,
    author    = {Rota, Alberto and Kiray, Mert and Karaoglu, Mert Asim and Ruhkamp, Patrick and De Momi, Elena and Navab, Nassir and Busam, Benjamin},
    title     = {UnReflectAnything: RGB-Only Highlight Removal by Rendering Synthetic Specular Supervision},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {241-250}
}
```
