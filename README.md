# Eva

## Overview ✨

Eva (**E**ncoding of **v**isual **a**tlas) is a foundation model for tissue imaging data that learns complex spatial representations of tissues at the molecular, cellular, and patient levels. Eva uses a novel vision transformer architecture and is pre-trained on masked image reconstruction of spatial proteomics and matched histopathology. 

### Model Architecture
<img src="figures/model_structure.png" width="80%">

## Installation ⚙️

```bash
git clone https://github.com/YAndrewL/Eva.git
cd Eva

conda env create -f env.yaml
conda activate Eva

pip install -e .  # ~10min
```

## Getting Started 🚀

👉 **Start with the [tutorials](./tutorials/)** for examples and visualizations.

They walk through:
- Loading the model from HuggingFace Hub
- Downloading marker embeddings
- Extracting embeddings
- Working with multi-modality inputs
- Masked prediction

A minimal quick start:
```python
from Eva.utils import load_from_hf, extract_features
from omegaconf import OmegaConf
import torch

conf = OmegaConf.load("config.yaml")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_from_hf(repo_id="yandrewl/Eva", conf=conf, device=device)

patch = torch.randn(1, 224, 224, 6)
biomarkers = ["DAPI", "CD3e", "CD20", "CD4", "CD8", "PanCK"]
features = extract_features(
    patch=patch,
    bms=[biomarkers],
    model=model,
    device=device,
    cls=False,
    channel_mode="full",
)
```


## Configuration 🛠️

The model requires a configuration file (YAML format) that specifies:
- Dataset parameters (patch_size, token_size, marker_dim, etc.)
- Channel mixer parameters (dim, n_layers, n_heads, etc.)
- Patch mixer parameters (dim, n_layers, n_heads, etc.)
- Decoder parameters (dim, n_layers, n_heads, etc.)

See `config.yaml` for an example configuration.


## Citation 📚
Please check Eva paper at [bioRxiv](https://www.biorxiv.org/content/10.64898/2025.12.10.693553v1), and please cite as:

```
@article {Liu2025.12.10.693553,
	author = {Liu, Yufan and Sharma, Rishabh and Bieniosek, Matthew and Kang, Amy and Wu, Eric and Chou, Peter and Li, Irene and Rahim, Maha and Bauer, Erica and Ji, Ran and Duan, Wei and Qian, Li and Luo, Ruibang and Sharma, Padmanee and Dhanasekaran, Renu and Sch{\"u}rch, Christian M. and Charville, Gregory and Mayer, Aaron T. and Zou, James and Trevino, Alexandro E. and Wu, Zhenqin},
	title = {Modeling patient tissues at molecular resolution with Eva},
	elocation-id = {2025.12.10.693553},
	year = {2025},
	doi = {10.64898/2025.12.10.693553},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/12/12/2025.12.10.693553},
	eprint = {https://www.biorxiv.org/content/early/2025/12/12/2025.12.10.693553.full.pdf},
	journal = {bioRxiv}
}
```