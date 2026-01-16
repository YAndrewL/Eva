# Eva 

## Overview

Eva (**E**ncoding of **v**isual **a**tlas) is a foundation model for tissue imaging data that learns complex spatial representations of tissues at the molecular, cellular, and patient levels. Eva uses a novel vision transformer architecture and is pre-trained on masked image reconstruction of spatial proteomics and matched histopathology. 

### Model Architecture
<img src="figures/model_structure.png" width="80%">

## Installation

```bash
git clone https://github.com/YAndrewL/Eva.git
cd Eva

conda env create -f env.yaml
conda activate Eva

pip install -e .  # ~10min
```

## Getting Started

For example and visualization, please check [tutorials](./tutorials/).

### Loading the Model

Eva model weights are open-sourced on [HuggingFace Hub](https://huggingface.co/yandrewl/Eva).

```python
from Eva.utils import load_from_hf, extract_features, create_model
from omegaconf import OmegaConf
import torch

# Load configuration
conf = OmegaConf.load("config.yaml")

# Load model from HuggingFace Hub
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_from_hf(
    repo_id="yandrewl/Eva",
    conf=conf,
    device=device
)
# ~20s
```

### Downloading Marker Embeddings

Download the GenePT marker embeddings from [Zenodo record](https://zenodo.org/records/10833191). Use the file **GenePT_gene_protein_embedding_model_3_text.pickle** and store it as **marker_embeddings/marker_embedding.pkl** locally.

### Extracting Embeddings
```python

# Extract embeddings
patch = torch.randn(1, 224, 224, 6)  # Shape: [B, H, W, C]
biomarkers = ["DAPI", "CD3e", "CD20", "CD4", "CD8", "PanCK"]  # biomarkers

features = extract_features(
    patch=patch,
    bms=[biomarkers],  # List of biomarker lists (one per batch item)
    model=model,
    device=device,
    cls=False,  # Use CLS token (True) or average patches (False)
    channel_mode="full"  # Options: "full", "HE", "MIF"
)
# ~ 31ms
```

### Multi-modality Inputs

When data include H&E (Hematoxylin and Eosin) channels, H&E should be added as the last three channels:

```python
mif_patch = torch.randn(1, 224, 224, 6) 
he_patch = torch.randn(1, 224, 224, 3)
patch = torch.cat([mif_patch, he_patch], dim=-1)
biomarkers = ["DAPI", "CD3e", "CD20", "CD4", "CD8", "PanCK", "HECHA1", "HECHA2", "HECHA3"]  # Last 3 are HE channels

# Extract features using different modality
features = extract_features(
    patch=patch,
    bms=[biomarkers],
    model=model,
    device=device,
    cls=False,
    channel_mode="MIF",  # Set to "HE" to use HE channels only, or "full" to use all channels
)
```


## Configuration

The model requires a configuration file (YAML format) that specifies:
- Dataset parameters (patch_size, token_size, marker_dim, etc.)
- Channel mixer parameters (dim, n_layers, n_heads, etc.)
- Patch mixer parameters (dim, n_layers, n_heads, etc.)
- Decoder parameters (dim, n_layers, n_heads, etc.)

See `config.yaml` for an example configuration.


## Citation
Please check Eva paper at [bioRxiv](https://www.biorxiv.org/content/10.64898/2025.12.10.693553v1), and cite as:

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