# Eva 

## Overview

Eva (**E**ncoding of **v**isual **a**tlas) is a foundation model for tissue imaging data that learns complex spatial representations of tissues at the molecular, cellular, and patient level. Eva uses a novel vision transformer architecture and is pretrained on masked image reconstruction of spatial proteomics and matched histopathology. 

### Model Architecture
<img src="figures/model_structure.png" width="60%">

## Installation

```bash
# Clone the repository
git clone https://github.com/YAndrewL/Eva.git
cd Eva

# Create and activate conda environment
conda env create -f env.yaml
conda activate Eva

# Install the package
pip install -e .
```

## Getting Started

### Loading the Model

Eva model weights are open-sourced on [HuggingFace](https://huggingface.co/yandrewl/Eva).

```python
from Eva.utils import load_from_hf, extract_features, create_model
from omegaconf import OmegaConf
import torch

# Load configuration
conf = OmegaConf.load("config.yaml")

# Load model from HuggingFace
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_from_hf(
    repo_id="yandrewl/Eva",
    conf=conf,
    device=device
)
```

### Downloading Marker Embeddings

Download the GenePT marker embeddings from [Zenodo record](https://zenodo.org/records/10833191). Use the file **GenePT_gene_protein_embedding_model_3_text.pickle** and store it as **marker_embeddings/marker_embedding.pkl**.

### Generating Embeddings
```python
# Or use the convenience function
# model = create_model(conf, repo_id="your-username/eva-base", device=device)

# Extract embeddings
patch = torch.randn(1, 224, 224, 6)  # Shape: [B, H, W, C]
biomarkers = ["DAPI", "CD3e", "CD20", "CD4", "CD8", "PanCK"]  # List of biomarker names

features = extract_features(
    patch=patch,
    bms=[biomarkers],  # List of biomarker lists (one per batch item)
    model=model,
    device=device,
    cls=False,  # Use CLS token (True) or average patches (False)
    channel_mode="full"  # Options: "full", "HE", "MIF"
)

# or use model method
features = model.extract_features(
    patch=patch,
    bms=[biomarkers],
    device=device,
    cls=False,
    channel_mode="full"  # Options: "full", "HE", "MIF"
)
```

### Multi-modality Inputs

When data includes H&E (Hematoxylin and Eosin) channels, H&E should be added as the last three channels:

```python
mif_patch = torch.randn(1, 224, 224, 6) 
he_patch = torch.randn(1, 224, 224, 3)
patch = torch.cat([mif_patch, he_patch], dim=-1)
biomarkers = ["DAPI", "CD3e", "CD20", "CD4", "CD8", "PanCK", "HECHA1", "HECHA2", "HECHA3"]  # Last 3 are HE channels

# Extract features using MIF channels only
features = extract_features(
    patch=patch,
    bms=[biomarkers],
    model=model,
    device=device,
    cls=False,
    channel_mode="MIF"  # or "HE" to use HE channels only.
)
```


## Configuration

The model requires a configuration file (YAML format) that specifies:
- Dataset parameters (patch_size, token_size, marker_dim, etc.)
- Channel mixer parameters (dim, n_layers, n_heads, etc.)
- Patch mixer parameters (dim, n_layers, n_heads, etc.)
- Decoder parameters (dim, n_layers, n_heads, etc.)

See `config.yaml` for an example configuration.

## References