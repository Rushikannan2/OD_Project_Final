# OCT Image Reconstruction using Variational Autoencoder (VAE)

This project implements a Variational Autoencoder (VAE) for grayscale OCT image reconstruction using PyTorch.

The entire pipeline runs locally and includes:

- image preprocessing
- dataset splitting
- PyTorch dataset loader
- VAE training
- validation
- checkpointing
- reconstruction preview
- evaluation

No Kaggle or Colab required. Everything runs offline.

---

## Project Pipeline

```
raw_png
   ↓ preprocessing
processed_png
   ↓ dataset split
dataset/train | val | test
   ↓
PyTorch DataLoader
   ↓
VAE training + evaluation
```

---

## Folder Structure

```
project/
│
├── raw_png/               # original OCT images
├── processed_png/         # resized images
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── vae_outputs/           # training outputs
│   ├── previews/
│   └── checkpoints/
│
├── notebook.ipynb
├── requirements.txt
└── README.md
```

---

## Python Version

Recommended:

```
Python 3.10 or 3.11
```

Python 3.12 is not recommended yet due to PyTorch compatibility.

---

## Installation

### CPU version (works on any laptop)

```
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

### GPU version (CUDA 12.1, NVIDIA GPU)

```
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify GPU:

```python
import torch
print(torch.cuda.is_available())
```

---

## How to Run

1. Put raw OCT images inside:

```
raw_png/
```

2. Run preprocessing section  
   → creates `processed_png`

3. Run dataset split  
   → creates `dataset/train/val/test`

4. Run training section  
   → saves model in `vae_outputs`

5. Run test section  
   → shows reconstruction results

---

## Model Details

- Input size: **512 × 512 grayscale**
- Latent channels: **8**
- Symmetric encoder-decoder architecture
- Loss: **L1 + KL divergence**
- Mixed precision supported on GPU
- Resume training supported
- Deterministic validation pipeline

---

## Output

Training produces:

- epoch preview images
- best model checkpoint
- resume checkpoint
- validation metrics

All saved in:

```
vae_outputs/
```

---

## Compatibility

- Windows ✔
- Linux ✔
- macOS ✔
- CPU ✔
- GPU ✔

---

## Features

- Fully local pipeline (no cloud dependency)
- Notebook-safe paths
- Resume training after crash
- Mixed precision acceleration
- Portable project structure
- Stable library versions

---

## Notes

This project is designed for:

- medical image reconstruction experiments
- VAE research
- diffusion model pretraining
- academic projects

The pipeline is reproducible and experiment-safe.

---

## Author

Student project – OCT VAE pipeline
