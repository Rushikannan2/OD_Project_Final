import torch
import os
from huggingface_hub import hf_hub_download
from vae import VAE
from ldm import get_unet, get_scheduler

# ================================
# HF TOKEN (FROM RENDER ENV)
# ================================
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("❌ HF_TOKEN not set in environment variables")

# ================================
# DEVICE
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ================================
# HUGGING FACE REPO
# ================================
REPO_ID = "rushikannan/OCT-Model"

# ================================
# DOWNLOAD MODELS FROM HF
# ================================
print("⬇ Downloading models from Hugging Face...")

VAE_PATH = hf_hub_download(
    repo_id=REPO_ID,
    filename="VAE_CONDITIONAL_LDM.pth",
    token=HF_TOKEN
)

LDM_PATH = hf_hub_download(
    repo_id=REPO_ID,
    filename="conditional_ldm_best.pth",
    token=HF_TOKEN
)

# ================================
# LOAD VAE
# ================================
print("🔄 Loading VAE...")

vae = VAE().to(device)
vae.load_state_dict(torch.load(VAE_PATH, map_location=device))
vae.eval()

# ================================
# LOAD UNET
# ================================
print("🔄 Loading UNet...")

unet = get_unet().to(device)

ckpt = torch.load(LDM_PATH, map_location=device)

if isinstance(ckpt, dict):

    if "ema_model" in ckpt:
        print("✅ Loading EMA model (BEST)")
        unet.load_state_dict(ckpt["ema_model"])

    elif "model" in ckpt:
        print("⚠ Loading raw model")
        unet.load_state_dict(ckpt["model"])

    else:
        print("⚠ Loading direct state_dict")
        unet.load_state_dict(ckpt)

else:
    unet.load_state_dict(ckpt)

unet.eval()

# ================================
# SCHEDULER
# ================================
scheduler = get_scheduler()

print("✅ Models loaded successfully from Hugging Face!")
