import torch
import os
from huggingface_hub import hf_hub_download
from backend.vae import VAE
from backend.ldm import get_unet, get_scheduler

# ================================
# DEVICE
# ================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# ================================
# HF TOKEN
# ================================
HF_TOKEN = os.getenv("HF_TOKEN")

# ================================
# HUGGING FACE REPO
# ================================
REPO_ID = "rushikannan/OCT-Model"

# ================================
# GLOBAL CACHE (LAZY LOAD)
# ================================
vae = None
unet = None
scheduler = None

# ================================
# LOAD MODELS (LAZY)
# ================================
def load_models():
    global vae, unet, scheduler

    if vae is not None:
        return vae, unet, scheduler  # already loaded

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

    print("🔄 Loading VAE...")
    vae_model = VAE().to(device)
    vae_model.load_state_dict(torch.load(VAE_PATH, map_location=device))
    vae_model.eval()

    print("🔄 Loading UNet...")
    unet_model = get_unet().to(device)

    ckpt = torch.load(LDM_PATH, map_location=device)

    if isinstance(ckpt, dict):
        if "ema_model" in ckpt:
            print("✅ Loading EMA model (BEST)")
            unet_model.load_state_dict(ckpt["ema_model"])
        elif "model" in ckpt:
            print("⚠ Loading raw model")
            unet_model.load_state_dict(ckpt["model"])
        else:
            print("⚠ Loading direct state_dict")
            unet_model.load_state_dict(ckpt)
    else:
        unet_model.load_state_dict(ckpt)

    unet_model.eval()

    sched = get_scheduler()

    # store globally
    vae = vae_model
    unet = unet_model
    scheduler = sched

    print("✅ Models loaded successfully!")

    return vae, unet, scheduler