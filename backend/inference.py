import torch
import torch.nn.functional as F
import cv2
import os

from model_loader import vae, unet, scheduler, device
from torchvision.utils import save_image

# ================================
# DEVICE INFO
# ================================
print(f"🚀 Using device: {device}")

# ================================
# FIXED SEED
# ================================
torch.manual_seed(42)

# ================================
# SOBEL KERNELS
# ================================
sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=device).view(1,1,3,3).float()
sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], device=device).view(1,1,3,3).float()

# ================================
# CONDITION FUNCTION
# ================================
def get_condition(img):
    img = (img + 1) / 2
    img = F.avg_pool2d(img, 3, stride=1, padding=1)

    gx = F.conv2d(img, sobel_x, padding=1)
    gy = F.conv2d(img, sobel_y, padding=1)

    edges = torch.sqrt(gx**2 + gy**2 + 1e-8)
    edges = edges / (edges.amax(dim=[1,2,3], keepdim=True) + 1e-8)
    edges = edges * 2 - 1

    edges = F.avg_pool2d(edges, 3, stride=1, padding=1)  # smoother conditioning

    return F.interpolate(edges, size=(64,64))

# ================================
# GENERATION FUNCTION
# ================================
@torch.no_grad()
def generate_image():

    print("⚡ Generating OCT image (HF integrated)...")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    static_dir = os.path.join(BASE_DIR, "static")
    input_path = os.path.join(static_dir, "sample.png")
    output_path = os.path.join(static_dir, "output.png")

    os.makedirs(static_dir, exist_ok=True)

    # ================================
    # LOAD IMAGE
    # ================================
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ Missing input image: {input_path}")

    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("❌ OpenCV failed to load image")

    img = cv2.resize(img, (512,512))

    img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    img = (img / 255.0) * 2 - 1
    img = img.to(device)

    # ================================
    # CONDITION
    # ================================
    cond = get_condition(img)

    # ================================
    # LATENT INIT (UNCHANGED)
    # ================================
    mean, _ = vae.encode(img)
    z = torch.randn_like(mean)   # ✅ keeping your original logic

    # ================================
    # SAMPLING LOOP
    # ================================
    scheduler.set_timesteps(100)

    for t in scheduler.timesteps:
        z_input = torch.cat([z, cond], dim=1)
        noise_pred = unet(z_input, t).sample
        z = scheduler.step(noise_pred, t, z).prev_sample

    # ================================
    # DECODE
    # ================================
    out = vae.decode(z)
    out = torch.clamp((out + 1) / 2, 0, 1)

    # ================================
    # SAVE OUTPUT
    # ================================
    save_image(out.cpu(), output_path)  # safer for CPU/GPU

    print(f"💾 Output saved at: {output_path}")

    return output_path
