import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================
# SAFE GROUP NORM
# ================================

def norm_layer(ch):
    return nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)

# ================================
# RESIDUAL BLOCK
# ================================

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()

        self.norm1 = norm_layer(ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)

        self.norm2 = norm_layer(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)

        self.act = nn.SiLU()

    def forward(self, x):
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        return x + h


# ================================
# DOWN BLOCK
# ================================

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, 4, 2, 1)
        self.norm = norm_layer(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# ================================
# UP BLOCK
# ================================

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1)
        self.norm = norm_layer(out_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# ================================
# VAE MODEL
# ================================

class VAE(nn.Module):
    def __init__(self, im_channels=1, base_ch=128, z_channels=32):
        super().__init__()

        # ---------------- Encoder ----------------
        self.conv_in = nn.Conv2d(im_channels, base_ch, 3, padding=1)

        self.down1 = DownBlock(base_ch, base_ch * 2)
        self.res1  = ResBlock(base_ch * 2)

        self.down2 = DownBlock(base_ch * 2, base_ch * 4)
        self.res2  = ResBlock(base_ch * 4)

        self.down3 = DownBlock(base_ch * 4, base_ch * 4)
        self.res3  = ResBlock(base_ch * 4)

        mid_ch = base_ch * 4

        # Latent stats
        self.to_stats = nn.Conv2d(mid_ch, z_channels * 2, 3, padding=1)

        # ---------------- Decoder ----------------
        self.from_latent = nn.Conv2d(z_channels, mid_ch, 3, padding=1)

        self.res4 = ResBlock(mid_ch)

        self.up1 = UpBlock(mid_ch, base_ch * 4)
        self.res5 = ResBlock(base_ch * 4)

        self.up2 = UpBlock(base_ch * 4, base_ch * 2)
        self.res6 = ResBlock(base_ch * 2)

        self.up3 = UpBlock(base_ch * 2, base_ch)
        self.res7 = ResBlock(base_ch)

        self.norm_out = norm_layer(base_ch)
        self.conv_out = nn.Conv2d(base_ch, im_channels, 3, padding=1)

    # ================================
    # ENCODE
    # ================================

    def encode(self, x):
        x = self.conv_in(x)

        x = self.res1(self.down1(x))
        x = self.res2(self.down2(x))
        x = self.res3(self.down3(x))

        mean, logvar = torch.chunk(self.to_stats(x), 2, dim=1)

        logvar = torch.clamp(logvar, -10, 10)  # stability

        return mean, logvar

    # ================================
    # REPARAMETERIZATION (OPTIONAL)
    # ================================

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    # ================================
    # DECODE
    # ================================

    def decode(self, z):
        x = self.from_latent(z)

        x = self.res4(x)

        x = self.res5(self.up1(x))
        x = self.res6(self.up2(x))
        x = self.res7(self.up3(x))

        x = self.conv_out(F.silu(self.norm_out(x)))

        return torch.tanh(x)

    # ================================
    # FORWARD
    # ================================

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)

        return recon, mean, logvar