from diffusers import UNet2DModel, DDIMScheduler

# ================================
# UNET (CORRECT)
# ================================

def get_unet():
    return UNet2DModel(
        sample_size=64,
        in_channels=33,   # 32 latent + 1 condition
        out_channels=32,

        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),

        down_block_types=(
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D"
        ),

        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D"
        )
    )

# ================================
# SCHEDULER (🔥 CRITICAL FIX)
# ================================

def get_scheduler():
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2"   # 🔥 MUST MATCH TRAINING
    )

    scheduler.set_timesteps(100)  # 🔥 match your inference

    return scheduler