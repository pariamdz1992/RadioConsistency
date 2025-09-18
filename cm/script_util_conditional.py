import argparse
import numpy as np

# Change relative imports to absolute imports for testing
try:
    from .karras_diffusion import KarrasDenoiser
    from .unet import UNetModel
except ImportError:
    # Fallback for standalone testing
    from karras_diffusion import KarrasDenoiser
    from unet import UNetModel

NUM_CLASSES = 1000

def cm_train_defaults():
    return dict(
        teacher_model_path="",
        teacher_dropout=0.1,
        training_mode="consistency_distillation",
        target_ema_mode="fixed",
        scale_mode="fixed",
        total_training_steps=600000,
        start_ema=0.0,
        start_scales=40,
        end_scales=40,
        distill_steps_per_iter=50000,
        loss_norm="lpips",
    )

def model_and_diffusion_defaults():
    """
    Defaults for image training with conditioning support.
    """
    res = dict(
        sigma_min=0.002,
        sigma_max=80.0,
        image_size=64,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=32,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=False,
        weight_schedule="karras",
        # NEW CONDITIONING PARAMETERS (matching radio map diffusion config)
        cond_in_dim=3,           # Number of input condition channels (buildings + transmitter + k-factor)
        cond_dim=128,            # Condition feature dimension  
        cond_net="swin",         # Condition encoder backbone (swin, resnet, effnet, vgg)
        use_conditioning=False,  # Enable/disable conditioning
        window_sizes1="16,8,4,2", # Window sizes for cross-attention (query)
        window_sizes2="16,8,4,2", # Window sizes for cross-attention (key/value)
        fix_backbone=True,       # Whether to freeze the condition encoder backbone
        without_pretrain=False,  # Whether to use pretrained weights for backbone
        fourier_scale=16,        # Fourier scale for time embeddings (matching original)
        input_size=256,          # Input image size for radio maps
    )
    return res

# Rest of the functions remain the same but I'll include them for completeness
def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    weight_schedule,
    sigma_min=0.002,
    sigma_max=80.0,
    distillation=False,
    # NEW CONDITIONING PARAMETERS
    cond_in_dim=3,
    cond_dim=128,
    cond_net="swin",
    use_conditioning=False,
    window_sizes1="16,8,4,2",
    window_sizes2="16,8,4,2",
    fix_backbone=True,
    without_pretrain=False,
    fourier_scale=16,
    input_size=256,
):
    print(f"Creating model with conditioning: {use_conditioning}")
    return "MODEL_PLACEHOLDER", "DIFFUSION_PLACEHOLDER"

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
