import argparse

from .karras_diffusion import KarrasDenoiser
from .unet import UNetModel
from .conditioned_consistency_unet import ConditionedConsistencyUNet
import numpy as np

NUM_CLASSES = 1000


def cm_train_defaults():
    """Consistency model training defaults - unchanged from original."""
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
    Defaults for conditioned radio map training.
    
    MAJOR CHANGES from original:
    1. Added conditioning parameters (cond_net, cond_in_dim, fix_bb)
    2. Changed default image_size to 64 (matching your confirmed radio data)
    3. Added in_channels and out_channels for radio data (1 channel)
    """
    res = dict(
        sigma_min=0.002,
        sigma_max=80.0,
        image_size=64,  # Changed from original - matches your radio data
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
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
        # NEW CONDITIONING PARAMETERS for radio maps
        cond_net="swin",        # backbone: swin, effnet, resnet, vgg
        cond_in_dim=3,          # buildings, transmitters, cars
        fix_bb=True,            # freeze backbone weights
        in_channels=1,          # radio maps are single channel
        out_channels=1,         # radio map output is single channel
        use_conditioned_model=True,  # flag to use conditioned vs standard model
    )
    return res


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
    cond_net="swin",
    cond_in_dim=3,
    fix_bb=True,
    in_channels=1,
    out_channels=1,
    use_conditioned_model=True,
):
    """
    Create model and diffusion for conditioned radio map training.
    
    MAJOR CHANGE: This function now creates either:
    1. ConditionedConsistencyUNet (for radio maps with spatial conditioning)
    2. Standard UNetModel (fallback for other data)
    """
    
    if use_conditioned_model:
        # Use our conditioned model for radio maps
        model = create_conditioned_model(
            image_size=image_size,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            learn_sigma=learn_sigma,
            class_cond=class_cond,
            use_checkpoint=use_checkpoint,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            dropout=dropout,
            resblock_updown=resblock_updown,
            use_fp16=use_fp16,
            use_new_attention_order=use_new_attention_order,
            cond_net=cond_net,
            cond_in_dim=cond_in_dim,
            fix_bb=fix_bb,
            in_channels=in_channels,
            out_channels=out_channels,
        )
    else:
        # Fallback to standard model
        model = create_model(
            image_size=image_size,
            num_channels=num_channels,
            num_res_blocks=num_res_blocks,
            channel_mult=channel_mult,
            learn_sigma=learn_sigma,
            class_cond=class_cond,
            use_checkpoint=use_checkpoint,
            attention_resolutions=attention_resolutions,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            dropout=dropout,
            resblock_updown=resblock_updown,
            use_fp16=use_fp16,
            use_new_attention_order=use_new_attention_order,
        )
    
    # Diffusion schedule remains the same
    diffusion = KarrasDenoiser(
        sigma_data=0.5,
        sigma_max=sigma_max,
        sigma_min=sigma_min,
        distillation=distillation,
        weight_schedule=weight_schedule,
    )
    return model, diffusion


def create_conditioned_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    cond_net="swin",
    cond_in_dim=3,
    fix_bb=True,
    in_channels=1,
    out_channels=1,
):
    """
    Create the conditioned consistency model for radio maps.
    
    THIS IS THE NEW FUNCTION that creates ConditionedConsistencyUNet
    instead of the standard UNetModel.
    """
    
    # Handle channel multipliers (same logic as original)
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)  # Your radio data size
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    # Handle attention resolutions (same logic as original)
    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    # CREATE CONDITIONED MODEL instead of standard UNet
    return ConditionedConsistencyUNet(
        # Core U-Net parameters (adapted from UNetModel)
        dim=num_channels,                    # num_channels -> dim
        dim_mults=channel_mult,              # channel_mult -> dim_mults
        channels=in_channels,                # in_channels -> channels (radio maps = 1)
        out_dim=out_channels,                # out_channels -> out_dim (radio maps = 1)
        resnet_block_groups=8,               # standard groupnorm groups
        
        # CONDITIONING PARAMETERS (new for radio maps)
        cond_in_dim=cond_in_dim,            # 3 channels: buildings, transmitters, cars
        cond_net=cond_net,                  # swin, effnet, resnet, vgg backbone
        fix_bb=fix_bb,                      # freeze backbone weights
        
        # Size and architecture
        input_size=[image_size, image_size], # [64, 64] for your radio data
        
        # Advanced parameters (some adapted, some new defaults)
        window_sizes1=[[8, 8], [4, 4], [2, 2], [1, 1]],  # cross-attention windows
        window_sizes2=[[8, 8], [4, 4], [2, 2], [1, 1]],  # cross-attention windows
        
        # Note: Some UNet parameters like num_heads, attention_resolutions 
        # are handled differently in ConditionedConsistencyUNet
        # The model uses its own attention mechanism with RelationNet
    )


def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
):
    """
    Create standard UNet model (fallback - unchanged from original).
    
    KEPT UNCHANGED for compatibility with non-radio datasets.
    """
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,  # Standard RGB
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),  # Standard RGB
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )


# ALL FUNCTIONS BELOW ARE UNCHANGED from original script_util.py
# (keeping them for compatibility)

def create_ema_and_scales_fn(
    target_ema_mode,
    start_ema,
    scale_mode,
    start_scales,
    end_scales,
    total_steps,
    distill_steps_per_iter,
):
    def ema_and_scales_fn(step):
        if target_ema_mode == "fixed" and scale_mode == "fixed":
            target_ema = start_ema
            scales = start_scales
        elif target_ema_mode == "fixed" and scale_mode == "progressive":
            target_ema = start_ema
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            scales = scales + 1

        elif target_ema_mode == "adaptive" and scale_mode == "progressive":
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            c = -np.log(start_ema) * start_scales
            target_ema = np.exp(-c / scales)
            scales = scales + 1
        elif target_ema_mode == "fixed" and scale_mode == "progdist":
            distill_stage = step // distill_steps_per_iter
            scales = start_scales // (2**distill_stage)
            scales = np.maximum(scales, 2)

            sub_stage = np.maximum(
                step - distill_steps_per_iter * (np.log2(start_scales) - 1),
                0,
            )
            sub_stage = sub_stage // (distill_steps_per_iter * 2)
            sub_scales = 2 // (2**sub_stage)
            sub_scales = np.maximum(sub_scales, 1)

            scales = np.where(scales == 2, sub_scales, scales)

            target_ema = 1.0
        else:
            raise NotImplementedError

        return float(target_ema), int(scales)

    return ema_and_scales_fn


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
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
