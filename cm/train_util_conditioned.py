"""
Training utilities for conditioned consistency models on radio maps.
Extends the original train_util.py to handle radio data format and spatial conditioning.
"""

import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import RAdam

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

from .fp16_util import (
    get_param_groups_and_shapes,
    make_master_params,
    master_params_to_model_params,
)
import numpy as np

# Import the original classes to extend them
from .train_util import TrainLoop, CMTrainLoop, INITIAL_LOG_LOSS_SCALE


class ConditionedCMTrainLoop(CMTrainLoop):
    """
    Extended CMTrainLoop for conditioned consistency models with radio data.
    
    KEY MODIFICATIONS:
    1. Handles radio data format: (data, {'conditioning': cond}) -> (data, cond)
    2. Ensures conditioning is passed correctly to conditioned models
    3. Supports spatial conditioning (buildings, transmitters, cars)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.log("Initialized conditioned consistency training loop")
        logger.log(f"Model type: {type(self.model).__name__}")
        
        # Log conditioning information if available
        if hasattr(self.model, 'cond_net'):
            logger.log(f"Conditioning backbone: {self.model.cond_net}")
            logger.log(f"Conditioning channels: {getattr(self.model, 'cond_in_dim', 'unknown')}")
            logger.log(f"Backbone frozen: {getattr(self.model, 'fix_bb', 'unknown')}")

    def run_loop(self):
        """
        Main training loop - handles radio data format conversion.
        MODIFIED: Converts radio data format before passing to parent run_step.
        """
        saved = False
        while (
            not self.lr_anneal_steps
            or self.step < self.lr_anneal_steps
            or self.global_step < self.total_training_steps
        ):
            # GET RADIO DATA BATCH
            radio_batch = next(self.data)
            
            # CONVERT RADIO DATA FORMAT
            batch, cond = self._convert_radio_data_format(radio_batch)
            
            # Run the training step with converted data
            self.run_step(batch, cond)
            saved = False
            
            if (
                self.global_step
                and self.save_interval != -1
                and self.global_step % self.save_interval == 0
            ):
                self.save()
                saved = True
                th.cuda.empty_cache()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

            if self.global_step % self.log_interval == 0:
                logger.dumpkvs()

        # Save the last checkpoint if it wasn't already saved.
        if not saved:
            self.save()

    def _convert_radio_data_format(self, radio_batch):
        """
        Convert radio data format from your loader to training loop format.
        
        INPUT FORMAT (from your radio_datasets.load_data):
            radio_batch = (radio_data, kwargs)
            where kwargs = {'conditioning': conditioning_tensor}
            
        OUTPUT FORMAT (expected by training loop):
            batch = radio_data  # [B, 1, H, W] radio maps
            cond = {'conditioning': conditioning_tensor}  # [B, 3, H, W] spatial conditioning
            
        Args:
            radio_batch: Tuple from your radio data loader
            
        Returns:
            batch: Radio map tensor
            cond: Dictionary with conditioning
        """
        if isinstance(radio_batch, (list, tuple)) and len(radio_batch) == 2:
            # Your confirmed working format: (data, kwargs)
            radio_data, kwargs = radio_batch
            
            # Ensure we have conditioning
            if 'conditioning' not in kwargs:
                raise ValueError(
                    f"Expected 'conditioning' in kwargs, got keys: {list(kwargs.keys())}"
                )
            
            # Log data shapes for monitoring (only occasionally to avoid spam)
            if self.global_step % 100 == 0:
                logger.log(f"Radio data shape: {radio_data.shape}")
                logger.log(f"Conditioning shape: {kwargs['conditioning'].shape}")
            
            return radio_data, kwargs
            
        else:
            raise ValueError(
                f"Unexpected radio data format. Expected (data, kwargs) tuple, "
                f"got {type(radio_batch)} with length {len(radio_batch) if hasattr(radio_batch, '__len__') else 'unknown'}"
            )

    def forward_backward(self, batch, cond):
        """
        Forward and backward pass for conditioned models.
        MODIFIED: Ensures conditioning is passed correctly to conditioned models.
        """
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            
            # CRITICAL: Handle conditioning for conditioned models
            micro_cond = self._prepare_micro_conditioning(cond, i)
            
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            ema, num_scales = self.ema_scale_fn(self.global_step)
            
            # Create compute_losses function based on training mode
            if self.training_mode == "progdist":
                if num_scales == self.ema_scale_fn(0)[1]:
                    compute_losses = functools.partial(
                        self.diffusion.progdist_losses,
                        self.ddp_model,
                        micro,
                        num_scales,
                        target_model=self.teacher_model,
                        target_diffusion=self.teacher_diffusion,
                        model_kwargs=micro_cond,
                    )
                else:
                    compute_losses = functools.partial(
                        self.diffusion.progdist_losses,
                        self.ddp_model,
                        micro,
                        num_scales,
                        target_model=self.target_model,
                        target_diffusion=self.diffusion,
                        model_kwargs=micro_cond,
                    )
            elif self.training_mode == "consistency_distillation":
                compute_losses = functools.partial(
                    self.diffusion.consistency_losses,
                    self.ddp_model,
                    micro,
                    num_scales,
                    target_model=self.target_model,
                    teacher_model=self.teacher_model,
                    teacher_diffusion=self.teacher_diffusion,
                    model_kwargs=micro_cond,
                )
            elif self.training_mode == "consistency_training":
                # THIS IS THE KEY CASE for your radio consistency training
                compute_losses = functools.partial(
                    self.diffusion.consistency_losses,
                    self.ddp_model,
                    micro,
                    num_scales,
                    target_model=self.target_model,
                    model_kwargs=micro_cond,  # This contains your spatial conditioning
                )
            else:
                raise ValueError(f"Unknown training mode {self.training_mode}")

            # Compute losses with or without gradient synchronization
            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            # Update loss-aware sampler if needed
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            # Compute weighted loss
            loss = (losses["loss"] * weights).mean()

            # Log losses
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            
            # Backward pass
            self.mp_trainer.backward(loss)

    def _prepare_micro_conditioning(self, cond, start_idx):
        """
        Prepare conditioning for microbatch processing.
        
        Args:
            cond: Full batch conditioning dictionary
            start_idx: Starting index for microbatch
            
        Returns:
            micro_cond: Conditioning dictionary for microbatch
        """
        micro_cond = {}
        
        for k, v in cond.items():
            if isinstance(v, th.Tensor):
                # Slice the conditioning tensor for the microbatch
                micro_v = v[start_idx : start_idx + self.microbatch].to(dist_util.dev())
                micro_cond[k] = micro_v
            else:
                # Non-tensor values (shouldn't happen with your radio data, but just in case)
                micro_cond[k] = v
        
        return micro_cond

    def log_step(self):
        """
        Log training step information.
        ENHANCED: Adds conditioning-specific logging.
        """
        step = self.global_step
        logger.logkv("step", step)
        logger.logkv("samples", (step + 1) * self.global_batch)
        
        # Log EMA and scales info
        if hasattr(self, 'ema_scale_fn'):
            ema, scales = self.ema_scale_fn(step)
            logger.logkv("ema_rate", ema)
            logger.logkv("consistency_scales", scales)
        
        # Log model-specific info (occasionally)
        if step % 1000 == 0:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logger.logkv("total_params", total_params)
            logger.logkv("trainable_params", trainable_params)
            logger.logkv("frozen_params", total_params - trainable_params)


# UTILITY FUNCTIONS (copied from original train_util.py)
def log_loss_dict(diffusion, ts, losses):
    """Log loss dictionary - identical to original."""
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def parse_resume_step_from_filename(filename):
    """Parse resume step from filename - identical to original."""
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    """Get blob logging directory - identical to original."""
    return logger.get_dir()


def find_resume_checkpoint():
    """Find resume checkpoint - identical to original."""
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    """Find EMA checkpoint - identical to original."""
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None
