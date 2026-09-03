"""Diffusion Policy — LIBRARY edition (contrast to the from-scratch model.py).

Same network, same math, same DiffusionPolicy interface (compute_loss /
select_action / reset). The point of this file is pedagogical: show which pieces
of model.py you can hand off to a battle-tested library, and — just as important
— which pieces you CANNOT, because no library ships them.

WHAT GOT REPLACED BY A LIBRARY
  1. NoiseScheduler (cosine betas, add_noise, ddpm_step, ddim_step, _predict_x0)
       -> diffusers.DDPMScheduler / diffusers.DDIMScheduler
       This is the biggest win: ~150 lines of derived DDPM/DDIM math collapse
       into two constructor calls. The library also handles edge cases we glossed
       over (variance types, thresholding, karras sigmas, etc.).
  2. SinusoidalPosEmb + the 2-layer timestep MLP
       -> diffusers.models.embeddings.Timesteps + TimestepEmbedding
       Timesteps = the fixed sinusoidal encoding; TimestepEmbedding = the learned
       Linear->act->Linear head. Exactly what we hand-rolled.
  3. EMAModel (decay warmup, shadow params, copy_to)
       -> diffusers.training_utils.EMAModel  (wrapped to keep our train.py API)

WHAT STAYED CUSTOM (no library equivalent — this is the real lesson)
  - SpatialSoftmax: there is NO SpatialSoftmax in torch, torchvision, or diffusers.
    The only off-the-shelf one lives in robomimic (robomimic.models.base_nets),
    a heavy robotics-specific dependency. So for the vision encoder you either
    keep the custom layer (what we do — imported unchanged from model.py) or take
    on robomimic. Good to know before you go looking for a one-liner that doesn't exist.
  - The FiLM conditional 1D U-Net: diffusers DOES ship a UNet1DModel, but its
    conditioning is designed for a different setup (RL trajectory / value guidance),
    not Diffusion Policy's "one global observation vector FiLM-modulates every
    block". Forcing it would be more code than keeping ours. So the U-Net body
    stays custom; only its timestep encoder is swapped to the diffusers version.
  - Min/max & mean/std normalization buffers, and the BatchNorm->GroupNorm swap:
    trivial arithmetic / a recursive walk. No cleaner library form.

USAGE
  Drop-in for train.py / rollout.py — change the import line only:
      from model_lib import DiffusionPolicy, EMAModel   # instead of `from model import ...`
"""
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# --- library components (the whole point of this file) -----------------------
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.models.embeddings import Timesteps, TimestepEmbedding
from diffusers.training_utils import EMAModel as _DiffusersEMA

from config import DiffusionConfig

# --- unchanged custom pieces, imported straight from the from-scratch model --
# These have no library equivalent (SpatialSoftmax) or no cleaner library form
# (normalization, conv blocks, the FiLM residual block). Reusing them keeps this
# file focused on the parts that actually differ.
from model import (
    MinMaxNormalize,
    MinMaxUnnormalize,
    MeanStdNormalize,
    DiffusionRgbEncoder,          # ResNet18(GroupNorm) + SpatialSoftmax  (custom!)
    Conv1dBlock,
    ConditionalResidualBlock1D,   # the FiLM residual block
    Downsample1d,
    Upsample1d,
)


# =============================================================================
# Conditional 1D U-Net — same body as model.py, but the timestep encoder is
# now the diffusers Timesteps + TimestepEmbedding pair instead of our hand-rolled
# SinusoidalPosEmb + MLP.
# =============================================================================
class ConditionalUnet1DLib(nn.Module):
    """FiLM-conditioned 1D U-Net whose diffusion-step encoder comes from diffusers.

    Compare with ConditionalUnet1D in model.py: everything below the timestep
    encoder is identical (same down/mid/up FiLM blocks). Only the 4 lines that
    turn an integer timestep into a conditioning vector changed.
    """

    def __init__(self, cfg: DiffusionConfig, global_cond_dim: int):
        super().__init__()
        input_dim = cfg.action_dim
        dsed = cfg.diffusion_step_embed_dim

        # --- Diffusion timestep encoder (LIBRARY) ------------------------------
        # Timesteps: fixed sinusoidal encoding of the scalar timestep.
        #   flip_sin_to_cos + downscale_freq_shift=0 reproduce the common DDPM
        #   convention (cos first, then sin) used across diffusers models.
        # TimestepEmbedding: the learned MLP (Linear -> SiLU -> Linear) on top.
        # Together these replace our SinusoidalPosEmb + 2-layer Mish MLP.
        self.time_proj = Timesteps(num_channels=dsed, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedding = TimestepEmbedding(in_channels=dsed, time_embed_dim=dsed)

        cond_dim = dsed + global_cond_dim

        # --- Encoder (down) path -- identical to model.py ----------------------
        # The U-Net convolves over the TIME axis of the action trajectory (length
        # = horizon), progressively halving time-resolution while growing channels.
        # all_dims = channel widths at each level, e.g. [8, 256, 512, 1024]:
        #   8 = action_dim (input), then cfg.down_dims = (256, 512, 1024).
        all_dims = [input_dim] + list(cfg.down_dims)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))   # [(8,256),(256,512),(512,1024)]
        mid_dim = all_dims[-1]                              # bottleneck width (1024)
        # Shared kwargs for every FiLM residual block. cond_dim = timestep-embed +
        # obs vector; each block uses it to FiLM-modulate its conv activations.
        blk = dict(kernel_size=cfg.kernel_size, n_groups=cfg.n_groups,
                   use_film_scale_modulation=cfg.use_film_scale_modulation)

        # Down path: two residual blocks per level, then halve time (except the
        # deepest level, which hands straight to the bottleneck via Identity).
        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, **blk),   # change channels
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, **blk),  # refine
                Downsample1d(dim_out) if not is_last else nn.Identity(),        # T -> T/2
            ]))

        # Bottleneck: two blocks at the deepest width, no resolution change.
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, **blk),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, **blk),
        ])

        # Up path: mirror of down. Each level concatenates the matching skip
        # connection (hence dim_out*2 input channels), then upsamples T -> 2T.
        # reversed(in_out[1:]) walks levels back out: [(512,1024),(256,512)].
        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim, **blk),  # *2 = skip concat
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, **blk),
                Upsample1d(dim_in) if not is_last else nn.Identity(),             # T -> 2T
            ]))

        # Project back to action_dim channels at full time-resolution = predicted noise.
        self.final_conv = nn.Sequential(
            Conv1dBlock(cfg.down_dims[0], cfg.down_dims[0], cfg.kernel_size, cfg.n_groups),
            nn.Conv1d(cfg.down_dims[0], input_dim, kernel_size=1),
        )

    def forward(self, sample: Tensor, timestep: Tensor, global_cond: Tensor) -> Tensor:
        """sample: (B, horizon, action_dim); timestep: (B,); global_cond: (B, cond)."""
        import einops
        x = einops.rearrange(sample, "b t c -> b c t")

        # LIBRARY timestep encoding (replaces the from-scratch SinusoidalPosEmb+MLP).
        t_emb = self.time_proj(timestep)                 # (B, dsed) sinusoidal
        t_emb = t_emb.to(dtype=sample.dtype)
        t_feat = self.time_embedding(t_emb)              # (B, dsed) learned
        global_feature = torch.cat([t_feat, global_cond], dim=-1)  # (B, cond_dim)

        # Down path: run the two blocks, stash the pre-downsample activation as a
        # skip connection, then halve the time axis. skips is a LIFO stack.
        skips = []
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, global_feature)
            x = resnet2(x, global_feature)
            skips.append(x)              # save for the mirrored up-level
            x = downsample(x)

        # Bottleneck.
        for mid in self.mid_modules:
            x = mid(x, global_feature)

        # Up path: concat the matching skip (pop, so deepest-first = correct order),
        # run the two blocks, then double the time axis back toward full horizon.
        for resnet1, resnet2, upsample in self.up_modules:
            x = torch.cat([x, skips.pop()], dim=1)
            x = resnet1(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        # Back to (B, horizon, action_dim): the predicted noise for every timestep.
        x = self.final_conv(x)
        return einops.rearrange(x, "b c t -> b t c")


# =============================================================================
# EMA — thin wrapper around diffusers.training_utils.EMAModel so it keeps the
# same interface our train.py already calls (step/copy_to/state_dict/_get_decay).
# =============================================================================
# WHY EMA matters for Diffusion Policy:
# During training, the model weights bounce around (SGD noise). At any given
# step, the raw weights might be slightly off. EMA maintains a slow-moving
# average: θ_ema = decay * θ_ema + (1-decay) * θ_current. Over many steps
# this averages out the noise, giving a smoother, higher-quality model for
# evaluation and deployment. DP relies on this heavily (training weights produce
# noticeably worse rollouts than EMA weights).
#
# Decay warmup: the very first weight update is the initialization, which EMA
# shouldn't trust. We ramp up decay from 0 (fully trust new) toward `max_decay`
# using the formula: decay = 1 - (1 + step/inv_gamma)^(-power). After a few
# hundred steps this saturates near max_decay (0.9999).
class EMAModel:
    """Exponential moving average via diffusers, API-compatible with model.EMAModel.

    diffusers' EMAModel tracks PARAMETERS only (not buffers). That's fine here:
    the only buffers are the fixed normalization stats, which never change during
    training, so the EMA copy and the live model share identical buffers anyway.
    """

    def __init__(self, model: nn.Module, cfg: DiffusionConfig):
        self._ema = _DiffusersEMA(
            model.parameters(),
            decay=cfg.ema_decay,
            min_decay=cfg.ema_min_decay,
            use_ema_warmup=True,        # ramp decay up over early steps
            inv_gamma=cfg.ema_inv_gamma,
            power=cfg.ema_power,
        )
        self.step_count = 0

    def step(self, model: nn.Module):
        """Update the EMA with the current model parameters."""
        self._ema.step(model.parameters())
        self.step_count += 1

    def _get_decay(self) -> float:
        """Decay ramps up from min_decay toward max_decay over training steps."""
        # diffusers sets cur_decay_value after the first .step().
        return float(getattr(self._ema, "cur_decay_value", 0.0) or 0.0)

    def copy_to(self, model: nn.Module):
        """Overwrite the model's params in-place with the EMA (shadow) params. Load EMA weights into a model for evaluation/saving."""
        self._ema.copy_to(model.parameters())

    def state_dict(self) -> dict:
        """Serialize for checkpoint saving."""
        return self._ema.state_dict()

    def load_state_dict(self, state: dict):
        self._ema.load_state_dict(state)


# =============================================================================
# DiffusionPolicy — same interface as model.DiffusionPolicy, but the noise
# process is delegated to diffusers schedulers.
# =============================================================================
# Ties everything together: normalization, vision encoder, U-Net, scheduler,
# and the observation/action queues for receding-horizon inference.
class DiffusionPolicy(nn.Module):
    """Conditional Diffusion Policy using diffusers schedulers for the DDPM/DDIM math."""

    def __init__(self, cfg: DiffusionConfig, stats: dict):
        super().__init__()
        self.cfg = cfg

        # --- Normalization (custom; unchanged) --------------------------------
        image_keys = [f"observation.images.{c}" for c in cfg.cameras]
        depth_keys = [f"observation.depths.{c}" for c in cfg.cameras] if cfg.use_depth else []
        self.normalize_minmax = MinMaxNormalize(stats, ["observation.state", "action"])
        self.normalize_images = MeanStdNormalize(stats, image_keys + depth_keys)
        self.unnormalize_action = MinMaxUnnormalize(stats, "action")

        # --- Vision encoder (custom SpatialSoftmax; unchanged) ----------------
        self.vision_encoder = DiffusionRgbEncoder(cfg)

        # Global conditioning vector = per-step obs features, concatenated across
        # the n_obs_steps observation window. Per step = one feature block per
        # camera (image_feature_dim each) plus the proprioceptive state vector.
        # That flat vector FiLM-conditions every U-Net block.
        per_obs_dim = len(cfg.cameras) * cfg.image_feature_dim + cfg.state_dim
        global_cond_dim = cfg.n_obs_steps * per_obs_dim

        # --- U-Net (custom body, library timestep encoder) --------------------
        self.unet = ConditionalUnet1DLib(cfg, global_cond_dim)

        # --- Noise schedulers (LIBRARY) ---------------------------------------
        # One config drives both. DDPMScheduler handles training add_noise and
        # (optionally) full-length sampling; DDIMScheduler gives fast, deterministic
        # sampling from the SAME betas. This replaces the entire hand-rolled
        # NoiseScheduler (cosine schedule + add_noise + ddpm/ddim steps).
        sched_kwargs = dict(
            num_train_timesteps=cfg.num_train_timesteps,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            beta_schedule=cfg.beta_schedule,       # "squaredcos_cap_v2"
            clip_sample=cfg.clip_sample,
            clip_sample_range=cfg.clip_sample_range,
            prediction_type=cfg.prediction_type,   # "epsilon"
        )
        self.noise_scheduler = DDPMScheduler(**sched_kwargs)
        if cfg.sampler == "ddim":
            self.inference_scheduler = DDIMScheduler(**sched_kwargs)
        else:
            self.inference_scheduler = self.noise_scheduler

        from collections import deque
        self._deque = deque
        self._obs_queue = None
        self._action_queue = None

    # --- observation encoding (identical to model.py) -------------------------
    def _encode_obs(self, batch: dict) -> Tensor:
        """Turn the raw observation window into the flat global conditioning vector."""
        cfg = self.cfg
        B = batch["observation.state"].shape[0]
        n_obs = cfg.n_obs_steps
        state = batch["observation.state"]  # (B, n_obs, S) proprioception

        cam_feats = []
        for cam in cfg.cameras:
            imgs = batch[f"observation.images.{cam}"]  # (B, n_obs, 3, H, W)
            if cfg.use_depth:
                # RGBD: stack the depth channel to make the encoder input 4-channel.
                depths = batch[f"observation.depths.{cam}"][:, :, :1]
                imgs = torch.cat([imgs, depths], dim=2)
            # Fold (B, n_obs) into the batch dim so the CNN sees plain images,
            # then unfold back so features stay aligned per (sample, obs-step).
            imgs_flat = imgs.reshape(B * n_obs, *imgs.shape[2:])
            feat = self.vision_encoder(imgs_flat).reshape(B, n_obs, -1)
            cam_feats.append(feat)

        # Concatenate state + all camera features per step, then flatten the
        # obs-step axis into one vector: (B, n_obs * per_obs_dim).
        per_step = torch.cat([state] + cam_feats, dim=-1)
        return per_step.reshape(B, -1)

    # --- training loss --------------------------------------------------------
    def compute_loss(self, batch: dict) -> tuple[Tensor, dict]:
        cfg = self.cfg
        batch = self.normalize_minmax(batch)
        batch = self.normalize_images(batch)

        global_cond = self._encode_obs(batch)
        trajectory = batch["action"]  # (B, horizon, A), normalized to [-1,1]
        B = trajectory.shape[0]

        # LIBRARY forward diffusion. The training objective (Ho et al. DDPM):
        #   1. draw fresh Gaussian noise the shape of the clean trajectory,
        #   2. draw a random diffusion step t per sample (uniform over [0, T)),
        #   3. add_noise applies x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*noise
        #      in ONE shot (closed form), no iterative loop needed for training.
        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=trajectory.device
        ).long()
        noisy = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # The U-Net sees the corrupted trajectory + its step + the obs, and must
        # predict the noise that was added (epsilon prediction).
        noise_pred = self.unet(noisy, timesteps, global_cond)  # (B, horizon, A)

        # MSE on the noise (epsilon prediction), masked on padded steps.
        if cfg.do_mask_loss_for_padding and "action_is_pad" in batch:
            mask = ~batch["action_is_pad"] # (B, horizon), True = a REAL step
            loss = F.mse_loss(noise_pred, noise, reduction="none") # (B, horizon, A) per-element squared error
            # Expand mask to action dims: (B, horizon, 1).
            loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum() * cfg.action_dim).clamp_min(1) 
        else:
            loss = F.mse_loss(noise_pred, noise)

        return loss, {"mse_loss": loss.item()}

    # --- inference (receding horizon; identical control flow to model.py) -----
    def reset(self):
        self._obs_queue = self._deque(maxlen=self.cfg.n_obs_steps)
        self._action_queue = self._deque()

    @torch.no_grad()
    def select_action(self, obs: dict) -> Tensor:
        """Pick ONE action for the current MuJoCo step (called every frame).

        Receding-horizon strategy:
          1. Accumulate observations into a window (n_obs_steps=2 frames).
          2. When the action queue is empty: run the full denoising loop to get
             a `horizon`-step trajectory, then push `n_action_steps` into the queue.
          3. Pop one action from the queue each call.

        Input `obs` (single environment, B=1):
            observation.state:       (1, S)
            observation.images.cam:  (1, C, H, W)   per camera
        Returns:
            (1, A=8) absolute control target in real control units (un-normalized).
        """
        self.eval()
        cfg = self.cfg
        if self._obs_queue is None:
            self.reset()

        self._obs_queue.append(obs)
        # If we still have buffered actions, just pop one — no denoising needed.
        if len(self._action_queue) > 0:
            return self._action_queue.popleft()

        # --- Stack the observation window (pad by repeating the earliest if needed) ---
        while len(self._obs_queue) < cfg.n_obs_steps:
            self._obs_queue.appendleft(self._obs_queue[0])

        obs_list = list(self._obs_queue)
        obs_batch = {
            "observation.state": torch.stack(
                [o["observation.state"].squeeze(0) for o in obs_list], dim=0
            ).unsqueeze(0)
        }
        for cam in cfg.cameras:
            key = f"observation.images.{cam}"
            obs_batch[key] = torch.stack(
                [o[key].squeeze(0) for o in obs_list], dim=0
            ).unsqueeze(0)
            if cfg.use_depth:
                dkey = f"observation.depths.{cam}"
                if dkey in obs_list[0]:
                    obs_batch[dkey] = torch.stack(
                        [o[dkey].squeeze(0) for o in obs_list], dim=0
                    ).unsqueeze(0)

        obs_batch = self.normalize_minmax(obs_batch)
        obs_batch = self.normalize_images(obs_batch)
        global_cond = self._encode_obs(obs_batch)  # (1, cond)
        device = global_cond.device

        # LIBRARY reverse diffusion. Set the denoising schedule (DDIM uses only
        # num_inference_steps of the T train steps -> big speedup), then iterate
        # from pure noise down to a clean trajectory.
        scheduler = self.inference_scheduler
        n_steps = cfg.num_inference_steps if cfg.sampler == "ddim" else cfg.num_train_timesteps
        scheduler.set_timesteps(n_steps, device=device)

        # x_T ~ N(0, I): start from pure noise the shape of a full trajectory.
        sample = torch.randn((1, cfg.horizon, cfg.action_dim), device=device)
        for t in scheduler.timesteps:                       # high -> low noise
            noise_pred = self.unet(sample, t.reshape(1).to(device), global_cond)
            # scheduler.step does x0-prediction + the DDPM/DDIM posterior internally,
            # returning the slightly-less-noisy sample x_{t-1}.
            sample = scheduler.step(noise_pred, t, sample).prev_sample

        # Map the [-1,1] trajectory back to real action units.
        trajectory = self.unnormalize_action(sample)  # (1, horizon, A)

        # Receding horizon: the trajectory is anchored at the current obs
        # (index n_obs_steps-1). Execute the next n_action_steps actions, queue
        # them, and replan once the queue drains. (Predict long, act short.)
        start = cfg.n_obs_steps - 1
        end = start + cfg.n_action_steps
        for a in trajectory[0, start:end]:
            self._action_queue.append(a.unsqueeze(0))
        return self._action_queue.popleft()
