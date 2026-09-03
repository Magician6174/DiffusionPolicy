"""Diffusion Policy configuration for the Panda pick-and-place task.

Single source of truth for dims, architecture, and training hyperparameters.
Defaults follow the Diffusion Policy paper (Chi et al., 2303.04137) and lerobot's
proven presets, adapted to our 8-DoF / 3-camera setup.

Mental model (how this differs from ACT, our other policy):
  ACT     : CVAE. One observation -> a 100-step action chunk in ONE forward pass;
            temporally ensemble overlapping chunks at test time.
  Diffusion: DDPM. Condition on the last `n_obs_steps` observations, then START
            FROM PURE NOISE and iteratively DENOISE a `horizon`-step action
            trajectory over `num_train_timesteps` reverse steps. Execute only the
            first `n_action_steps` of it, then re-observe and replan (receding
            horizon). The network never outputs actions directly -- it predicts
            the NOISE that was added, and we subtract it, step by step.
"""
from dataclasses import dataclass, field, asdict

import torch


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class DiffusionConfig:
    # --- data -------------------------------------------------------------
    repo_id: str = "panda_pick_place"
    # Local path the dataset is synced to (from S3) before training. Random-access
    # video decode over S3 is far too slow, so we always read from local disk.
    data_root: str = "data/panda_pick_place"
    cameras: tuple = ("front", "diag", "wrist")
    state_dim: int = 8                 # 7 arm joints + gripper finger pos
    action_dim: int = 8                # 7 arm joint targets + gripper ctrl
    image_hw: tuple = (480, 640)     # native recorded resolution (H, W)
    # Downsample images to this (H, W) before the vision backbone. 480x640 is
    # wastefully large for a policy encoder (SpatialSoftmax collapses the whole
    # feature map to 32 keypoints anyway), and full-res is what drives CUDA OOM
    # / slow steps. 240x320 = half res = ~1/4 the ResNet activation memory, with
    # negligible policy-quality cost. Set to None to feed native resolution.
    resize_hw: tuple | None = (240, 320)
    fps: int = 30
    # RGBD toggle. When True, every batch must include `observation.depths.{cam}`
    # (recorder saves depth as a uint8-quantized video; decoded tensor has 3
    # identical channels -- we take the first). The shared vision backbone's
    # conv1 is expanded from 3 to 4 input channels; the new depth channel is
    # initialized to mean(pretrained RGB filters) so edge/blob detectors get a
    # warm start on depth-as-shape-cue instead of starting from noise.
    # Currently False: the on-disk dataset is RGB-only. Flip to True once RGBD
    # demos are recorded (nothing else changes -- the model wires up the 4th
    # channel and the depth normalization buffers automatically).
    use_depth: bool = False

    # --- receding-horizon shapes -----------------------------------------
    # DP conditions on the last `n_obs_steps` observations, predicts a `horizon`
    # of actions, and executes the first `n_action_steps` before replanning.
    #   obs  ->  [ .. horizon actions .. ]  ->  execute n_action_steps  ->  re-obs
    n_obs_steps: int = 2               # observation window (current + 1 past)
    horizon: int = 16                  # action steps predicted per denoise (~0.5s @30fps)
    n_action_steps: int = 8            # steps executed open-loop before replanning
    # Whether to zero out the loss on action steps that ran past the episode end
    # (padded frames). True mirrors ACT; keeps padding from polluting the target.
    do_mask_loss_for_padding: bool = True

    # --- vision encoder ---------------------------------------------------
    # One shared ResNet18 across all cameras and obs steps. Two DP-specific
    # deviations from a stock classifier backbone (both matter, see model.py):
    #   1) BatchNorm -> GroupNorm : BN running stats are unreliable at the small
    #      effective batch DP trains with; GroupNorm is batch-size independent.
    #   2) global average pool -> SpatialSoftmax : collapses the (h,w) feature
    #      grid into `num_keypoints` expected (x,y) locations -- a keypoint-like
    #      spatial representation that preserves WHERE things are (crucial: the
    #      object pose is randomized and lives only in the image, not the state).
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    spatial_softmax_num_keypoints: int = 32
    image_feature_dim: int = 64        # per-camera embedding after the keypoint head
    use_group_norm: bool = True

    # --- 1D temporal U-Net (the denoiser) --------------------------------
    # Operates on the action trajectory as a 1D signal of length `horizon` with
    # `action_dim` channels. FiLM injects the observation+timestep conditioning
    # into every residual block (scale & shift the feature maps).
    down_dims: tuple = (256, 512, 1024)     # channel width at each U-Net level
    kernel_size: int = 5
    n_groups: int = 8                       # GroupNorm groups inside the U-Net
    diffusion_step_embed_dim: int = 128     # size of the sinusoidal timestep code
    use_film_scale_modulation: bool = True  # FiLM does scale+shift (True) vs shift-only

    # --- diffusion process (DDPM) ----------------------------------------
    num_train_timesteps: int = 100          # T: forward-noising / reverse-denoise steps
    beta_schedule: str = "squaredcos_cap_v2"  # cosine schedule (DP default); or "linear"
    beta_start: float = 1e-4                # only used by the linear schedule
    beta_end: float = 0.02                  # only used by the linear schedule
    prediction_type: str = "epsilon"        # network predicts the added noise
    clip_sample: bool = True                # clamp the predicted x0 each reverse step
    clip_sample_range: float = 1.0          # to [-1, 1] -- WHY actions are min-max normed
    # Inference sampler. DDPM = full `num_train_timesteps` reverse steps (slow,
    # exact). DDIM = deterministic, skips steps for a big speedup at rollout.
    sampler: str = "ddim"
    num_inference_steps: int = 10           # DDIM reverse steps (10-16 is plenty)

    # --- training ---------------------------------------------------------
    batch_size: int = 64
    num_workers: int = 4
    # DataLoader prefetch depth per worker. shm working set peaks at
    # ~ num_workers * prefetch_factor * batch_bytes. On a 3.8 GB /dev/shm
    # (SageMaker Studio), 4 workers * 1 * ~0.57 GB (bs=32, resized) ~= 2.3 GB
    # fits; prefetch_factor=2 would double that and can overflow. Keep at 1.
    prefetch_factor: int = 1
    optimizer_lr: float = 1e-4
    optimizer_lr_backbone: float = 1e-4
    optimizer_weight_decay: float = 1e-6
    adam_betas: tuple = (0.95, 0.999)
    adam_eps: float = 1e-8
    n_steps: int = 100_000
    grad_clip_norm: float = 10.0
    val_fraction: float = 0.1
    seed: int = 1000

    # EMA (exponential moving average of weights). DP relies on this heavily:
    # the raw training weights are noisy; a slow EMA copy is what actually gets
    # evaluated and saved. Decay ramps up over training (see EMAModel).
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_inv_gamma: float = 1.0
    ema_power: float = 0.75
    ema_min_decay: float = 0.0

    # mixed precision (safe + fast on CUDA; auto-disabled off-CUDA)
    use_amp: bool = True

    # --- logging / io -----------------------------------------------------
    device: str = field(default_factory=_auto_device)
    output_dir: str = "checkpoints"
    log_freq: int = 100
    val_freq: int = 1000
    save_freq: int = 5000

    def __post_init__(self):
        assert self.horizon >= self.n_obs_steps + self.n_action_steps - 1, (
            "horizon must cover the observation window plus the executed actions."
        )
        # AMP only makes sense on CUDA; MPS/CPU autocast is flaky or pointless.
        if self.device != "cuda":
            self.use_amp = False

    def to_dict(self) -> dict:
        return asdict(self)
