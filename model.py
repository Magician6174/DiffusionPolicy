"""Diffusion Policy model — a conditional DDPM over action trajectories (Chi et al., 2303.04137).

Architecture overview (read this before diving into code):
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  TRAINING                                                                    │
  │    1. Encode observations (images + state) -> global_cond vector.            │
  │    2. Sample noise ε ~ N(0,I) and timestep t ~ Uniform(0,T).                 │
  │    3. Corrupt the ground-truth action trajectory: x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε │
  │    4. U-Net predicts the noise: ε̂ = UNet(x_t, t, global_cond).               │
  │    5. Loss = MSE(ε̂, ε).                                                      │
  │                                                                              │
  │  INFERENCE (rollout)                                                         │
  │    1. Encode observations -> global_cond.                                    │
  │    2. Start from pure noise x_T ~ N(0,I).                                    │
  │    3. For t = T, T-1, ..., 1, 0:                                             │
  │         ε̂ = UNet(x_t, t, global_cond)                                        │
  │         x_{t-1} = scheduler.step(ε̂, t, x_t)   (DDPM or DDIM)                 │
  │    4. x₀ = denoised action trajectory (horizon steps).                       │
  │    5. Execute the first n_action_steps, discard the rest, replan.            │
  └──────────────────────────────────────────────────────────────────────────────┘

Key differences from ACT (why they're worth learning side-by-side):
  - ACT is a one-shot CVAE (1 forward pass -> full chunk). Diffusion is iterative
    (many forward passes through the U-Net to refine noise -> actions).
  - ACT uses transformer encoder/decoder + cross-attention. Diffusion uses a 1D
    convolutional U-Net + FiLM conditioning (scale/shift from the obs embedding).
  - ACT temporal-ensembles overlapping chunks each frame. Diffusion uses receding
    horizon (execute n_action_steps open-loop, then replan from scratch).
  - ACT normalizes actions with mean/std. Diffusion uses min/max to [-1,1] because
    the denoising process is bounded (it clips intermediate x₀ predictions).
"""
import math
from collections import deque
from copy import deepcopy

import einops
from einops.layers.torch import Rearrange
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter

from config import DiffusionConfig


# =============================================================================
# Normalization — owned by the model so it travels with the checkpoint
# =============================================================================
# WHY MIN-MAX for actions/state (not mean/std like ACT)?
# The DDPM denoising process clips intermediate predictions to a fixed range
# ([-1,1] by default). If we used mean/std normalization, the normalized values
# could land anywhere, and the clipping would destroy information. Min-max
# guarantees all data lives in [-1,1], matching the diffusion's operating range.
# Images still use mean/std (they live in [0,1] naturally from pixel decoding).

def _to_tensor(x) -> Tensor:
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


class MinMaxNormalize(nn.Module):
    """x -> (x - min) / (max - min) * 2 - 1, mapping to [-1, 1]."""

    def __init__(self, stats: dict, keys: list[str]):
        super().__init__()
        self._keys = list(keys)
        for key in self._keys:
            safe = key.replace(".", "_")
            mn = _to_tensor(stats[key]["min"])
            mx = _to_tensor(stats[key]["max"])
            self.register_buffer(f"{safe}__min", mn)
            self.register_buffer(f"{safe}__range", (mx - mn).clamp_min(1e-8))

    def _get(self, key):
        safe = key.replace(".", "_")
        return getattr(self, f"{safe}__min"), getattr(self, f"{safe}__range")

    def forward(self, batch: dict) -> dict:
        out = dict(batch)
        for key in self._keys:
            if key not in out:
                continue
            mn, rng = self._get(key)
            out[key] = (out[key] - mn) / rng * 2.0 - 1.0
        return out


class MinMaxUnnormalize(nn.Module):
    """Inverse of MinMaxNormalize: [-1,1] -> original range."""

    def __init__(self, stats: dict, key: str = "action"):
        super().__init__()
        mn = _to_tensor(stats[key]["min"])
        mx = _to_tensor(stats[key]["max"])
        self.register_buffer("min", mn)
        self.register_buffer("range", (mx - mn).clamp_min(1e-8))

    def forward(self, x: Tensor) -> Tensor:
        return (x + 1.0) / 2.0 * self.range + self.min


class MeanStdNormalize(nn.Module):
    """(x - mean) / std for image keys (they live in [0,1] by pixel decoding)."""

    def __init__(self, stats: dict, keys: list[str]):
        super().__init__()
        self._keys = list(keys)
        for key in self._keys:
            safe = key.replace(".", "_")
            self.register_buffer(f"{safe}__mean", _to_tensor(stats[key]["mean"]))
            self.register_buffer(f"{safe}__std", _to_tensor(stats[key]["std"]).clamp_min(1e-8))

    def _get(self, key):
        safe = key.replace(".", "_")
        return getattr(self, f"{safe}__mean"), getattr(self, f"{safe}__std")

    def forward(self, batch: dict) -> dict:
        out = dict(batch)
        for key in self._keys:
            if key not in out:
                continue
            mean, std = self._get(key)
            out[key] = (out[key] - mean) / std
        return out


# =============================================================================
# Sinusoidal positional embedding for the diffusion timestep
# =============================================================================
# The U-Net needs to know WHAT noise level it's being asked to denoise. We
# encode the integer timestep t as a fixed sinusoidal embedding (same math as
# the Transformer pos embed), then learn an MLP on top. This is standard in
# DDPM (Ho et al. 2020) and inherited by Diffusion Policy.

class SinusoidalPosEmb(nn.Module):
    """Map a scalar timestep to a rich D-dimensional sinusoidal embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """x: (B,) integer or float timesteps. Returns: (B, dim)."""
        device = x.device
        half_dim = self.dim // 2
        # log-spaced frequencies: from 1 (slow oscillation) to 1/10000 (fast).
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # (D/2,)
        emb = x[:, None].float() * emb[None, :]  # (B, D/2)  outer product
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, D) interleaved sin/cos
        return emb


# =============================================================================
# Spatial Softmax — the DP-specific vision pooling layer
# =============================================================================
# WHY SpatialSoftmax instead of global average pooling?
# Average pooling says "what features are present" (location-invariant).
# SpatialSoftmax says "WHERE are the features" — for each learned keypoint
# filter, it returns the expected (x,y) position in the image where that
# feature is most active. This preserves SPATIAL information about object poses,
# which is exactly what the policy needs when the object location varies.
#
# How it works:
#   1. The feature map (B, C, H, W) has C channels. Each channel is a
#      spatial "heat map" of a learned filter's activation.
#   2. Apply a softmax across the spatial (H*W) dimension of each channel,
#      turning activations into a proper probability distribution over positions.
#   3. Compute the expected (x, y) under that distribution: a 2D "keypoint".
#   4. Output: (B, C, 2) -> flatten -> (B, C*2). We have C expected-position
#      pairs, one per filter — a compact, spatially-grounded representation.
# We ADD a 1x1 conv to reduce C from the backbone's output channels (512) to
# `num_keypoints` before the softmax, so the output is (B, num_keypoints*2).

class SpatialSoftmax(nn.Module):
    """Expected spatial coordinates under a softmax over each channel."""

    def __init__(self, h: int, w: int, num_keypoints: int, in_channels: int):
        super().__init__()
        self.h, self.w = h, w
        self.num_keypoints = num_keypoints
        # Reduce channels from backbone (512) to num_keypoints via 1x1 conv.
        self.conv = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)
        # Pre-compute a meshgrid of normalized (x, y) coordinates in [-1, 1].
        # Every spatial cell has a fixed position; the softmax just picks among them.
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, w),
            torch.linspace(-1.0, 1.0, h),
            indexing="xy",
        )
        # (H*W,) flattened coordinate vectors — registered as buffers so they
        # move to the right device automatically.
        self.register_buffer("pos_x", pos_x.reshape(-1))  # (H*W,)
        self.register_buffer("pos_y", pos_y.reshape(-1))  # (H*W,)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C_in, H, W). Returns: (B, num_keypoints * 2)."""
        x = self.conv(x)  # (B, num_kp, H, W)
        B, K = x.shape[0], x.shape[1]
        # Flatten spatial dims and softmax -> each channel is a distribution over H*W cells.
        flat = x.reshape(B, K, -1)                   # (B, K, H*W)
        attention = F.softmax(flat, dim=-1)          # (B, K, H*W) -- sums to 1 per channel
        # Expected (x, y) = sum(attention_i * pos_i) over all spatial cells.
        expected_x = (attention * self.pos_x).sum(dim=-1)  # (B, K)
        expected_y = (attention * self.pos_y).sum(dim=-1)  # (B, K)
        return torch.cat([expected_x, expected_y], dim=-1)  # (B, K*2)


# =============================================================================
# Vision Encoder (ResNet18 + GroupNorm + SpatialSoftmax + linear projection)
# =============================================================================
# Shared across all cameras and observation steps (weight-tied). The input
# arrives as (B*n_obs*n_cams, C, H, W) and comes out as (B*n_obs*n_cams, feat_dim).

def _replace_bn_with_gn(module: nn.Module, num_groups: int = 16) -> nn.Module:
    """Recursively replace BatchNorm2d with GroupNorm throughout a model.

    WHY: BatchNorm statistics depend on batch composition — at tiny batch sizes
    (or per-camera/obs-step splitting), running stats are inaccurate and training
    is unstable. GroupNorm normalizes within groups of channels per-sample, so it
    is completely independent of other samples in the batch. This is why DP (and
    most diffusion-based methods) prefer GN over BN for the vision encoder.EMA 
    maintains a separate averaged copy of the model's weights. Those slowly-averaged 
    weights correspond to a different function than any single training checkpoint.
    The BN running stats in the EMA copy are therefore stale and mismatched relative to 
    the averaged weights — the normalization statistics were collected for weights that 
    no longer exist. You get an EMA model whose activations are normalized by the wrong 
    mean/variance.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_ch = child.num_features
            gn = nn.GroupNorm(min(num_groups, num_ch), num_ch)
            setattr(module, name, gn)
        else:
            _replace_bn_with_gn(child, num_groups)
    return module


class DiffusionRgbEncoder(nn.Module):
    """Per-camera image encoder: ResNet18(GroupNorm) -> SpatialSoftmax -> Linear."""

    def __init__(self, cfg: DiffusionConfig):
        super().__init__()
        # Load pretrained ResNet18 as the backbone.
        backbone = getattr(torchvision.models, cfg.vision_backbone)(
            weights=cfg.pretrained_backbone_weights,
        )
        if cfg.use_group_norm:
            _replace_bn_with_gn(backbone)
        # RGBD: expand conv1 from 3->4 input channels (same trick as ACT).
        if cfg.use_depth:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                4, old_conv.out_channels, old_conv.kernel_size,
                stride=old_conv.stride, padding=old_conv.padding, bias=(old_conv.bias is not None),
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:] = old_conv.weight.mean(dim=1, keepdim=True)
                if old_conv.bias is not None:
                    new_conv.bias.copy_(old_conv.bias)
            backbone.conv1 = new_conv

        # Extract up to layer4 (output: 512 channels, spatial h/32 x w/32).
        self.backbone = IntermediateLayerGetter(backbone, return_layers={"layer4": "feature_map"})
        backbone_out_ch = 512  # ResNet18 layer4 output channels

        # Resolution actually fed to the backbone: resize_hw if set, else native.
        # SpatialSoftmax needs the post-backbone spatial size, so it must match
        # the resolution forward() feeds in.
        self._resize_hw = cfg.resize_hw
        eff_hw = cfg.resize_hw if cfg.resize_hw is not None else cfg.image_hw
        # Probe the backbone for the real feature-map size instead of eff//32:
        # ResNet rounds UP at each pooling stage, so odd sizes don't divide evenly
        # (240 -> 8, not 7). A one-off dummy forward gives the exact (h, w).
        in_ch = 4 if cfg.use_depth else 3
        with torch.no_grad():
            _probe = torch.zeros(1, in_ch, eff_hw[0], eff_hw[1])
            feat_h, feat_w = self.backbone(_probe)["feature_map"].shape[-2:]

        # SpatialSoftmax: collapse (h,w) into expected keypoint positions.
        self.spatial_softmax = SpatialSoftmax(
            h=feat_h, w=feat_w,
            num_keypoints=cfg.spatial_softmax_num_keypoints,
            in_channels=backbone_out_ch,
        )
        # Final projection: keypoints -> compact embedding.
        kp_dim = cfg.spatial_softmax_num_keypoints * 2  # (x,y) per keypoint
        self.fc = nn.Sequential(
            nn.Linear(kp_dim, cfg.image_feature_dim),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (N, C, H, W) where N can be B*n_obs*n_cams. Returns (N, feat_dim)."""
        # Downsample before the backbone (the big memory/compute lever). Bilinear
        # is fine here -- we only need coarse spatial structure for keypoints.
        if self._resize_hw is not None and x.shape[-2:] != tuple(self._resize_hw):
            x = F.interpolate(x, size=self._resize_hw, mode="bilinear", align_corners=False)
        feat = self.backbone(x)["feature_map"]  # (N, 512, h, w)
        kp = self.spatial_softmax(feat)          # (N, num_kp*2)
        return self.fc(kp)                       # (N, image_feature_dim)


# =============================================================================
# 1D Temporal U-Net building blocks
# =============================================================================
# The U-Net operates on the action trajectory as a 1D signal:
#   - dimension 1 = time (horizon steps, length T=16)
#   - channels   = action_dim (8) at input, widened by convolutions inside
# This is NOT an image U-Net (2D). It's a sequence U-Net (1D) — Conv1d throughout.
# FiLM (Feature-wise Linear Modulation) injects the conditioning at every block:
#   output = scale * features + bias, where (scale, bias) come from a linear
#   projection of the conditioning vector. This tells each block "what obs/timestep
#   context you're working with" without cross-attention overhead.

class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish (the standard DP building block)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        # padding = kernel_size//2 keeps the temporal length unchanged (same-padding).
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, C_in, T). Returns: (B, C_out, T)."""
        return self.block(x)


class Downsample1d(nn.Module):
    """Halve temporal resolution via stride-2 convolution."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """Double temporal resolution via transposed convolution."""

    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    """The workhorse: a residual Conv1d block conditioned via FiLM.

    Structure:
        x ─┬─ Conv1dBlock ─── FiLM(cond) ─── Conv1dBlock ─── + ─── out
           │                                                   │
           └───────────── residual (1x1 conv if dims differ) ──┘

    FiLM (Feature-wise Linear Modulation):
        A linear layer maps the conditioning vector (obs + timestep embedding)
        to per-channel scale and bias: out = scale * features + bias.
        This is how the observation context influences EVERY layer of the U-Net
        without expensive attention — just a cheap affine transform per channel.
    """

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int,
                 kernel_size: int = 5, n_groups: int = 8,
                 use_film_scale_modulation: bool = True):
        super().__init__()
        self.use_film_scale = use_film_scale_modulation

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
        ])

        # FiLM conditioning: project cond vector -> per-channel scale+bias (or bias only).
        # If scale modulation: output 2*out_channels (first half = scale, second = bias).
        # If bias only: output out_channels (just bias, added to features).
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            # Unsqueeze the channel dim so it broadcasts over the time axis.
            Rearrange("batch channels -> batch channels 1"),
        )

        # Residual connection: 1x1 conv to match channel dims if they differ.
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        x:    (B, C_in, T)  feature map along the time axis.
        cond: (B, cond_dim) global conditioning vector (obs embedding + timestep).
        Returns: (B, C_out, T).
        """
        out = self.blocks[0](x)  # (B, C_out, T) — first conv+norm+mish

        # FiLM injection between the two conv blocks.
        embed = self.cond_encoder(cond)  # (B, C_out*2, 1) or (B, C_out, 1)
        if self.use_film_scale:
            # Split into scale and bias halves.
            scale, bias = embed.chunk(2, dim=1)  # each (B, C_out, 1)
            out = scale * out + bias             # affine modulation per channel
        else:
            out = out + embed                    # shift only

        out = self.blocks[1](out)  # (B, C_out, T) — second conv+norm+mish
        out = out + self.residual_conv(x)  # residual connection
        return out


# =============================================================================
# Conditional 1D U-Net — the denoiser
# =============================================================================
# U-Net architecture (why a U-Net for diffusion?):
#   The encoder path downsamples the noisy action trajectory, capturing long-range
#   temporal structure. The decoder path upsamples it back, guided by skip
#   connections from matching encoder levels (so fine-grained details aren't lost).
#   At every level, FiLM injects the "what I'm looking at + how noisy this is"
#   conditioning. The bottleneck (mid) captures the global trajectory shape.
#
# Data flow for our default config (action_dim=8, horizon=16, down_dims=(256,512,1024)):
#   Input (B, 8, 16) = noisy action trajectory
#   Down level 0: (B, 8, 16)   -> (B, 256, 16) -> downsample -> (B, 256, 8)
#   Down level 1: (B, 256, 8)  -> (B, 512, 8)  -> downsample -> (B, 512, 4)
#   Down level 2: (B, 512, 4)  -> (B, 1024, 4) -> (no downsample, last level)
#   Mid:          (B, 1024, 4) -> (B, 1024, 4)
#   Up level 0:   cat skip -> (B, 2048, 4) -> (B, 512, 4) -> upsample -> (B, 512, 8)
#   Up level 1:   cat skip -> (B, 1024, 8) -> (B, 256, 8) -> upsample -> (B, 256, 16)
#   Final conv:   (B, 256, 16) -> (B, 8, 16) = predicted noise ε̂

class ConditionalUnet1D(nn.Module):
    """1D temporal U-Net with FiLM conditioning for diffusion denoising."""

    def __init__(self, cfg: DiffusionConfig, global_cond_dim: int):
        super().__init__()
        input_dim = cfg.action_dim
        dsed = cfg.diffusion_step_embed_dim

        # --- Diffusion timestep encoder: integer t -> rich embedding -----------
        # SinusoidalPosEmb gives a fixed encoding; the MLP on top learns how to
        # use it. dsed=128 -> expand to 512 -> back to 128. The Mish activation
        # (smooth ReLU variant) is DP convention.
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),           # (B,) -> (B, 128)
            nn.Linear(dsed, dsed * 4),        # (B, 128) -> (B, 512)
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),        # (B, 512) -> (B, 128)
        )

        # Total conditioning dimension: timestep embed + global obs embedding.
        cond_dim = dsed + global_cond_dim

        # --- Build encoder (down) path -----------------------------------------
        all_dims = [input_dim] + list(cfg.down_dims)  # [8, 256, 512, 1024]
        in_out = list(zip(all_dims[:-1], all_dims[1:]))  # [(8,256), (256,512), (512,1024)]
        mid_dim = all_dims[-1]  # 1024

        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim,
                                          cfg.kernel_size, cfg.n_groups, cfg.use_film_scale_modulation),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim,
                                          cfg.kernel_size, cfg.n_groups, cfg.use_film_scale_modulation),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        # --- Bottleneck (mid) ---------------------------------------------------
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim,
                                      cfg.kernel_size, cfg.n_groups, cfg.use_film_scale_modulation),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim,
                                      cfg.kernel_size, cfg.n_groups, cfg.use_film_scale_modulation),
        ])

        # --- Build decoder (up) path -------------------------------------------
        # Skip connections double the channel dim at input (concat from encoder).
        # We reverse the encoder levels (excluding the first, which is the input dim).
        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                # dim_out*2 because we concat the skip connection (same channels as encoder output).
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim,
                                          cfg.kernel_size, cfg.n_groups, cfg.use_film_scale_modulation),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim,
                                          cfg.kernel_size, cfg.n_groups, cfg.use_film_scale_modulation),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        # --- Final projection back to action_dim --------------------------------
        self.final_conv = nn.Sequential(
            Conv1dBlock(cfg.down_dims[0], cfg.down_dims[0], cfg.kernel_size, cfg.n_groups),
            nn.Conv1d(cfg.down_dims[0], input_dim, kernel_size=1),
        )

    def forward(self, sample: Tensor, timestep: Tensor, global_cond: Tensor) -> Tensor:
        """Predict the noise ε̂ in the noisy action trajectory.

        Args:
            sample:      (B, horizon, action_dim) noisy trajectory x_t.
            timestep:    (B,) diffusion timestep indices (integer).
            global_cond: (B, global_cond_dim) observation embedding.
        Returns:
            (B, horizon, action_dim) predicted noise ε̂.
        """
        # Conv1d operates on (B, C, T), so transpose time/channels.
        x = einops.rearrange(sample, "b t c -> b c t")  # (B, action_dim, horizon)

        # Encode the diffusion timestep and concatenate with obs conditioning.
        t_embed = self.diffusion_step_encoder(timestep)  # (B, dsed)
        global_feature = torch.cat([t_embed, global_cond], dim=-1)  # (B, cond_dim)

        # --- Encoder (down) path with skip connections -------------------------
        encoder_skips = []
        for resnet1, resnet2, downsample in self.down_modules:
            x = resnet1(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skips.append(x)  # save BEFORE downsampling for skip connection
            x = downsample(x)

        # --- Bottleneck ---------------------------------------------------------
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # --- Decoder (up) path with skip connections ----------------------------
        for resnet1, resnet2, upsample in self.up_modules:
            skip = encoder_skips.pop()  # LIFO: deepest skip first
            x = torch.cat([x, skip], dim=1)  # channel-wise concat (doubles channels)
            x = resnet1(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        # --- Project back to action_dim and transpose back ----------------------
        x = self.final_conv(x)  # (B, action_dim, horizon)
        return einops.rearrange(x, "b c t -> b t c")  # (B, horizon, action_dim)


# =============================================================================
# Noise Scheduler — the heart of diffusion, implemented from scratch
# =============================================================================
# DDPM = Denoising Diffusion Probabilistic Models (Ho et al., 2020).
# The idea: define a FORWARD process that gradually adds Gaussian noise to data
# over T steps until it becomes pure noise, then train a network to REVERSE
# this process step by step. The math below is derived from the DDPM paper.
#
# Forward process (used in training — just a formula, no network):
#   q(x_t | x_0) = N(x_t ; √ᾱ_t · x_0, (1-ᾱ_t) · I)
#   Which means: x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,  ε ~ N(0,I)
#   where ᾱ_t = ∏_{s=1}^{t} α_s,  α_s = 1 - β_s.
#
# Reverse process (inference — network-guided):
#   DDPM: x_{t-1} = μ(x_t, t, ε̂) + σ_t · z, z ~ N(0,I)   (stochastic)
#   DDIM: x_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1-ᾱ_{t-1}) · ε̂  (deterministic, eta=0)
#
# Beta schedule: controls how quickly noise is added. Cosine schedule
# (squaredcos_cap_v2) adds noise more uniformly than linear, avoiding
# the "too easy at start, too hard at end" problem of linear schedules.

class NoiseScheduler:
    """Handles all diffusion noise operations: schedules, forward noising, reverse steps.

    This is NOT an nn.Module — it holds precomputed constants as plain tensors
    (moved to device as needed). The policy wrapper ensures device consistency.
    """

    def __init__(self, cfg: DiffusionConfig):
        self.num_train_timesteps = cfg.num_train_timesteps
        self.prediction_type = cfg.prediction_type
        self.clip_sample = cfg.clip_sample
        self.clip_sample_range = cfg.clip_sample_range

        # --- Compute betas (noise schedule) -----------------------------------
        if cfg.beta_schedule == "squaredcos_cap_v2":
            betas = self._cosine_beta_schedule(cfg.num_train_timesteps)
        elif cfg.beta_schedule == "linear":
            betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {cfg.beta_schedule}")

        # --- Precompute all derived constants ---------------------------------
        # α_t = 1 - β_t (how much signal is retained at step t)
        alphas = 1.0 - betas
        # ᾱ_t = cumulative product of α's = total signal fraction remaining at t.
        # At t=0, ᾱ≈1 (almost clean). At t=T-1, ᾱ≈0 (almost pure noise).
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        # Precompute common factors to avoid repeated sqrt in hot loops.
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        # For the DDPM posterior mean computation.
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # Inference timestep subsequence (set by set_timesteps before sampling).
        self._timesteps = None

    @staticmethod
    def _cosine_beta_schedule(T: int, s: float = 0.008, max_beta: float = 0.999) -> Tensor:
        """Cosine schedule (Nichol & Dhariwal, 2021). The idea: define ᾱ_t as a
        cosine curve from 1 to 0 over T steps, then derive betas from consecutive
        ratios. This makes the noise injection more uniform across steps compared
        to linear (which dumps most noise at the end).

        ᾱ(t) = cos²( (t/T + s) / (1+s) · π/2 )
        β_t = 1 - ᾱ_t / ᾱ_{t-1}
        """
        def alpha_bar(t):
            return math.cos((t + s) / (1 + s) * math.pi / 2) ** 2

        betas = []
        for i in range(T):
            t1 = i / T
            t2 = (i + 1) / T
            betas.append(min(1.0 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return torch.tensor(betas, dtype=torch.float32)

    def set_timesteps(self, num_inference_steps: int, sampler: str = "ddpm"):
        """Set the timestep subsequence for inference.

        DDPM uses all T timesteps. DDIM can skip steps (e.g. 10 steps for T=100),
        producing an evenly-spaced subsequence. Both traverse from T-1 down to 0.
        """
        if sampler == "ddpm":
            # Full reverse: T-1, T-2, ..., 0
            self._timesteps = list(range(self.num_train_timesteps - 1, -1, -1))
        elif sampler == "ddim":
            # Evenly-spaced subset: e.g. T=100, steps=10 -> [99,88,77,...,0]
            step_ratio = self.num_train_timesteps // num_inference_steps
            self._timesteps = list(
                (np.arange(0, num_inference_steps) * step_ratio).round().astype(np.int64)
            )[::-1]  # reverse so we go from noisy to clean
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

    @property
    def timesteps(self) -> list[int]:
        assert self._timesteps is not None, "Call set_timesteps() first."
        return self._timesteps

    def add_noise(self, original: Tensor, noise: Tensor, timesteps: Tensor) -> Tensor:
        """Forward diffusion: q(x_t | x_0) = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε.

        Args:
            original:  (B, T, D) clean action trajectories x_0.
            noise:     (B, T, D) standard Gaussian noise ε.
            timesteps: (B,) integer timestep indices.
        Returns:
            (B, T, D) noisy samples x_t.
        """
        device = original.device
        # Gather the ᾱ_t value for each sample in the batch. Shape: (B, 1, 1)
        # for broadcasting over (T, D).
        sqrt_acp = self.sqrt_alphas_cumprod.to(device)[timesteps][:, None, None]
        sqrt_one_minus_acp = self.sqrt_one_minus_alphas_cumprod.to(device)[timesteps][:, None, None]
        return sqrt_acp * original + sqrt_one_minus_acp * noise

    def _predict_x0(self, model_output: Tensor, timestep: int, sample: Tensor) -> Tensor:
        """Recover the predicted clean sample x̂_0 from the noise prediction.

        From the forward process equation:
            x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
        Rearranging (and substituting the model's ε̂ for the true ε):
            x̂_0 = (x_t - √(1-ᾱ_t) · ε̂) / √ᾱ_t

        We clip x̂_0 to [-1,1] because the true actions live there (min-max normed).
        Without clipping, the predicted x_0 can drift, leading to instability in
        the reverse chain. This "dynamic thresholding" was proposed by Imagen.
        """
        device = sample.device
        sqrt_acp = self.sqrt_alphas_cumprod.to(device)[timestep]
        sqrt_one_minus_acp = self.sqrt_one_minus_alphas_cumprod.to(device)[timestep]
        x0_pred = (sample - sqrt_one_minus_acp * model_output) / sqrt_acp
        if self.clip_sample:
            x0_pred = x0_pred.clamp(-self.clip_sample_range, self.clip_sample_range)
        return x0_pred

    def ddpm_step(self, model_output: Tensor, timestep: int, sample: Tensor) -> Tensor:
        """One DDPM reverse step: x_t -> x_{t-1} (stochastic).

        The posterior q(x_{t-1} | x_t, x_0) is Gaussian with:
            mean = (√ᾱ_{t-1} · β_t)/(1-ᾱ_t) · x̂_0  +  (√α_t · (1-ᾱ_{t-1}))/(1-ᾱ_t) · x_t
            variance = β_t · (1-ᾱ_{t-1}) / (1-ᾱ_t)

        We sample x_{t-1} = mean + √variance · z,  z ~ N(0,I).
        At t=0 no noise is added (we're done denoising).
        """
        device = sample.device
        x0_pred = self._predict_x0(model_output, timestep, sample)

        # Coefficients for the posterior mean.
        alpha_t = self.alphas.to(device)[timestep]
        alpha_cumprod_t = self.alphas_cumprod.to(device)[timestep]
        alpha_cumprod_prev = self.alphas_cumprod.to(device)[timestep - 1] if timestep > 0 else torch.tensor(1.0, device=device)
        beta_t = self.betas.to(device)[timestep]

        # Posterior mean (two-term formula).
        coeff_x0 = (alpha_cumprod_prev.sqrt() * beta_t) / (1.0 - alpha_cumprod_t)
        coeff_xt = (alpha_t.sqrt() * (1.0 - alpha_cumprod_prev)) / (1.0 - alpha_cumprod_t)
        mean = coeff_x0 * x0_pred + coeff_xt * sample

        # Posterior variance. No noise at t=0 (final step).
        if timestep > 0:
            variance = beta_t * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)
            noise = torch.randn_like(sample)
            return mean + variance.sqrt() * noise
        return mean

    def ddim_step(self, model_output: Tensor, timestep: int, prev_timestep: int, sample: Tensor) -> Tensor:
        """One DDIM reverse step: x_t -> x_{t-prev} (deterministic, eta=0).

        DDIM (Song et al., 2020) defines a NON-MARKOVIAN reverse process that
        shares the same marginals as DDPM but allows skipping steps. With eta=0
        the process is fully deterministic — same noise input -> same output.
        This is why DDIM is faster (10 steps vs 100) with minimal quality loss.

        Formula (eta=0 case):
            x_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1-ᾱ_{t-1}) · ε̂
        where x̂_0 = predicted clean sample (same as above).
        """
        device = sample.device
        x0_pred = self._predict_x0(model_output, timestep, sample)

        alpha_cumprod_prev = (
            self.alphas_cumprod.to(device)[prev_timestep] if prev_timestep >= 0
            else torch.tensor(1.0, device=device)
        )
        # "Predicted direction pointing to x_t" (the noise residual heading).
        pred_direction = (1.0 - alpha_cumprod_prev).sqrt() * model_output
        # Compose the denoised estimate.
        prev_sample = alpha_cumprod_prev.sqrt() * x0_pred + pred_direction
        return prev_sample

    def step(self, model_output: Tensor, timestep: int, sample: Tensor,
             prev_timestep: int | None = None, sampler: str = "ddpm") -> Tensor:
        """Dispatch to the correct reverse step method."""
        if sampler == "ddpm":
            return self.ddpm_step(model_output, timestep, sample)
        elif sampler == "ddim":
            assert prev_timestep is not None
            return self.ddim_step(model_output, timestep, prev_timestep, sample)
        raise ValueError(f"Unknown sampler: {sampler}")


# =============================================================================
# EMA (Exponential Moving Average) — smoothed weights for stable inference
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
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, cfg: DiffusionConfig):
        self.averaged = deepcopy(dict(model.named_parameters()))
        # Also track buffers (normalization stats, etc.)
        self.averaged_buffers = deepcopy(dict(model.named_buffers()))
        self.max_decay = cfg.ema_decay
        self.inv_gamma = cfg.ema_inv_gamma
        self.power = cfg.ema_power
        self.min_decay = cfg.ema_min_decay
        self.step_count = 0

    def _get_decay(self) -> float:
        """Decay ramps up from min_decay toward max_decay over training steps."""
        step = max(0, self.step_count - 1)
        value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        return max(self.min_decay, min(value, self.max_decay))

    @torch.no_grad()
    def step(self, model: nn.Module):
        """Update the EMA with the current model parameters."""
        decay = self._get_decay()
        self.step_count += 1
        for name, param in model.named_parameters():
            if param.dtype.is_floating_point:
                self.averaged[name].data.mul_(decay).add_(param.data, alpha=1.0 - decay)
        for name, buf in model.named_buffers():
            if buf.dtype.is_floating_point:
                self.averaged_buffers[name].data.copy_(buf.data)
            else:
                self.averaged_buffers[name].data.copy_(buf.data)

    def copy_to(self, model: nn.Module):
        """Load EMA weights into a model for evaluation/saving."""
        state = {}
        for name, param in self.averaged.items():
            state[name] = param.data
        for name, buf in self.averaged_buffers.items():
            state[name] = buf.data
        model.load_state_dict(state, strict=False)

    def state_dict(self) -> dict:
        """Serialize for checkpoint saving."""
        return {
            "averaged": {k: v.data for k, v in self.averaged.items()},
            "averaged_buffers": {k: v.data for k, v in self.averaged_buffers.items()},
            "step_count": self.step_count,
        }

    def load_state_dict(self, state: dict):
        for k, v in state["averaged"].items():
            self.averaged[k].data.copy_(v)
        for k, v in state["averaged_buffers"].items():
            self.averaged_buffers[k].data.copy_(v)
        self.step_count = state["step_count"]


# =============================================================================
# DiffusionPolicy — the full policy wrapper
# =============================================================================
# Ties everything together: normalization, vision encoder, U-Net, scheduler,
# and the observation/action queues for receding-horizon inference.

class DiffusionPolicy(nn.Module):
    """Conditional Diffusion Policy for visuomotor control.

    Training: compute_loss(batch) -> MSE on noise prediction.
    Inference: select_action(obs) -> one action per MuJoCo step (internally
               maintains obs history and an action queue for receding horizon).
    """

    def __init__(self, cfg: DiffusionConfig, stats: dict):
        super().__init__()
        self.cfg = cfg

        # --- Normalization (travel with checkpoint) ---------------------------
        # Actions and state: min-max to [-1,1] (required by the diffusion clip).
        # Images: mean/std (they're in [0,1] from pixel decoding; std-norm is standard).
        image_keys = [f"observation.images.{c}" for c in cfg.cameras]
        depth_keys = [f"observation.depths.{c}" for c in cfg.cameras] if cfg.use_depth else []
        self.normalize_minmax = MinMaxNormalize(stats, ["observation.state", "action"])
        self.normalize_images = MeanStdNormalize(stats, image_keys + depth_keys)
        self.unnormalize_action = MinMaxUnnormalize(stats, "action")

        # --- Vision encoder (shared across cameras and obs steps) -------------
        self.vision_encoder = DiffusionRgbEncoder(cfg)

        # --- Global conditioning dimension ------------------------------------
        # obs_cond = concat of all per-obs-step features, flattened over steps.
        # Per step: n_cameras * image_feature_dim + state_dim.
        per_obs_dim = len(cfg.cameras) * cfg.image_feature_dim + cfg.state_dim
        global_cond_dim = cfg.n_obs_steps * per_obs_dim

        # --- U-Net (the denoiser) ---------------------------------------------
        self.unet = ConditionalUnet1D(cfg, global_cond_dim)

        # --- Noise scheduler --------------------------------------------------
        self.noise_scheduler = NoiseScheduler(cfg)

        # --- Inference state (receding horizon) --------------------------------
        self._obs_queue: deque | None = None
        self._action_queue: deque | None = None

    def _encode_obs(self, batch: dict) -> Tensor:
        """Encode observations into a flat global conditioning vector.

        Input shapes (training):
            observation.state:       (B, n_obs, S=8)
            observation.images.cam:  (B, n_obs, C, H, W)   per camera
        Returns:
            (B, global_cond_dim) flat vector.
        """
        cfg = self.cfg
        B = batch["observation.state"].shape[0]
        n_obs = cfg.n_obs_steps
        n_cams = len(cfg.cameras)

        # State: already (B, n_obs, S) and min-max normalized.
        state = batch["observation.state"]  # (B, n_obs, S)

        # Images: reshape all cameras and obs steps into one batch for the encoder.
        cam_feats = []
        for cam in cfg.cameras:
            imgs = batch[f"observation.images.{cam}"]  # (B, n_obs, 3, H, W)
            if cfg.use_depth:
                depths = batch[f"observation.depths.{cam}"][:, :, :1]  # (B, n_obs, 1, H, W)
                imgs = torch.cat([imgs, depths], dim=2)  # (B, n_obs, 4, H, W)
            # Merge batch and obs dims for the encoder: (B*n_obs, C, H, W).
            imgs_flat = imgs.reshape(B * n_obs, *imgs.shape[2:])
            feat = self.vision_encoder(imgs_flat)  # (B*n_obs, image_feature_dim)
            feat = feat.reshape(B, n_obs, -1)  # (B, n_obs, image_feature_dim)
            cam_feats.append(feat)

        # Concat per obs step: [state, cam1_feat, cam2_feat, cam3_feat].
        # (B, n_obs, state_dim + n_cams*image_feat_dim)
        per_step = torch.cat([state] + cam_feats, dim=-1)
        # Flatten obs steps into one big vector: (B, n_obs * per_obs_dim).
        return per_step.reshape(B, -1)

    # --- Training loss --------------------------------------------------------
    def compute_loss(self, batch: dict) -> tuple[Tensor, dict]:
        """One training step: sample noise, corrupt actions, predict noise, MSE.

        Input `batch` (raw, un-normalized) holds:
            observation.state:       (B, n_obs, 8)
            observation.images.cam:  (B, n_obs, 3, 480, 640)  x3
            action:                  (B, horizon, 8)  ground-truth action trajectory
            action_is_pad:           (B, horizon)     True on padding steps
        Returns:
            loss: scalar MSE tensor for backprop.
            info: dict {mse_loss: float} for logging.
        """
        cfg = self.cfg
        # Normalize everything.
        batch = self.normalize_minmax(batch)
        batch = self.normalize_images(batch)

        # Encode observations -> conditioning.
        global_cond = self._encode_obs(batch)  # (B, global_cond_dim)

        # Ground-truth trajectory (normalized to [-1,1]).
        trajectory = batch["action"]  # (B, horizon, action_dim)
        B = trajectory.shape[0]

        # --- DDPM forward: add noise to trajectory ---
        # Sample random noise (same shape as trajectory).
        noise = torch.randn_like(trajectory)
        # Sample random timesteps uniformly from [0, T).
        timesteps = torch.randint(0, cfg.num_train_timesteps, (B,), device=trajectory.device)
        # Corrupt: x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)

        # --- Predict the noise via U-Net ---
        noise_pred = self.unet(noisy_trajectory, timesteps, global_cond)  # (B, horizon, A)

        # --- MSE loss (optionally masked on padding) ---
        if cfg.do_mask_loss_for_padding and "action_is_pad" in batch:
            mask = ~batch["action_is_pad"]  # (B, horizon) True = real step
            # Expand mask to action dims: (B, horizon, 1).
            loss = F.mse_loss(noise_pred, noise, reduction="none")  # (B, horizon, A)
            loss = (loss * mask.unsqueeze(-1)).sum() / (mask.sum() * cfg.action_dim).clamp_min(1)
        else:
            loss = F.mse_loss(noise_pred, noise)

        return loss, {"mse_loss": loss.item()}

    # --- Inference (receding-horizon rollout) ---------------------------------
    def reset(self):
        """Call at the start of each episode. Clears obs history and action queue."""
        self._obs_queue = deque(maxlen=self.cfg.n_obs_steps)
        self._action_queue = deque()

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

        # --- Accumulate observations into the window ---
        self._obs_queue.append(obs)

        # If we still have buffered actions, just pop one — no denoising needed.
        if len(self._action_queue) > 0:
            return self._action_queue.popleft()

        # --- Stack the observation window (pad by repeating the earliest if needed) ---
        while len(self._obs_queue) < cfg.n_obs_steps:
            self._obs_queue.appendleft(self._obs_queue[0])

        # Build a batched obs dict: (1, n_obs, ...) for each key.
        obs_batch = {}
        obs_list = list(self._obs_queue)
        # State: stack (n_obs, S) -> unsqueeze batch -> (1, n_obs, S).
        obs_batch["observation.state"] = torch.stack(
            [o["observation.state"].squeeze(0) for o in obs_list], dim=0
        ).unsqueeze(0)  # (1, n_obs, S)
        for cam in cfg.cameras:
            key = f"observation.images.{cam}"
            obs_batch[key] = torch.stack(
                [o[key].squeeze(0) for o in obs_list], dim=0
            ).unsqueeze(0)  # (1, n_obs, C, H, W)
            if cfg.use_depth:
                dkey = f"observation.depths.{cam}"
                if dkey in obs_list[0]:
                    obs_batch[dkey] = torch.stack(
                        [o[dkey].squeeze(0) for o in obs_list], dim=0
                    ).unsqueeze(0)

        # Normalize observations.
        obs_batch = self.normalize_minmax(obs_batch)
        obs_batch = self.normalize_images(obs_batch)

        # Encode -> global conditioning.
        global_cond = self._encode_obs(obs_batch)  # (1, global_cond_dim)

        # --- Conditional denoising loop: noise -> action trajectory ---
        device = global_cond.device
        # Start from pure Gaussian noise.
        trajectory = torch.randn(
            (1, cfg.horizon, cfg.action_dim), device=device
        )

        # Set up the reverse timestep schedule.
        self.noise_scheduler.set_timesteps(cfg.num_inference_steps, cfg.sampler)
        timesteps = self.noise_scheduler.timesteps

        for i, t in enumerate(timesteps):
            # Predict noise at this step.
            t_tensor = torch.tensor([t], device=device, dtype=torch.long)
            noise_pred = self.unet(trajectory, t_tensor, global_cond)

            # Reverse step: remove some noise.
            if cfg.sampler == "ddpm":
                trajectory = self.noise_scheduler.step(noise_pred, t, trajectory, sampler="ddpm")
            else:
                # DDIM needs the PREVIOUS timestep in the subsequence.
                prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else -1
                trajectory = self.noise_scheduler.step(
                    noise_pred, t, trajectory, prev_timestep=prev_t, sampler="ddim"
                )

        # trajectory is now (1, horizon, A) in [-1,1] normalized actions.
        # Unnormalize to real control units.
        trajectory = self.unnormalize_action(trajectory)  # (1, horizon, A)

        # --- Slice the executed portion and buffer it ---
        # The action indices we execute: starting from slot (n_obs_steps-1)
        # (the one aligned with the current frame) for n_action_steps.
        start = cfg.n_obs_steps - 1
        end = start + cfg.n_action_steps
        actions = trajectory[0, start:end]  # (n_action_steps, A)

        # Push into the queue (each is (1, A) for consistency with ACT interface).
        for a in actions:
            self._action_queue.append(a.unsqueeze(0))

        return self._action_queue.popleft()
