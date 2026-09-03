"""Dataset plumbing for Diffusion Policy training.

We reuse lerobot's `LeRobotDataset` purely as a *reader* (the recorder already
wrote the data in this format). The DP-specific piece is `delta_timestamps`,
which makes the loader return, per sample:
  - a short window of PAST observations   (n_obs_steps frames), and
  - a `horizon`-long window of actions     (the trajectory the U-Net denoises),
plus an `action_is_pad` mask for windows that run past the episode boundary.

Index alignment (this is the fiddly bit -- follows the reference DP setup):
  Let current frame = 0. With n_obs_steps=2, horizon=16, n_action_steps=8:
    observation indices : [-1, 0]                     (previous + current)
    action indices      : [-1, 0, 1, ..., 14]         (16 steps, starting 1 back)
  So the predicted trajectory is aligned to the observation window, and at
  inference we execute action slots [n_obs_steps-1 : n_obs_steps-1+n_action_steps]
  = [1:9], i.e. the 8 actions that follow the current observation.

Note: lerobot's `policies/__init__` is broken in this checkout (a groot import
clash), but `lerobot.datasets` imports cleanly, so we touch only that.
"""
import os

# Local-only dataset; never reach the Hub. Set before importing lerobot.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# Known OpenMP duplicate-runtime quirk on this machine.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from config import DiffusionConfig


def observation_delta_indices(cfg: DiffusionConfig) -> list[int]:
    # Past observation window: [-(n_obs_steps-1), ..., 0]. E.g. n_obs_steps=2 -> [-1, 0].
    return list(range(1 - cfg.n_obs_steps, 1))


def action_delta_indices(cfg: DiffusionConfig) -> list[int]:
    # Action horizon, aligned so slot (n_obs_steps-1) is the action at the current
    # frame. E.g. n_obs_steps=2, horizon=16 -> [-1, 0, 1, ..., 14].
    start = 1 - cfg.n_obs_steps
    return list(range(start, start + cfg.horizon))


def _delta_timestamps(cfg: DiffusionConfig) -> dict:
    """Map each observation/action key to the timestamps (seconds) it should be
    sampled at, relative to the current frame. lerobot converts these to frame
    offsets internally and returns stacked windows."""
    obs_ts = [i / cfg.fps for i in observation_delta_indices(cfg)]
    act_ts = [i / cfg.fps for i in action_delta_indices(cfg)]
    dt = {
        "observation.state": obs_ts,
        "action": act_ts,
    }
    for cam in cfg.cameras:
        dt[f"observation.images.{cam}"] = obs_ts
        if cfg.use_depth:
            dt[f"observation.depths.{cam}"] = obs_ts
    return dt


def episode_split(cfg: DiffusionConfig) -> tuple[list[int], list[int]]:
    """Split episode indices into train/val (split by *episode*, never by frame,
    so val frames are from demos the policy never trained on)."""
    meta = LeRobotDatasetMetadata(cfg.repo_id, root=cfg.data_root)
    # Use the actual episodes table length (not info.json which may be stale).
    n = len(meta.episodes)
    rng = np.random.default_rng(cfg.seed)
    order = rng.permutation(n)
    n_val = max(1, int(round(n * cfg.val_fraction)))
    val = sorted(order[:n_val].tolist())
    train = sorted(order[n_val:].tolist())
    return train, val


class _ResizeImages(torch.utils.data.Dataset):
    """Downsample image/depth tensors to cfg.resize_hw INSIDE the worker, before
    collation. This is the fix for /dev/shm exhaustion: the DataLoader ships the
    resized (small) tensors between worker and main process, not native 480x640.

    Why here and not only in the model: the model's on-GPU resize still runs for
    rollout (MuJoCo feeds native frames straight to the policy, never through this
    dataset). During training the model-side resize just sees already-correct
    sizes and no-ops. Two resize sites, one shared target size -- consistent.

    lerobot returns image tensors as float32 CHW in [0,1]; delta_timestamps stacks
    them to (n_obs, C, H, W), which is already the 4D shape F.interpolate wants.
    """

    def __init__(self, ds, cfg: DiffusionConfig):
        self.ds = ds
        self.size = tuple(cfg.resize_hw) if cfg.resize_hw is not None else None
        self.keys = [f"observation.images.{c}" for c in cfg.cameras]
        if cfg.use_depth:
            self.keys += [f"observation.depths.{c}" for c in cfg.cameras]

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        sample = self.ds[i]
        if self.size is not None:
            for k in self.keys:
                v = sample.get(k)
                if v is not None and v.shape[-2:] != self.size:
                    sample[k] = F.interpolate(
                        v, size=self.size, mode="bilinear", align_corners=False
                    )
        return sample


def make_datasets(cfg: DiffusionConfig):
    """Return (train_ds, val_ds, stats). `stats` is the raw lerobot stats dict
    used by the model for normalization."""
    train_eps, val_eps = episode_split(cfg)
    delta = _delta_timestamps(cfg)
    common = dict(root=cfg.data_root, delta_timestamps=delta)

    train_ds = LeRobotDataset(cfg.repo_id, episodes=train_eps, **common)
    val_ds = LeRobotDataset(cfg.repo_id, episodes=val_eps, **common)
    stats = train_ds.meta.stats
    print(f"[dataset] train: {len(train_eps)} eps / {len(train_ds)} frames | "
          f"val: {len(val_eps)} eps / {len(val_ds)} frames")
    # Resize images in-worker to keep the DataLoader shm working set small.
    train_ds = _ResizeImages(train_ds, cfg)
    val_ds = _ResizeImages(val_ds, cfg)
    return train_ds, val_ds, stats


def make_loaders(cfg: DiffusionConfig):
    train_ds, val_ds, stats = make_datasets(cfg)
    common = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device == "cuda"),
        drop_last=False,
        persistent_workers=cfg.num_workers > 0,
    )
    if cfg.num_workers > 0:
        # Cap prefetch depth so the shm working set stays under /dev/shm (see config).
        common["prefetch_factor"] = cfg.prefetch_factor
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True, **common)
    val_loader = torch.utils.data.DataLoader(val_ds, shuffle=False, **common)
    return train_loader, val_loader, stats


# --- batch helpers ------------------------------------------------------------
# A collated batch is a dict with:
#   "observation.state"            (B, n_obs_steps, state_dim)
#   "observation.images.{cam}"     (B, n_obs_steps, 3, H, W)   float in [0,1]
#   "observation.depths.{cam}"     (B, n_obs_steps, 3, H, W)   (only if use_depth)
#   "action"                       (B, horizon, action_dim)
#   "action_is_pad"                (B, horizon) bool
# (default_collate stacks these; "task" comes through as a list[str] we ignore)

def batch_to_device(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out
