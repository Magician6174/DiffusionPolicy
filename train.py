"""Train Diffusion Policy on the Panda pick-and-place dataset.

Pure offline imitation: sample (obs_window, action_horizon) -> add noise to
actions -> U-Net predicts noise -> MSE loss -> backprop. The environment is
never touched here; closed-loop success is measured separately by rollout.py.

Key difference from ACT training:
  - Loss is MSE(ε̂, ε) on the noise prediction, not L1+KL on actions directly.
  - EMA of weights is tracked and used for checkpointing (the EMA model is what
    gets evaluated; raw training weights are noisier and perform worse).
  - AdamW with betas=(0.95, 0.999) and weight_decay=1e-6 (DP convention; the
    aggressive beta1=0.95 is from DDPM/score-matching literature).

Usage (SageMaker Studio, CUDA):
    python train.py --data_root /home/.../panda_pick_place --n_steps 100000
    python train.py --smoke            # tiny end-to-end shape/sanity check
"""
import argparse
import csv
import json
import time
from dataclasses import fields
from pathlib import Path

import torch

from config import DiffusionConfig
from dataset import make_loaders, batch_to_device
from model_lib import DiffusionPolicy, EMAModel  # library-backed build; swap to 'from model import ...' for the fully from-scratch version


def parse_args() -> DiffusionConfig:
    p = argparse.ArgumentParser()
    for f in fields(DiffusionConfig):
        if f.type in (int, float, str):
            p.add_argument(f"--{f.name}", type=eval(f.type) if isinstance(f.type, str) else f.type)
        elif f.type == bool or f.name == "use_amp":
            p.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"))
    p.add_argument("--smoke", action="store_true", help="tiny run to validate the pipeline")
    args = p.parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "smoke" and v is not None}
    cfg = DiffusionConfig(**overrides)
    if args.smoke:
        cfg.batch_size = 2
        cfg.num_workers = 0
        cfg.n_steps = 4
        cfg.val_freq = 2
        cfg.save_freq = 1000
        cfg.log_freq = 1
    return cfg


def build_optimizer(policy: DiffusionPolicy, cfg: DiffusionConfig):
    """Separate LR groups for backbone (pretrained) vs rest (from scratch).

    DP typically uses a single LR for everything (unlike ACT which decouples
    backbone), but we preserve the split for flexibility. Default: both 1e-4."""
    backbone, rest = [], []
    for n, pm in policy.named_parameters():
        if not pm.requires_grad:
            continue
        (backbone if "vision_encoder.backbone" in n else rest).append(pm)
    return torch.optim.AdamW(
        [
            {"params": rest, "lr": cfg.optimizer_lr},
            {"params": backbone, "lr": cfg.optimizer_lr_backbone},
        ],
        betas=cfg.adam_betas,
        eps=cfg.adam_eps,
        weight_decay=cfg.optimizer_weight_decay,
    )


def cycle(loader):
    """Infinite batch stream. See ACT/train.py for the rationale."""
    while True:
        for b in loader:
            yield b


@torch.no_grad()
def evaluate(policy, val_loader, cfg, max_batches=50):
    """Compute validation MSE (noise prediction error on held-out episodes).

    Note: unlike classification val loss, diffusion val MSE is a very noisy signal
    because each sample uses a random timestep. It's still useful as a coarse
    overfitting indicator, but closed-loop rollout is the real success metric."""
    policy.train()  # keep the same mode as training for consistency
    tot, n = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        batch = batch_to_device(batch, cfg.device)
        _, info = policy.compute_loss(batch)
        tot += info["mse_loss"]
        n += 1
    return tot / max(n, 1)


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    print(f"[train] device={cfg.device} amp={cfg.use_amp} bs={cfg.batch_size} "
          f"steps={cfg.n_steps} horizon={cfg.horizon} n_obs={cfg.n_obs_steps}")

    train_loader, val_loader, stats = make_loaders(cfg)
    policy = DiffusionPolicy(cfg, stats).to(cfg.device)
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"[train] trainable params: {n_params/1e6:.1f}M")

    optimizer = build_optimizer(policy, cfg)
    amp_device = "cuda" if cfg.device == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(enabled=cfg.use_amp)

    # EMA: maintain a smoothed copy of the model for evaluation/checkpointing.
    ema = EMAModel(policy, cfg) if cfg.use_ema else None

    log_path = out / "train_log.csv"
    with open(log_path, "w", newline="") as fp:
        csv.writer(fp).writerow(["step", "mse_loss", "val_loss", "sec_per_step", "ema_decay"])

    best_val = float("inf")
    data_iter = cycle(train_loader)
    policy.train()
    t0 = time.time()

    for step in range(1, cfg.n_steps + 1):
        batch = batch_to_device(next(data_iter), cfg.device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=amp_device, enabled=cfg.use_amp):
            loss, info = policy.compute_loss(batch)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        # Update EMA after each step.
        if ema is not None:
            ema.step(policy)

        val = ""
        if step % cfg.val_freq == 0 or step == cfg.n_steps:
            # For validation, temporarily load EMA weights.
            if ema is not None:
                # Save current training weights, load EMA, evaluate, then restore.
                train_state = {k: v.clone() for k, v in policy.state_dict().items()}
                ema.copy_to(policy)
                val = evaluate(policy, val_loader, cfg)
                policy.load_state_dict(train_state)
            else:
                val = evaluate(policy, val_loader, cfg)
            policy.train()
            if val < best_val:
                best_val = val
                # Save EMA weights (or raw if no EMA) as best checkpoint.
                save_state = policy.state_dict()
                if ema is not None:
                    # Clone AFTER loading EMA: state_dict() tensors share storage
                    # with the live params, so the in-place restore below would
                    # otherwise clobber save_state back to the training weights.
                    ema.copy_to(policy)
                    save_state = {k: v.clone() for k, v in policy.state_dict().items()}
                    policy.load_state_dict(train_state)
                torch.save({"step": step, "model": save_state,
                            "config": cfg.to_dict(), "val_loss": val},
                           out / "best.pt")
                print(f"[train] step {step}: new best val_loss={val:.6f} -> best.pt")

        if step % cfg.log_freq == 0 or step == cfg.n_steps:
            sps = (time.time() - t0) / cfg.log_freq
            t0 = time.time()
            decay_str = f"{ema._get_decay():.6f}" if ema else "N/A"
            print(f"step {step}/{cfg.n_steps} mse={info['mse_loss']:.6f} "
                  f"val={val if val == '' else f'{val:.6f}'} ({sps:.2f}s/it) "
                  f"ema_decay={decay_str}")
            with open(log_path, "a", newline="") as fp:
                csv.writer(fp).writerow([step, info["mse_loss"], val, f"{sps:.3f}", decay_str])

        if step % cfg.save_freq == 0:
            save_state = policy.state_dict()
            if ema is not None:
                train_state_periodic = {k: v.clone() for k, v in policy.state_dict().items()}
                ema.copy_to(policy)
                # Clone the EMA weights before the in-place restore (see best.pt note).
                save_state = {k: v.clone() for k, v in policy.state_dict().items()}
                policy.load_state_dict(train_state_periodic)
            torch.save({"step": step, "model": save_state,
                        "config": cfg.to_dict()}, out / f"step_{step:06d}.pt")

    # Save last checkpoint (EMA weights).
    save_state = policy.state_dict()
    if ema is not None:
        ema.copy_to(policy)
        save_state = policy.state_dict()
    torch.save({"step": cfg.n_steps, "model": save_state,
                "config": cfg.to_dict()}, out / "last.pt")
    if ema is not None:
        torch.save(ema.state_dict(), out / "ema_state.pt")
    print(f"[train] done. best val_loss={best_val:.6f}. checkpoints in {out}")


if __name__ == "__main__":
    main()
