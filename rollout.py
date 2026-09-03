"""Closed-loop evaluation of a trained Diffusion Policy in MuJoCo (mac / glfw).

This is the *only* measure of real performance: training MSE on noise prediction
correlates even more weakly with task success than ACT's L1. Here we load a
checkpoint (EMA weights), spawn randomized scenes, drive the arm with the policy
(receding-horizon: denoise every `n_action_steps` frames), and check whether the
object ends up in the bin.

Key difference from ACT rollout:
  - ACT calls the policy EVERY frame and temporally ensembles overlapping chunks.
  - Diffusion also calls select_action EVERY frame, but internally the policy
    only runs the expensive denoising loop when its action queue is empty (every
    n_action_steps frames). Between replanning events, it pops pre-computed
    actions from the queue — so it's still one action per frame, just cheaper
    on most frames.
  - DP uses `n_obs_steps=2` (previous + current frame), so the policy maintains
    an internal observation history buffer (handled inside select_action).

Run from Panda/ (one level above Diffusion/) so `import control`/`scene_gen` resolve:
    cd Panda
    KMP_DUPLICATE_LIB_OK=TRUE MUJOCO_GL=glfw \
      ~/miniconda3/envs/python_robotics/bin/python Diffusion/rollout.py \
      --checkpoint Diffusion/checkpoints/best.pt --episodes 20
"""
import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MUJOCO_GL", "glfw")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import mujoco
import torch

# Diffusion package lives in ./Diffusion; add it so flat imports resolve.
HERE = os.path.dirname(os.path.abspath(__file__))
PANDA = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PANDA)

from config import DiffusionConfig
# Must match the build train.py used (checkpoint key names differ between the
# from-scratch and library timestep encoders). train.py imports model_lib, so
# rollout does too; swap BOTH to 'from model import ...' to use the from-scratch build.
from model_lib import DiffusionPolicy

import control as C
from scene_gen import SceneManager


def load_policy(ckpt_path: str, device: str, stats_path: str) -> tuple[DiffusionPolicy, DiffusionConfig]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = DiffusionConfig(**ckpt["config"])
    cfg.device = device
    stats = json.load(open(stats_path))
    policy = DiffusionPolicy(cfg, stats)
    policy.load_state_dict(ckpt["model"])
    policy.to(device).eval()
    return policy, cfg


def build_obs(viewer, cams, device, use_depth: bool = False) -> dict:
    """Grab cameras + proprio state into the policy's expected obs dict.

    Note: for Diffusion Policy the obs is a SINGLE frame here; the policy
    internally accumulates a window of n_obs_steps frames via its obs queue.
    This differs from the training batch (which stacks the window upfront)
    because at rollout each frame arrives one at a time."""
    obs = {}
    state = np.concatenate([C.data.qpos[:7], [C.data.qpos[7]]]).astype(np.float32)
    obs["observation.state"] = torch.from_numpy(state).unsqueeze(0).to(device)  # (1, 8)
    for cam in cams:
        if use_depth:
            from recorder import depth_to_uint8
            rgb, dep = viewer.grab_rgbd(cam)
        else:
            rgb = viewer.grab(cam)
            dep = None
        t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
        obs[f"observation.images.{cam}"] = t.unsqueeze(0).to(device)  # (1, 3, H, W)
        if use_depth:
            dep_u8 = depth_to_uint8(dep)
            td = torch.from_numpy(dep_u8).permute(2, 0, 1).float() / 255.0
            obs[f"observation.depths.{cam}"] = td.unsqueeze(0).to(device)
    return obs


def is_success(info: dict) -> bool:
    """Object resting inside the bin (same criterion as ACT rollout)."""
    bx, by, bin_z = info["bin_pose"]
    half = info["bin"]["half"]
    rim = bin_z + info["bin"]["wh"]
    obj = C.data.body("object").xpos
    in_xy = abs(obj[0] - bx) <= half and abs(obj[1] - by) <= half
    below_rim = obj[2] <= rim + 0.01
    on_floor = obj[2] > 0.0
    return bool(in_xy and below_rim and on_floor)


def run_episode(policy, cfg, sm, rng, viewer, max_steps, render=True) -> bool:
    info = C.reset_episode(sm, rng, viewer)
    policy.reset()  # clear obs/action queues for a fresh episode
    substeps = max(1, round((1.0 / cfg.fps) / C.SIM_DT))

    success = False
    for _ in range(max_steps):
        obs = build_obs(viewer, cfg.cameras, cfg.device, use_depth=cfg.use_depth)
        action = policy.select_action(obs)[0].cpu().numpy()  # (8,) absolute targets
        C.data.ctrl[:7] = action[:7]
        C.data.ctrl[C.GRIPPER] = action[7]
        np.clip(C.data.ctrl, C.CTRL_RANGE[:, 0], C.CTRL_RANGE[:, 1], out=C.data.ctrl)
        for _ in range(substeps):
            mujoco.mj_step(C.model, C.data)
        if render:
            viewer.render()
        if is_success(info):
            success = True
            break
    return success


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="Diffusion/checkpoints/best.pt")
    ap.add_argument("--stats", default="Diffusion/data/panda_pick_place/meta/stats.json")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=500, help="policy frames per episode (@fps)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="cpu/mps/cuda (default: auto)")
    ap.add_argument("--no_render", action="store_true")
    args = ap.parse_args()

    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    policy, cfg = load_policy(args.checkpoint, device, args.stats)
    print(f"[rollout] device={device} episodes={args.episodes} "
          f"horizon={cfg.horizon} n_action_steps={cfg.n_action_steps} "
          f"sampler={cfg.sampler} inference_steps={cfg.num_inference_steps}")

    sm = SceneManager()
    rng = np.random.default_rng(args.seed)
    viewer = C.Viewer(title="Diffusion Policy rollout")

    results = []
    for ep in range(args.episodes):
        ok = run_episode(policy, cfg, sm, rng, viewer,
                         max_steps=args.max_steps, render=not args.no_render)
        results.append(ok)
        print(f"  episode {ep + 1:2d}/{args.episodes}: {'SUCCESS' if ok else 'fail'} "
              f"(running {sum(results)}/{len(results)} = {np.mean(results):.0%})")

    viewer.close()
    print(f"\n[rollout] success rate: {sum(results)}/{len(results)} = {np.mean(results):.1%}")


if __name__ == "__main__":
    main()
