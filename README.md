# Diffusion Policy — Conditional DDPM over Action Trajectories (Panda pick-and-place)

From-scratch Diffusion Policy implementation ([Chi et al. 2023](https://arxiv.org/abs/2303.04137))
for the MuJoCo Panda pick-and-place task. Trained offline on teleop demos
(LeRobot dataset), evaluated closed-loop in the sim.

## What Diffusion Policy is (1-minute recap)
A **conditional denoising diffusion model (DDPM)** that generates action
trajectories from noise, conditioned on observations.
- **Forward process** (training only): corrupt a ground-truth action trajectory
  with scheduled Gaussian noise: x_t = √ᾱ_t·x₀ + √(1-ᾱ_t)·ε.
- **U-Net**: a 1D temporal convolutional network that predicts the noise ε̂ given
  the noisy trajectory x_t, the diffusion timestep t, and an observation
  embedding (injected via FiLM — feature-wise linear modulation).
- **Vision encoder**: ResNet18 (GroupNorm + SpatialSoftmax) -> per-camera
  keypoint embeddings. Shared across cameras and observation steps.
- **Inference**: start from x_T ~ N(0,I), iteratively denoise using the U-Net
  for T steps (DDPM, stochastic) or fewer (DDIM, deterministic). Then execute
  `n_action_steps` of the denoised trajectory in the environment.
- **Loss**: **MSE(ε̂, ε)** — the network learns to predict what noise was added.
- **EMA**: exponential moving average of weights for stable inference.

## How it differs from ACT (side-by-side learning)
| aspect | ACT | Diffusion Policy |
|--------|-----|-----------------|
| generation | one-shot CVAE (1 forward pass) | iterative denoising (10-100 passes) |
| backbone | transformer encoder/decoder | 1D temporal U-Net + FiLM |
| conditioning | cross-attention on obs tokens | global conditioning vector |
| action horizon | chunk_size=100 | horizon=16 |
| execution | temporal ensembling (every frame) | receding horizon (execute 8, replan) |
| obs window | 1 frame | 2 frames (n_obs_steps) |
| normalization | mean/std | min-max to [-1,1] (clip-bounded) |
| loss | L1 + KL divergence | MSE on noise |
| inference speed | 1 forward pass per frame | 10 U-Net forward passes per replan |

## Two builds: from-scratch vs. library
There are **two interchangeable network files** with the identical
`DiffusionPolicy` / `EMAModel` interface — `train.py` picks one via its import line:
- **`model.py`** — everything hand-rolled (the DDPM/DDIM math, sinusoidal timestep
  embedding, and EMA are all derived from first principles). This is the *learning*
  reference: read it to understand what actually happens.
- **`model_lib.py`** — the same network, but the noise scheduler → `diffusers.DDPMScheduler`/
  `DDIMScheduler`, the timestep encoder → `diffusers.Timesteps`+`TimestepEmbedding`, and
  EMA → `diffusers.training_utils.EMAModel`. **This is what we train with**, because the
  scheduler is the highest-risk piece to hand-roll and it fails *silently* (loss still drops,
  but samples drift) — not worth risking on a multi-hour GPU run.

What stays custom in **both** (no library ships them): **SpatialSoftmax** (only in the heavy
`robomimic` dep), the **FiLM conditional 1D U-Net body**, and the normalization buffers.
To train the pure from-scratch version, change one line in `train.py`:
`from model_lib import ...` → `from model import ...`.

## Files
| file | role |
|------|------|
| `config.py`  | `DiffusionConfig` dataclass — all dims/hparams, one source of truth |
| `dataset.py` | `LeRobotDataset` reader + observation window / action horizon delta_timestamps |
| `model.py`   | **from-scratch** network (ResNet+SpatialSoftmax vision, 1D U-Net+FiLM, hand-rolled NoiseScheduler, EMA, DiffusionPolicy) |
| `model_lib.py` | **library-backed** drop-in twin: diffusers schedulers + timestep-embed + EMA; custom SpatialSoftmax/FiLM U-Net shared with `model.py`. Used by `train.py` |
| `train.py`   | offline training loop (AdamW, AMP on CUDA, EMA, val-loss → `best.pt`) |
| `rollout.py` | closed-loop eval in MuJoCo (mac/glfw) → task success rate |

## Setup notes
- **State** = `[qpos[:7], finger_pos]` (8-d). **Action** = `[ctrl[:7], gripper_ctrl 0-255]` (8-d, absolute joint targets).
- 3 cameras: `front`, `diag` (fixed) + `wrist` (eye-in-hand), 480×640.
- RGBD toggle via `use_depth` config flag (dataset currently RGB-only).
- Normalization: action/state use **min-max to [-1,1]** (images: mean/std). Min-max is required because the DDPM reverse process clips intermediate predictions to [-1,1].
- Imports only `lerobot.datasets` (same as ACT; lerobot.policies is broken in this checkout).
- `KMP_DUPLICATE_LIB_OK=TRUE` required (OpenMP duplicate-runtime quirk on this machine).

## Training (SageMaker Studio, g6 / CUDA)
Run **from inside `Diffusion/`** (flat imports: `from config import ...`).

```bash
# 1. Dataset sync (same data as ACT — one S3 location, two policies).
aws s3 sync s3://<bucket>/panda_pick_place ./local_data/panda_pick_place

# 2. Verify video decode (same check as ACT).
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; \
ds=D('panda_pick_place', root='./local_data/panda_pick_place'); \
print(ds[0]['observation.images.front'].shape)"

# 3. Train
KMP_DUPLICATE_LIB_OK=TRUE python train.py \
  --data_root ./local_data/panda_pick_place \
  --n_steps 100000 --batch_size 64

# Smoke test (tiny end-to-end shape/sanity check, any device):
python train.py --smoke --device cpu --data_root data/panda_pick_place
```
Checkpoints: `best.pt` (lowest val MSE, EMA weights), periodic `step_NNNNNN.pt`,
`last.pt`, and `ema_state.pt`. `train_log.csv` logs MSE/val/ema_decay.

## Rollout / evaluation (mac, glfw)
Closed-loop is the **only** real success metric (diffusion val MSE is an even
weaker proxy than ACT's val L1). Run **from `Panda/`** (one level above
`Diffusion/`) so `control`/`scene_gen` resolve. Pull `best.pt` from Studio first.

```bash
cd Panda
KMP_DUPLICATE_LIB_OK=TRUE MUJOCO_GL=glfw \
  ~/miniconda3/envs/python_robotics/bin/python Diffusion/rollout.py \
  --checkpoint Diffusion/checkpoints/best.pt --episodes 20
# --no_render to skip the live window, --device cpu/mps/cuda, --max_steps N
```
Reports per-episode SUCCESS/fail and the overall success rate.
