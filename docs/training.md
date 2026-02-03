# Swarm RL Training Guide

This document describes the curriculum training scripts and all configurable options for training drone policies on the Swarm subnet.

---

## 1. Overview

Training is split into **three phases**:

| Phase | Script | Task mix | Default behavior |
|-------|--------|----------|-------------------|
| **1** | `RL/train_phase1_open_only.py` | Open only (challenge type 4, no obstacles) | **Reuses last Phase 1 model by default** if it exists; use `--load ''` to train from scratch |
| **2** | `RL/train_phase2_easy_obstacles.py` | Open + easy obstacles (types 4 and 3) | **Loads Phase 1 model by default** to enhance |
| **3** | `RL/train_phase3_full_mix.py` | All challenge types (same as validator) | **Loads Phase 2 model by default** to enhance |

Each phase uses a **resettable task wrapper**: every episode gets a **new random task** (new seed), so the policy sees many different scenarios instead of a fixed set.

Each phase also uses a **phase-specific reward function**. You can customize rewards per phase (e.g. denser shaping in phase 1, obstacle penalties in phase 2) by editing **`RL/rewards.py`** (`phase1_reward`, `phase2_reward`, `phase3_reward`). See section **14. Phase-specific reward functions** below.

---

## 2. Dependencies

Install from repo root:

```bash
pip install -r requirements.txt
```

Required for RL training include: `stable-baselines3`, `torch`, `gymnasium`, and the Swarm env (e.g. `gym-pybullet-drones` from the repo). If you see `ModuleNotFoundError: No module named 'stable_baselines3'`, run:

```bash
pip install stable-baselines3 torch gymnasium
pip install -r requirements.txt
```

---

## 3. Seed and number of envs

- **`--seed`** (int, optional): Global RNG seed. When set, training is reproducible (same sequence of tasks and updates).
- **`--n-envs`** (int, default: 4): Number of parallel environments. More envs = more diverse experience per update and faster training (at higher CPU/memory cost).

Example:

```bash
python RL/train_phase1_open_only.py --seed 42 --n-envs 8
```

---

## 4. Hyperparameters

All phase scripts accept the same PPO/A2C hyperparameters:

| Option | Default | Description |
|--------|---------|-------------|
| `--lr` | 3e-4 | Learning rate |
| `--n-steps` | 2048 | Steps per env per update (PPO/A2C) |
| `--batch-size` | 64 | Minibatch size for gradient updates |
| `--gamma` | 0.99 | Discount factor |
| `--ent-coef` | 0.01 | Entropy coefficient (exploration) |
| `--clip-range` | 0.2 | PPO clip range (PPO only) |
| `--gae-lambda` | 0.95 | GAE lambda for advantage estimation |

Example:

```bash
python RL/train_phase1_open_only.py --timesteps 200000 --lr 1e-4 --n-steps 4096 --batch-size 128
```

---

## 5. Evaluation logging

Evaluation runs on **fixed held-out seeds** every `--eval-freq` steps and logs to TensorBoard:

- **eval/success_rate**: Fraction of eval episodes that reached the goal.
- **eval/mean_time_sec**: Mean time to goal (or 0 for failures).
- **eval/mean_reward**: Mean episode return.

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--eval-freq` | 10000 | Run evaluation every N steps |
| `--n-eval-episodes` | 10 | Number of episodes per evaluation |
| `--eval-seed-start` | 100000 | First seed for eval (seeds eval_seed_start .. eval_seed_start + n_eval_episodes - 1) |

TensorBoard:

```bash
tensorboard --logdir logs/
```

Then open the run for `phase1`, `phase2`, or `phase3` depending on which script you ran.

---

## 6. Checkpointing and resume

- **`--save-freq`** (int, default: 50000): Save a checkpoint every N steps.
- **`--checkpoint-dir`** (str): Directory for checkpoints (e.g. `logs/phase1_checkpoints`).
- **`--resume`** (str): Path to a checkpoint ZIP to **continue training** (same phase). Overrides `--load` when both are set and the resume path exists.

Example:

```bash
# Save checkpoints every 25k steps
python RL/train_phase1_open_only.py --timesteps 100000 --save-freq 25000 --checkpoint-dir logs/phase1_checkpoints

# Later: resume from a checkpoint
python RL/train_phase1_open_only.py --timesteps 150000 --resume logs/phase1_checkpoints/phase1_50000_steps.zip
```

Phase 2 and 3 also support **`--load`** to start from the **previous phase’s** final model (see below).

---

## 7. TensorBoard log directory

- **`--tensorboard-log`** (str): Directory for TensorBoard logs (e.g. `logs/phase1`). Defaults are `logs/phase1`, `logs/phase2`, `logs/phase3` for the three scripts.

---

## 8. Device selection

- **`--device`** (str, default: `"auto"`): Device for the model. Use `cpu`, `cuda`, or `cuda:0` (etc.) to force a device; `auto` lets SB3 choose.

Example:

```bash
python RL/train_phase1_open_only.py --device cuda
```

---

## 9. Algorithm selection

- **`--algo`** (str, default: `ppo`): Algorithm. Choices: `ppo`, `a2c`.

Example:

```bash
python RL/train_phase3_full_mix.py --algo a2c --timesteps 200000
```

When using **`--load`** or **`--resume`**, the loaded checkpoint must have been trained with the same algorithm.

---

## 10. Phase-specific options

### Phase 1

- **`--load`** (default: `swarm/submission_template/phase1_open_only.zip`): **Reuse the last Phase 1 model** and continue training. If the file exists, it is loaded; otherwise training starts from scratch. Set to `''` to always train from scratch.
- **`--resume`**: Resume from a Phase 1 checkpoint (e.g. from `--checkpoint-dir`); takes precedence over `--load` if the file exists.

### Phase 2

- **`--load`** (default: `swarm/submission_template/phase1_open_only.zip`): **Load the previous phase (Phase 1) model to enhance.** Set to `''` to train from scratch.
- **`--resume`**: Resume from a Phase 2 checkpoint (takes precedence over `--load` if the file exists).

### Phase 3

- **`--load`** (default: `swarm/submission_template/phase2_easy_obstacles.zip`): **Load the previous phase (Phase 2) model to enhance.** Set to `''` to train from scratch.
- **`--resume`**: Resume from a Phase 3 checkpoint.
- **`--output`** (default: `phase3_full_mix.zip`): Output filename under `swarm/submission_template/`. Use e.g. `ppo_policy.zip` if you want the same name as the submission template.

---

## 11. Full command reference (Phase 1)

```text
python RL/train_phase1_open_only.py [OPTIONS]

  --timesteps          Total training steps (default: 100000)
  --load               Reuse Phase 1 model path (default: swarm/.../phase1_open_only.zip). Set to '' for from-scratch.
  --seed               Global RNG seed (optional)
  --n-envs             Number of parallel envs (default: 4)
  --lr                 Learning rate (default: 3e-4)
  --n-steps            Steps per env per update (default: 2048)
  --batch-size         Minibatch size (default: 64)
  --gamma              Discount factor (default: 0.99)
  --ent-coef           Entropy coefficient (default: 0.01)
  --clip-range         PPO clip range (default: 0.2)
  --gae-lambda         GAE lambda (default: 0.95)
  --device             Device: auto|cpu|cuda|cuda:0 (default: auto)
  --algo               Algorithm: ppo|a2c (default: ppo)
  --resume             Path to checkpoint to continue training
  --save-freq          Save checkpoint every N steps (default: 50000)
  --checkpoint-dir     Checkpoint directory (default: logs/phase1_checkpoints)
  --tensorboard-log    TensorBoard log dir (default: logs/phase1)
  --eval-freq          Evaluation frequency in steps (default: 10000)
  --n-eval-episodes    Eval episodes per evaluation (default: 10)
  --eval-seed-start    First seed for eval episodes (default: 100000)
```

Phase 2 and Phase 3 add **`--load`** and (Phase 3) **`--output`**; other options are the same.

---

## 12. Recommended workflow

1. **Phase 1** (open only):
   ```bash
   python RL/train_phase1_open_only.py --timesteps 100000 --n-envs 4 --seed 42
   ```
   Check TensorBoard for `eval/success_rate`; run longer if needed.

2. **Phase 2** (open + easy obstacles; loads Phase 1 by default):
   ```bash
   python RL/train_phase2_easy_obstacles.py --timesteps 100000 --n-envs 4 --seed 42
   ```

3. **Phase 3** (full mix; loads Phase 2 by default):
   ```bash
   python RL/train_phase3_full_mix.py --timesteps 200000 --n-envs 4 --output ppo_policy.zip
   ```

4. Use the final model (e.g. `swarm/submission_template/ppo_policy.zip`) in `drone_agent.py` and build your miner submission ZIP.

---

## 13. File layout

| Path | Purpose |
|------|--------|
| `RL/train_phase1_open_only.py` | Phase 1 training script |
| `RL/train_phase2_easy_obstacles.py` | Phase 2 training script |
| `RL/train_phase3_full_mix.py` | Phase 3 training script |
| `RL/env_wrapper.py` | `ResettableTaskWrapper`, `RewardWrapper` |
| `RL/rewards.py` | Phase-specific reward functions (edit for custom rewards) |
| `RL/callbacks.py` | Eval callback (success rate, time, reward) and eval helper |
| `docs/training.md` | This document |

Checkpoints and logs are written under `logs/` by default (see `--checkpoint-dir` and `--tensorboard-log`).

---

## 14. Phase-specific reward functions

Reward logic is split per phase so you can tune it for more complicated training later.

**File:** `RL/rewards.py`

- **`phase1_reward(prev_info, info, r_orig, terminated, truncated)`** — used in Phase 1 (open only). Default: returns `r_orig` (env reward unchanged).
- **`phase2_reward(...)`** — used in Phase 2 (open + easy obstacles). Default: returns `r_orig`.
- **`phase3_reward(...)`** — used in Phase 3 (full mix). Default: returns `r_orig`.

**Arguments:**

- `prev_info`, `info`: dicts with keys such as `distance_to_goal`, `success`, `collision`, `score`, `t_to_goal` (see the env’s `_computeInfo`).
- `r_orig`: reward from the underlying env (increment of `flight_reward`).
- `terminated`, `truncated`: episode end flags.

**Return:** The reward (float) used for this step.

**Customization examples:**

- **Phase 1:** Add dense shaping, e.g. `r = r_orig + 0.1 * (prev_info.get("distance_to_goal", 0) - info.get("distance_to_goal", 0))` (reward for getting closer to goal), capped so it doesn’t dominate.
- **Phase 2:** Add a collision penalty, e.g. `if info.get("collision"): r_orig -= 0.5`.
- **Phase 3:** Add a small time penalty per step or a bonus for finishing under target time.

The validator’s **final** score is still `flight_reward` (success + time). Keep your shaping terms small so that “reach goal quickly” remains the main objective.
