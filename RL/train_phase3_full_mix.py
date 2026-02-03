#!/usr/bin/env python3
"""Phase 3: Full mix of all challenge types (like validator). Loads Phase 2 by default."""
import argparse
import random
import sys
from pathlib import Path

# Ensure project root is on sys.path so we use the local 'swarm' package
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

try:
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
except ImportError as e:
    print("Missing dependency for RL training.", file=sys.stderr)
    print("Install with: pip install stable-baselines3 torch gymnasium", file=sys.stderr)
    print("From repo root: pip install -r requirements.txt", file=sys.stderr)
    raise SystemExit(1) from e

from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task_with_type
from swarm.constants import SIM_DT

from RL.env_wrapper import ResettableTaskWrapper, RewardWrapper
from RL.rewards import phase3_reward
from RL.callbacks import EvalLoggingCallbackFromVecEnv

DEFAULT_PREVIOUS = "swarm/submission_template/phase2_easy_obstacles.zip"


def make_env_no_gui(task):
    return make_env(task, gui=False)


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Full mix")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--load", type=str, default=DEFAULT_PREVIOUS,
                        help="Load previous phase (Phase 2) model to enhance. Set to '' to train from scratch.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--algo", type=str, default="ppo", choices=("ppo", "a2c"))
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--checkpoint-dir", type=str, default="logs/phase3_checkpoints")
    parser.add_argument("--tensorboard-log", type=str, default="logs/phase3")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed-start", type=int, default=100000)
    parser.add_argument("--output", type=str, default="phase3_full_mix.zip")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        try:
            import torch
            torch.manual_seed(args.seed)
        except Exception:
            pass

    def task_factory():
        return random_task_with_type(sim_dt=SIM_DT, seed=random.randint(0, 2**31 - 1), challenge_type=None)

    def eval_task_factory(seed: int):
        return random_task_with_type(sim_dt=SIM_DT, seed=seed, challenge_type=None)

    def make_wrapped():
        inner = ResettableTaskWrapper(task_factory, make_env_no_gui)
        return RewardWrapper(inner, phase3_reward)

    env = DummyVecEnv([make_wrapped for _ in range(args.n_envs)])

    algo_kw = dict(
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        device=args.device,
        tensorboard_log=args.tensorboard_log,
        verbose=1,
    )
    if args.algo == "ppo":
        algo_kw["clip_range"] = args.clip_range

    load_path = args.resume if args.resume and Path(args.resume).exists() else (args.load if args.load and Path(args.load).exists() else None)
    if load_path:
        if args.algo == "ppo":
            model = PPO.load(load_path, env=env, **algo_kw)
        else:
            model = A2C.load(load_path, env=env, **algo_kw)
        print(f"Loaded previous phase model to enhance: {load_path}")
    else:
        if args.load and args.load != "":
            print(f"Warning: --load {args.load} not found, training from scratch")
        if args.algo == "ppo":
            model = PPO("MultiInputPolicy", env, **algo_kw)
        else:
            model = A2C("MultiInputPolicy", env, **algo_kw)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    callbacks = [
        CheckpointCallback(save_freq=args.save_freq, save_path=args.checkpoint_dir, name_prefix="phase3"),
        EvalLoggingCallbackFromVecEnv(
            task_factory=eval_task_factory,
            make_env_fn=make_env_no_gui,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            eval_seed_start=args.eval_seed_start,
            verbose=1,
        ),
    ]
    model.learn(args.timesteps, callback=CallbackList(callbacks))

    out_dir = Path(__file__).parent.parent / "swarm" / "submission_template"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / args.output))
    env.close()

    print(f"Saved: {out_dir / args.output}")
    print("Use this model in drone_agent.py and create submission.zip for the miner.")


if __name__ == "__main__":
    main()
