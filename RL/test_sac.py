#!/usr/bin/env python3
"""
Test SAC policy locally with the same preprocessing and action mapping as training.
Uses full-difficulty tasks (stage 2 / random_task) by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from swarm.constants import SIM_DT, SPEED_LIMIT
from swarm.utils.env_factory import make_env
from swarm.validator.reward import flight_reward
from swarm.validator.task_gen import random_task
from gym_pybullet_drones.utils.enums import ActionType

from RL.curriculum import sample_task
from RL.preprocessing import ObsPreprocessWrapper
from RL.reward_shaping import RewardShapingWrapper
from RL.action_wrapper import ActionMappingWrapper
from RL.config import REWARD_CONFIG, V_MAX_TRAINING, YAW_RATE_MAX_TRAINING


def build_wrapped_env_for_task(task):
    """Build env with same wrappers as training (no curriculum; single task)."""
    base = make_env(task, gui=False)
    base = ObsPreprocessWrapper(base)
    base = RewardShapingWrapper(base, REWARD_CONFIG)
    base = ActionMappingWrapper(
        base,
        v_max=V_MAX_TRAINING,
        yaw_rate_max=YAW_RATE_MAX_TRAINING,
    )
    return base


def run_episode(task, model, *, gui=False):
    env = build_wrapped_env_for_task(task)
    obs, _ = env.reset()
    t_sim = 0.0
    success = False
    while t_sim < task.horizon:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        t_sim += SIM_DT
        if terminated or truncated:
            success = info.get("success", False)
            base = env.env
            while hasattr(base, "env"):
                base = base.env
            success = success or getattr(base, "_success", False)
            break
    env.close()
    score = flight_reward(success=success, t=t_sim, horizon=task.horizon, task=task)
    return success, t_sim, score


def main():
    parser = argparse.ArgumentParser(description="Test SAC policy locally")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("swarm/submission_template/sac_policy.zip"),
        help="Path to SAC .zip",
    )
    parser.add_argument("--seed", type=int, default=1, help="Seed for task generation")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--stage", type=int, default=2, help="Curriculum stage (0-2); 2 = full difficulty")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    model = SAC.load(str(args.model), device="cpu")

    results = []
    for ep in range(args.episodes):
        if args.stage == 2:
            task = random_task(sim_dt=SIM_DT, seed=args.seed + ep)
        else:
            task = sample_task(args.stage, SIM_DT, args.seed + ep)
        success, t, score = run_episode(task, model)
        results.append((success, t, score))
        print(f"  Episode {ep+1}: success={success}, time={t:.2f}s, score={score:.3f}")

    if results:
        successes = sum(r[0] for r in results)
        times = [r[1] for r in results]
        scores = [r[2] for r in results]
        print("----------------------------------------------------")
        print(f"Success rate: {successes}/{len(results)}")
        print(f"Avg time:     {np.mean(times):.2f} s")
        print(f"Avg score:    {np.mean(scores):.3f}")
        print("----------------------------------------------------")


if __name__ == "__main__":
    main()
