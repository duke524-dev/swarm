#!/usr/bin/env python3
"""
Train SAC for drone navigation with curriculum, observation preprocessing,
reward shaping, and action mapping.

Usage:
  python RL/train_sac.py --timesteps 100000 --save_path swarm/submission_template/sac_policy.zip
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from swarm.utils.env_factory import make_env
from swarm.constants import SIM_DT

from RL.curriculum import sample_task_with_mixing, CurriculumWrapper
from RL.preprocessing import ObsPreprocessWrapper
from RL.reward_shaping import RewardShapingWrapper
from RL.action_wrapper import ActionMappingWrapper
from RL.config import (
    REWARD_CONFIG,
    EVAL_EPISODES_INTERVAL,
    PROMOTION_SUCCESS_RATE_THRESHOLD,
    PROMOTION_CONSECUTIVE_M,
    MIX_EASY_FRACTION,
    V_MAX_TRAINING,
    YAW_RATE_MAX_TRAINING,
)


def build_wrapped_env(current_stage: list):
    """Build env with curriculum + preprocessing + reward shaping + action mapping."""

    def make_inner():
        task = sample_task_with_mixing(
            current_stage[0], SIM_DT, random.randrange(2**32), mix_easy_fraction=MIX_EASY_FRACTION
        )
        return make_env(task, gui=False)

    base = CurriculumWrapper(make_inner)
    base = ObsPreprocessWrapper(base)
    base = RewardShapingWrapper(base, REWARD_CONFIG)
    base = ActionMappingWrapper(
        base,
        v_max=V_MAX_TRAINING,
        yaw_rate_max=YAW_RATE_MAX_TRAINING,
    )
    return Monitor(base)


def evaluate_success_rate(model, current_stage: list, n_episodes: int = 50) -> float:
    """Run n_episodes with current stage and return fraction of successes."""
    env = build_wrapped_env(current_stage)
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        # Unwrap to get base env success flag (set in _check_collision)
        base = env.env
        while hasattr(base, "env"):
            base = base.env
        if getattr(base, "_success", False) or info.get("success", False):
            successes += 1
    env.close()
    return successes / n_episodes


def main():
    parser = argparse.ArgumentParser(description="Train SAC with curriculum for Swarm subnet")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--save_path", type=Path, default=None, help="Path to save SAC model .zip")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--stage", type=int, default=0, help="Starting curriculum stage (0-2)")
    parser.add_argument("--chunk", type=int, default=20_000, help="Timesteps per chunk before eval/promotion")
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = Path(__file__).parent.parent / "swarm" / "submission_template" / "sac_policy.zip"
    if args.seed is not None:
        random.seed(args.seed)

    current_stage = [args.stage]
    env = DummyVecEnv([lambda: build_wrapped_env(current_stage)])

    model = SAC(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=None,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[256, 256])),
    )

    remaining = args.timesteps
    consecutive_good = 0
    while remaining > 0 and current_stage[0] < 2:
        chunk = min(args.chunk, remaining)
        model.learn(total_timesteps=chunk)
        remaining -= chunk
        rate = evaluate_success_rate(model, current_stage, n_episodes=EVAL_EPISODES_INTERVAL)
        print(f"[Curriculum] Stage {current_stage[0]} success rate: {rate:.2%}")
        if rate >= PROMOTION_SUCCESS_RATE_THRESHOLD:
            consecutive_good += 1
            if consecutive_good >= PROMOTION_CONSECUTIVE_M:
                current_stage[0] = min(2, current_stage[0] + 1)
                print(f"[Curriculum] Promoted to stage {current_stage[0]}")
                consecutive_good = 0
        else:
            consecutive_good = 0

    if remaining > 0:
        model.learn(total_timesteps=remaining)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.save_path))
    print(f"\n✅ SAC model saved to: {args.save_path}")
    print(f"   Final curriculum stage: {current_stage[0]}")
    env.close()


if __name__ == "__main__":
    main()
