"""
Training callbacks: evaluation (success rate, time), checkpointing.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv


def _run_eval_episodes(
    model,
    task_factory: Callable,
    make_env_fn: Callable,
    n_episodes: int,
    eval_seed_start: int,
    deterministic: bool = True,
) -> tuple[float, float, float]:
    """Run n_episodes with fixed seeds, return (success_rate, mean_time, mean_reward)."""
    successes = []
    times = []
    rewards_sum = []
    for i in range(n_episodes):
        seed = eval_seed_start + i
        task = task_factory(seed)
        env = make_env_fn(task)
        obs, _ = env.reset(seed=task.map_seed)
        total_r = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            action = np.asarray(action).flatten()
            if hasattr(env, "action_space"):
                lo, hi = env.action_space.low.flatten(), env.action_space.high.flatten()
                action = np.clip(action, lo, hi)
            # Env expects batch action (1, 5)
            obs, r, terminated, truncated, info = env.step(action[None, :])
            total_r += r
            done = terminated or truncated
        env.close()
        successes.append(1.0 if info.get("success", False) else 0.0)
        times.append(float(info.get("t_to_goal") or 0.0))
        rewards_sum.append(total_r)
    return (
        np.mean(successes),
        np.mean(times),
        np.mean(rewards_sum),
    )


class EvalLoggingCallback(BaseCallback):
    """
    Runs evaluation episodes on fixed seeds every eval_freq steps and logs
    success_rate, mean_time, mean_reward to TensorBoard.
    """

    def __init__(
        self,
        task_factory: Callable[[int], Any],  # (eval_seed) -> MapTask
        make_env_fn: Callable,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        eval_seed_start: int = 100000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.task_factory = task_factory
        self.make_env_fn = make_env_fn
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed_start = eval_seed_start

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0 or self.n_calls == 0:
            return True
        # Build task_factory that takes seed index and returns task
        def factory(seed: int):
            return self.task_factory(seed)

        success_rate, mean_time, mean_reward = _run_eval_episodes(
            self.model,
            factory,
            self.make_env_fn,
            self.n_eval_episodes,
            self.eval_seed_start,
        )
        if self.logger is not None:
            self.logger.record("eval/success_rate", success_rate)
            self.logger.record("eval/mean_time_sec", mean_time)
            self.logger.record("eval/mean_reward", mean_reward)
        if self.verbose:
            print(
                f"[Eval] step={self.n_calls} success_rate={success_rate:.3f} "
                f"mean_time={mean_time:.2f}s mean_reward={mean_reward:.3f}"
            )
        return True


class EvalLoggingCallbackFromVecEnv(BaseCallback):
    """
    Same as EvalLoggingCallback but task_factory is (seed) -> MapTask
    and we use the same signature for _run_eval_episodes.
    """

    def __init__(
        self,
        task_factory: Callable[[int], Any],  # (seed) -> MapTask
        make_env_fn: Callable,
        eval_freq: int = 10000,
        n_eval_episodes: int = 10,
        eval_seed_start: int = 100000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.task_factory = task_factory
        self.make_env_fn = make_env_fn
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed_start = eval_seed_start

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0 or self.n_calls == 0:
            return True

        def factory(seed: int):
            return self.task_factory(seed)

        success_rate, mean_time, mean_reward = _run_eval_episodes(
            self.model,
            factory,
            self.make_env_fn,
            self.n_eval_episodes,
            self.eval_seed_start,
        )
        if self.logger is not None:
            self.logger.record("eval/success_rate", success_rate)
            self.logger.record("eval/mean_time_sec", mean_time)
            self.logger.record("eval/mean_reward", mean_reward)
        if self.verbose:
            print(
                f"[Eval] step={self.n_calls} success_rate={success_rate:.3f} "
                f"mean_time={mean_time:.2f}s mean_reward={mean_reward:.3f}"
            )
        return True
