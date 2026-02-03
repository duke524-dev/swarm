"""
Wrapper that creates a new task (and env) on each reset(), so training sees
many different scenarios instead of a fixed set per env.
Also provides RewardWrapper to plug in phase-specific reward functions.
"""
from __future__ import annotations

from typing import Any, Callable

from gymnasium import Wrapper

# Type for task factory: () -> MapTask
TaskFactory = Callable[[], Any]

# Reward fn: (prev_info, info, r_orig, terminated, truncated) -> float
RewardFn = Callable[[dict, dict, float, bool, bool], float]


class RewardWrapper(Wrapper):
    """
    Replaces the env's step reward with the result of reward_fn.
    Use this to plug in phase-specific reward logic without changing the core env.
    """

    def __init__(self, env, reward_fn: RewardFn):
        super().__init__(env)
        self._reward_fn = reward_fn
        self._prev_info: dict[str, Any] = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._prev_info = {}
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        info = info if info is not None else {}
        r_new = self._reward_fn(self._prev_info, info, r, terminated, truncated)
        self._prev_info = info.copy()
        return obs, r_new, terminated, truncated, info


class ResettableTaskWrapper(Wrapper):
    """
    Wraps an env so that each reset() uses a new task from task_factory().
    Creates a new underlying env each episode so the policy sees diverse tasks.
    """

    def __init__(self, task_factory: TaskFactory, make_env_fn: Callable):
        """
        Args:
            task_factory: Callable that returns a MapTask (no args).
            make_env_fn: Callable (task) -> env. Used to build env from task.
        """
        task0 = task_factory()
        env0 = make_env_fn(task0)
        super().__init__(env0)
        self._task_factory = task_factory
        self._make_env_fn = make_env_fn
        self._current_env = env0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        task = self._task_factory()
        if self._current_env is not None:
            try:
                self._current_env.close()
            except Exception:
                pass
        self._current_env = self._make_env_fn(task)
        self.env = self._current_env
        return self._current_env.reset(seed=task.map_seed, options=options)
