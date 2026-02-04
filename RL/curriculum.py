# RL/curriculum.py
"""
Curriculum task sampling for SAC: stage 0 (easy) -> stage 1 (medium) -> stage 2 (full).
Uses optional MapTask fields: goal_tol_override, noise_scale, horizon_override.
"""
from __future__ import annotations

import random
from typing import Optional, Callable

import gymnasium as gym

from swarm.protocol import MapTask
from swarm.validator.task_gen import (
    get_type_params,
    _random_start,
    _goal_from_start,
    _goal_from_origin,
)
from swarm.constants import (
    RANDOM_START,
    START_PLATFORM,
    START_PLATFORM_SURFACE_Z,
    START_PLATFORM_TAKEOFF_BUFFER,
    START_PLATFORM_RANDOMIZE,
    START_PLATFORM_MIN_Z,
    START_PLATFORM_MAX_Z,
)


# Stage config: (r_min, r_max), horizon_override, goal_tol_override, noise_scale, challenge_types
CURRICULUM_STAGES = {
    0: {
        "r_min": 3.0,
        "r_max": 5.0,
        "horizon_override": 90.0,
        "goal_tol_override": 0.8,
        "noise_scale": 0.3,
        "challenge_types": [3, 4],
    },
    1: {
        "r_min": 5.0,
        "r_max": 10.0,
        "horizon_override": 75.0,
        "goal_tol_override": 0.6,
        "noise_scale": 0.7,
        "challenge_types": [2, 3, 4],
    },
    2: {
        "r_min": 10.0,
        "r_max": 25.0,
        "horizon_override": None,
        "goal_tol_override": None,
        "noise_scale": 1.0,
        "challenge_types": [1, 2, 3, 4, 5],
    },
}


def sample_task(stage: int, sim_dt: float, seed: Optional[int] = None) -> MapTask:
    """
    Sample a MapTask for the given curriculum stage.
    Stage 0 = easy (short distance, low noise, large goal tol).
    Stage 1 = medium. Stage 2 = full difficulty (all challenge types, no overrides).
    """
    if seed is None:
        seed = random.randrange(2**32)
    stage = max(0, min(2, stage))
    config = CURRICULUM_STAGES[stage]
    rng = random.Random(seed)
    type_rng = random.Random(seed + 12345)
    chosen_type = type_rng.choice(config["challenge_types"])
    params = get_type_params(chosen_type).copy()
    params["r_min"] = config["r_min"]
    params["r_max"] = config["r_max"]
    if config.get("horizon_override") is not None:
        params["horizon"] = config["horizon_override"]

    if RANDOM_START:
        start = _random_start(rng, params)
        goal = _goal_from_start(rng, start, params)
    else:
        if START_PLATFORM:
            start_z = START_PLATFORM_SURFACE_Z + START_PLATFORM_TAKEOFF_BUFFER
        else:
            start_z = 1.5
        start = (0.0, 0.0, start_z)
        goal = _goal_from_origin(rng, params)

    return MapTask(
        map_seed=seed,
        start=start,
        goal=goal,
        sim_dt=sim_dt,
        horizon=params["horizon"],
        challenge_type=chosen_type,
        version="1",
        goal_tol_override=config.get("goal_tol_override"),
        noise_scale=config.get("noise_scale"),
        horizon_override=config.get("horizon_override"),
    )


def sample_task_with_mixing(
    stage: int,
    sim_dt: float,
    seed: Optional[int] = None,
    mix_easy_fraction: float = 0.15,
) -> MapTask:
    """
    Sample a task for the current stage; with probability mix_easy_fraction
    sample from an easier stage (min(stage, 1)) to reduce forgetting.
    """
    if seed is None:
        seed = random.randrange(2**32)
    rng = random.Random(seed)
    if rng.random() < mix_easy_fraction:
        stage = min(stage, 1)
        seed = seed + 7777
    return sample_task(stage, sim_dt, seed)


class CurriculumWrapper(gym.Wrapper):
    """
    Recreates the inner env on each reset so each episode uses a newly sampled task
    from the curriculum (via make_inner callback). Use with a callable that reads
    current stage and returns make_env(task).
    """

    def __init__(self, make_inner: Callable[[], gym.Env]):
        # Create first env so we have valid observation/action space
        self._make_inner = make_inner
        env = make_inner()
        super().__init__(env)

    def reset(self, **kwargs):
        self.env.close()
        self.env = self._make_inner()
        return self.env.reset(**kwargs)
