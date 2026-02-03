"""
Phase-specific reward functions for curriculum training.

Each function has the same signature and is used by RewardWrapper to compute
the per-step reward. Customize these for denser shaping or different objectives
per phase (e.g. phase 1: goal-reaching, phase 2: add obstacle-avoidance,
phase 3: full mix with time pressure).

Signature:
    prev_info: dict from previous step (distance_to_goal, success, etc.)
    info: dict from current step (same keys)
    r_orig: float, reward from the underlying env (increment of flight_reward)
    terminated: bool
    truncated: bool
    -> float: reward to use for this step
"""
from __future__ import annotations

from typing import Any


def phase1_reward(
    prev_info: dict[str, Any],
    info: dict[str, Any],
    r_orig: float,
    terminated: bool,
    truncated: bool,
) -> float:
    """
    Phase 1 (open only): use the same reward as the project default.

    The env (MovingDroneAviary) already computes the per-step reward as the
    increment of flight_reward (swarm/validator/reward.py): 0.5 * success_term
    + 0.5 * time_term, clamped to [0, 1]. We pass that through unchanged.
    Override this function to add e.g. distance-to-goal shaping.
    """
    # Same as default in project: env reward (increment of flight_reward)
    return r_orig


def phase2_reward(
    prev_info: dict[str, Any],
    info: dict[str, Any],
    r_orig: float,
    terminated: bool,
    truncated: bool,
) -> float:
    """
    Phase 2 (open + easy obstacles): reward for goal-reaching while avoiding obstacles.

    Default: use env reward unchanged. Override to add e.g. collision penalty
    or progress-based shaping.
    """
    return r_orig


def phase3_reward(
    prev_info: dict[str, Any],
    info: dict[str, Any],
    r_orig: float,
    terminated: bool,
    truncated: bool,
) -> float:
    """
    Phase 3 (full mix): reward for all challenge types, aligned with validator.

    Default: use env reward unchanged. Override to add time pressure or
    type-specific terms.
    """
    return r_orig
