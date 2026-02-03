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
    Phase 1 (open only): reward = how close and how fast we go to the goal.

    There are no obstacles in type‑4 tasks, so we shape reward purely by
    progress toward the goal and time:

    - positive when distance_to_goal decreases (moving closer)
    - small time penalty each step (encourages shorter paths)
    - big bonus on success
    - optional penalty on collision (should be rare in open env)
    """
    # Current and previous distances to goal (meters)
    dist_now = float(info.get("distance_to_goal", 0.0))
    dist_prev = float(prev_info.get("distance_to_goal", dist_now))

    # Progress toward goal: positive if we moved closer this step
    progress = dist_prev - dist_now  # meters closer

    # Scale progress so typical per-step reward stays moderate.
    # You can tune 0.5 up/down based on learning behavior.
    r = 0.5 * progress

    # Small time penalty every step to encourage faster arrival.
    r -= 0.001

    # Big bonus when we actually reach the goal.
    if info.get("success", False):
        r += 2.0

    # Optional penalty if something goes wrong even in open env.
    if info.get("collision", False):
        r -= 1.0

    return r


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
