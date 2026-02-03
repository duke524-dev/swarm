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

import numpy as np

from swarm.constants import SIM_DT, SPEED_LIMIT


# Phase 1 speed shaping: reward for moving toward goal faster, penalty for near-zero movement
SPEED_TOWARD_GOAL_SCALE = 0.03   # reward per m/s toward goal (e.g. 1 m/s -> +0.03/step)
LOW_SPEED_THRESHOLD_M = 0.0005   # treat progress below this as "standing still" (meters/step)
LOW_SPEED_PENALTY = 0.008        # penalty per step when not moving toward goal (encourages throttle)
# Alignment: bonus when flight direction matches goal direction (smaller angle -> higher bonus)
ALIGNMENT_BONUS_SCALE = 0.025    # reward per unit cos(angle); 1 = straight toward goal -> +0.025/step
MIN_SPEED_FOR_ALIGNMENT = 0.05   # only apply alignment bonus when speed > this (m/s), avoid div by zero


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
    - bonus for speed toward goal (encourages larger velocity commands)
    - alignment bonus: reward when flight direction matches goal direction (smaller angle -> more bonus)
    - penalty for very low movement (discourages hovering)
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
    r = 0.5 * progress

    # Alignment bonus: reward when velocity direction points toward goal (angle between goal vec and flight vec small)
    pos = info.get("drone_position")
    vel = info.get("drone_velocity")
    goal_pos = info.get("goal_position")
    if pos is not None and vel is not None and goal_pos is not None:
        pos = np.asarray(pos, dtype=np.float64)
        vel = np.asarray(vel, dtype=np.float64)
        goal_pos = np.asarray(goal_pos, dtype=np.float64)
        speed = float(np.linalg.norm(vel))
        if speed >= MIN_SPEED_FOR_ALIGNMENT:
            goal_vec = goal_pos - pos
            dist_to_goal = float(np.linalg.norm(goal_vec))
            if dist_to_goal > 1e-6:
                goal_dir = goal_vec / dist_to_goal
                flight_dir = vel / speed
                cos_angle = float(np.clip(np.dot(goal_dir, flight_dir), -1.0, 1.0))
                r += ALIGNMENT_BONUS_SCALE * max(0.0, cos_angle)

    # Speed toward goal (m/s): reward moving faster toward the goal
    speed_toward_goal = progress / SIM_DT
    if speed_toward_goal > 0:
        r += SPEED_TOWARD_GOAL_SCALE * min(speed_toward_goal, SPEED_LIMIT)
    # Penalty for standing still when not at goal (encourages policy to use throttle)
    if not info.get("success", False) and not info.get("collision", False):
        if progress < LOW_SPEED_THRESHOLD_M:
            r -= LOW_SPEED_PENALTY

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
