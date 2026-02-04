# RL/action_wrapper.py
"""
Maps policy output (e.g. [-1,1]^5) to env action space: direction, speed, yaw rate.
Output stays in env.action_space Box for compatibility with validator clipping.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym


class ActionMappingWrapper(gym.ActionWrapper):
    """
    Maps policy output (vx, vy, vz, speed, yaw) in [-1,1] to physical velocity + yaw rate,
    then back to the Box range expected by the env (so validator-style clipping still applies).
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        v_max: float = 2.0,
        yaw_rate_max: float = 1.5,
        v_z_up_max: float | None = None,
        v_z_down_max: float | None = None,
        eps: float = 1e-8,
    ):
        super().__init__(env)
        self._v_max = v_max
        self._yaw_rate_max = yaw_rate_max
        self._v_z_up_max = v_z_up_max if v_z_up_max is not None else v_max
        self._v_z_down_max = v_z_down_max if v_z_down_max is not None else v_max
        self._eps = eps
        # Env expects action in [low, high]; we output in same Box, scaled to physical then back
        low = np.asarray(env.action_space.low, dtype=np.float32).flatten()
        high = np.asarray(env.action_space.high, dtype=np.float32).flatten()
        self._low = low
        self._high = high
        # Typically [-1,1] for all 5; we'll map our v_cmd and yaw_rate back into that range
        self._action_dim = int(low.size)

    def action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 5:
            action = np.resize(action, 5)
        dir_raw = action[0:3]
        speed_raw = action[3]
        yaw_raw = action[4]

        n = np.linalg.norm(dir_raw) + self._eps
        if n < 1e-6:
            dir_vec = np.zeros(3, dtype=np.float32)
            speed = 0.0
        else:
            dir_vec = dir_raw / n
            # speed_raw in [-1,1] -> [0, 1] then * v_max
            speed = np.clip((speed_raw + 1.0) / 2.0, 0.0, 1.0) * self._v_max

        v_cmd = dir_vec * speed
        v_xy_norm = np.linalg.norm(v_cmd[0:2]) + self._eps
        if v_xy_norm > self._v_max:
            v_cmd[0] *= self._v_max / v_xy_norm
            v_cmd[1] *= self._v_max / v_xy_norm
        v_cmd[2] = np.clip(v_cmd[2], -self._v_z_down_max, self._v_z_up_max)

        yaw_rate = np.clip(yaw_raw, -1.0, 1.0) * self._yaw_rate_max

        # Map back to env Box: assume env expects [vx, vy, vz, speed, yaw] in [-1,1] or [0,1]
        # Validator uses action_space.low/high and then scales by SPEED_LIMIT. So we output
        # normalized values that represent the same physical command.
        scale = self._v_max + self._eps
        out = np.array([
            v_cmd[0] / scale,
            v_cmd[1] / scale,
            v_cmd[2] / scale,
            speed / self._v_max,
            yaw_rate / self._yaw_rate_max,
        ], dtype=np.float32)
        out = np.clip(out, self._low[:5], self._high[:5])
        if self._action_dim > 5:
            out = np.resize(out, self._action_dim)
        return out
