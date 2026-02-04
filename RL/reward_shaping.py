# RL/reward_shaping.py
"""
Dense reward shaping for SAC training: terminal rewards + progress, obstacle penalty,
gated alignment, time, energy, smoothness.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym


def _norm_xy(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v) + eps
    return v / n if n > eps else np.zeros_like(v)


class RewardShapingWrapper(gym.Wrapper):
    """
    Replaces env reward with shaped terminal + dense reward.
    Expects to wrap an env that returns dict obs (e.g. after ObsPreprocessWrapper)
    and exposes unwrapped._getDroneStateVector, unwrapped.task, unwrapped._search_area_center.
    """

    def __init__(self, env: gym.Env, config: dict | None = None):
        super().__init__(env)
        cfg = config or {}
        self.R_goal = float(cfg.get("R_goal", 200.0))
        self.R_crash = float(cfg.get("R_crash", 200.0))
        self.R_timeout = float(cfg.get("R_timeout", 50.0))
        self.k_prog = float(cfg.get("k_prog", 0.5))
        self.k_obs = float(cfg.get("k_obs", 2.0))
        self.d_safe = float(cfg.get("d_safe", 0.2))
        self.d_gate = float(cfg.get("d_gate", 0.3))
        self.k_align = float(cfg.get("k_align", 0.2))
        self.k_time = float(cfg.get("k_time", 0.005))
        self.k_energy = float(cfg.get("k_energy", 0.01))
        self.k_smooth = float(cfg.get("k_smooth", 0.05))
        self._prev_action = np.zeros(5, dtype=np.float32)
        self._eps = 1e-8

    def reset(self, **kwargs):
        self._prev_action = np.zeros(5, dtype=np.float32)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != 5:
            action = np.resize(action, 5)

        base = self.env
        while hasattr(base, "env"):
            base = base.env
        state_vec = base._getDroneStateVector(0)
        pos = state_vec[0:3]
        v = state_vec[10:13]
        v_xy = v[0:2]
        g_vec = np.asarray(base._search_area_center, dtype=np.float64) - pos
        g_xy = g_vec[0:2]

        # Terminal
        if terminated or truncated:
            if info.get("success"):
                r = self.R_goal
            elif info.get("collision"):
                r = -self.R_crash
            elif truncated:
                r = -self.R_timeout
            else:
                r = 0.0
            self._prev_action = action
            return obs, float(r), terminated, truncated, info

        # Dense
        g_xy_norm = _norm_xy(g_xy, self._eps)
        r_prog = self.k_prog * float(np.dot(v_xy, g_xy_norm))

        # Depth: obs["depth"] is (1,96,96) after preprocessing (near emphasis: high = near)
        depth = obs.get("depth")
        if depth is not None:
            d_flat = np.asarray(depth).flatten()
            d_near = float(np.percentile(d_flat, 95))
            r_obs = -self.k_obs * max(0.0, d_near - self.d_safe)
            if d_near <= self.d_gate:
                g_3 = _norm_xy(g_vec, self._eps)
                r_align = self.k_align * float(np.dot(v, g_3))
            else:
                r_align = 0.0
        else:
            r_obs = 0.0
            r_align = self.k_align * float(np.dot(v_xy, g_xy_norm)) if np.linalg.norm(g_xy) > self._eps else 0.0

        r_time = -self.k_time
        r_energy = -self.k_energy * (float(action[3]) ** 2)
        r_smooth = -self.k_smooth * float(np.sum((action - self._prev_action) ** 2))

        shaped = r_prog + r_align + r_obs + r_time + r_energy + r_smooth
        self._prev_action = action
        return obs, float(shaped), terminated, truncated, info
