# RL/preprocessing.py
"""
Observation preprocessing for SAC training: RGB, depth, and state normalization.
Wraps the env so the policy sees channels-first, normalized inputs.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# State layout from MovingDroneAviary._computeObs:
# obs_12 (pos 3, rpy 3, vel 3, ang_vel 3) = 12
# + ACTION_BUFFER_SIZE * 5 (action buffer)
# + 1 (altitude)
# + 3 (search area vector)
# We replace yaw with sin(yaw), cos(yaw) so state becomes 13 + buf*5 + 1 + 3
STATE_BASE = 12
STATE_ALT_IDX = -4   # index of altitude in raw state (before search vec)
STATE_SEARCH_SLICE = slice(-3, None)


def _parse_state_layout(raw_state: np.ndarray) -> tuple[int, int]:
    """Infer action buffer length and total raw state dim from raw state."""
    n = raw_state.size
    # n = 12 + buf*5 + 1 + 3 => buf = (n - 16) // 5
    action_buf_len = (n - 16) // 5
    return action_buf_len, n


def _raw_state_to_normalized(
    raw: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-8,
    clip: float = 10.0,
) -> np.ndarray:
    """
    Convert raw state to normalized: replace yaw with sin/cos, normalize search vector,
    then (state - mean) / (std + eps) and clip. mean/std are for the transformed state.
    """
    action_buf_len, _ = _parse_state_layout(raw)
    pos = raw[0:3]
    roll, pitch, yaw = raw[3], raw[4], raw[5]
    vel = raw[6:9]
    ang_vel = raw[9:12]
    sin_yaw = np.float32(np.sin(yaw))
    cos_yaw = np.float32(np.cos(yaw))
    head = np.concatenate([
        pos,
        np.array([roll, pitch, sin_yaw, cos_yaw], dtype=np.float32),
        vel,
        ang_vel,
    ])
    action_part = raw[12 : 12 + action_buf_len * 5]
    alt = raw[12 + action_buf_len * 5 : 13 + action_buf_len * 5]
    g = raw[STATE_SEARCH_SLICE].copy()
    g_norm = np.linalg.norm(g)
    if g_norm > eps:
        g = g / (g_norm + eps)
    state = np.concatenate([head, action_part, alt, g]).astype(np.float32)
    std_safe = np.maximum(std, eps)
    out = np.clip((state - mean) / std_safe, -clip, clip).astype(np.float32)
    return out


def _update_running_state_stats(
    raw_state: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    count: float,
    alpha: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Welford-style update for mean and variance (single sample)."""
    count_new = count + 1.0
    delta = raw_state - mean
    mean_new = mean + delta / count_new
    delta2 = raw_state - mean_new
    var_new = var + np.dot(delta, delta2)
    # Optional: use EMA for stability in non-stationary setting
    mean_ema = (1 - alpha) * mean + alpha * raw_state
    var_ema = (1 - alpha) * var + alpha * np.maximum((raw_state - mean_ema) ** 2, 1e-8)
    return mean_ema, var_ema, count_new


class ObsPreprocessWrapper(gym.ObservationWrapper):
    """
    Preprocesses raw dict observation to normalized, channels-first tensors.
    Maintains running mean/var for state normalization.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        depth_near_emphasis: bool = True,
        state_eps: float = 1e-8,
        state_clip: float = 10.0,
        state_ema_alpha: float = 0.01,
    ):
        super().__init__(env)
        self._depth_near_emphasis = depth_near_emphasis
        self._state_eps = state_eps
        self._state_clip = state_clip
        self._state_ema_alpha = state_ema_alpha
        # Infer state layout from env
        os = env.observation_space
        if not isinstance(os, spaces.Dict) or "state" not in os.spaces:
            raise ValueError("ObsPreprocessWrapper expects Dict observation with 'state'")
        raw_state_dim = int(os.spaces["state"].shape[0])
        self._action_buf_len, _ = _parse_state_layout(np.zeros(raw_state_dim))
        # Normalized state: 13 + action_buf*5 + 1 + 3
        self._state_dim_norm = 13 + self._action_buf_len * 5 + 1 + 3
        self._state_mean = np.zeros(self._state_dim_norm, dtype=np.float32)
        self._state_var = np.ones(self._state_dim_norm, dtype=np.float32)
        self._state_count = 0.0
        # Build normalized state from raw for first-time shape
        self._raw_state_dim = raw_state_dim
        self.observation_space = spaces.Dict({
            "rgb": spaces.Box(0.0, 1.0, (4, 96, 96), dtype=np.float32),
            "depth": spaces.Box(0.0, 1.0, (1, 96, 96), dtype=np.float32),
            "state": spaces.Box(
                low=-state_clip,
                high=state_clip,
                shape=(self._state_dim_norm,),
                dtype=np.float32,
            ),
        })

    def _preprocess_raw_state(self, raw: np.ndarray) -> np.ndarray:
        """Build normalized state vector (same layout as _raw_state_to_normalized) for stats update."""
        action_buf_len = self._action_buf_len
        pos = raw[0:3]
        roll, pitch, yaw = raw[3], raw[4], raw[5]
        vel = raw[6:9]
        ang_vel = raw[9:12]
        sin_yaw = np.float32(np.sin(yaw))
        cos_yaw = np.float32(np.cos(yaw))
        head = np.concatenate([
            pos,
            np.array([roll, pitch, sin_yaw, cos_yaw], dtype=np.float32),
            vel,
            ang_vel,
        ])
        action_part = raw[12 : 12 + action_buf_len * 5]
        alt = raw[12 + action_buf_len * 5 : 13 + action_buf_len * 5]
        g = raw[STATE_SEARCH_SLICE].copy()
        g_norm = np.linalg.norm(g)
        if g_norm > self._state_eps:
            g = g / (g_norm + self._state_eps)
        return np.concatenate([head, action_part, alt, g]).astype(np.float32)

    def _update_state_stats(self, raw_state: np.ndarray) -> None:
        state_norm_layout = self._preprocess_raw_state(raw_state)
        self._state_count += 1.0
        alpha = self._state_ema_alpha
        self._state_mean = (1 - alpha) * self._state_mean + alpha * state_norm_layout
        self._state_var = (1 - alpha) * self._state_var + alpha * np.maximum(
            (state_norm_layout - self._state_mean) ** 2, 1e-8
        )

    def observation(self, obs: dict) -> dict:
        rgb = obs["rgb"]
        depth = obs["depth"]
        state_raw = obs["state"]

        # RGB: uint8 [0,255] -> float32 [0,1], channels-first
        rgb_f = np.asarray(rgb, dtype=np.float32) / 255.0
        rgb_f = np.transpose(rgb_f, (2, 0, 1))  # (H,W,C) -> (C,H,W)

        # Depth: clip, no NaN/inf, channels-first, optional near emphasis
        depth_f = np.asarray(depth, dtype=np.float32).reshape(-1)
        depth_f = np.nan_to_num(depth_f, nan=1.0, posinf=1.0, neginf=0.0)
        depth_f = np.clip(depth_f, 0.0, 1.0).reshape(96, 96, 1)
        if self._depth_near_emphasis:
            depth_f = 1.0 - depth_f
        depth_f = np.transpose(depth_f, (2, 0, 1))  # (1, 96, 96)

        # State: update running stats then normalize
        self._update_state_stats(state_raw)
        std = np.sqrt(np.maximum(self._state_var, 1e-8))
        state_norm = _raw_state_to_normalized(
            state_raw,
            self._state_mean,
            std,
            eps=self._state_eps,
            clip=self._state_clip,
        )

        return {
            "rgb": rgb_f,
            "depth": depth_f,
            "state": state_norm,
        }
