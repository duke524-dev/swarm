# RL/config.py
"""Training config: reward shaping and curriculum parameters."""

# Reward shaping (RewardShapingWrapper)
REWARD_CONFIG = {
    "R_goal": 200.0,
    "R_crash": 200.0,
    "R_timeout": 50.0,
    "k_prog": 0.5,
    "k_obs": 2.0,
    "d_safe": 0.2,
    "d_gate": 0.3,
    "k_align": 0.2,
    "k_time": 0.005,
    "k_energy": 0.01,
    "k_smooth": 0.05,
}

# Curriculum promotion
EVAL_EPISODES_INTERVAL = 50
PROMOTION_SUCCESS_RATE_THRESHOLD = 0.4
PROMOTION_CONSECUTIVE_M = 3
MIX_EASY_FRACTION = 0.15

# Action mapping (ActionMappingWrapper)
V_MAX_TRAINING = 2.0
YAW_RATE_MAX_TRAINING = 1.5
