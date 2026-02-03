#!/usr/bin/env python3
"""
Evaluate a trained policy the same way validators do: same task generation,
same scoring (flight_reward), and multiple episodes to get success rate and
mean score like on the subnet.

Usage:
  # Evaluate PPO .zip over 20 episodes (default), mixed challenge types
  python RL/eval_validator_style.py --model swarm/submission_template/phase1_open_only.zip

  # More episodes, specific seed range (reproducible)
  python RL/eval_validator_style.py --model path/to/policy.zip -n 50 --seed-start 1000

  # Same as validator time-window seed (requires SWARM_VALIDATOR_SECRET_KEY in env)
  python RL/eval_validator_style.py --model path/to/policy.zip --sync-seed -n 10
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from swarm.constants import SIM_DT
from swarm.validator.task_gen import random_task, random_task_with_type

from RL.test_RL import _run_episode_speed_limit


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model validator-style: same tasks, same scoring, multiple episodes."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("swarm/submission_template/phase1_open_only.zip"),
        help="Path to the Stable-Baselines3 .zip policy file.",
    )
    parser.add_argument(
        "-n",
        "--num-episodes",
        type=int,
        default=20,
        help="Number of episodes to run (default: 20).",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="First seed for task generation (seeds seed_start .. seed_start+num_episodes-1).",
    )
    parser.add_argument(
        "--sync-seed",
        action="store_true",
        help="Use validator-style synchronized seed from current time window (needs SWARM_VALIDATOR_SECRET_KEY).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for loading the model (default: cpu).",
    )
    parser.add_argument(
        "--challenge-type",
        type=int,
        default=None,
        choices=[1, 2, 3, 4, 5],
        help="Fix task type: 1=city, 2=high obstacles, 3=easy, 4=open, 5=moving platform. Omit for mixed.",
    )
    parser.add_argument(
        "--speed-profile",
        action="store_true",
        help="Report speed change over time: per-episode quartiles (mean speed in 1st/2nd/3rd/4th quarter of steps) and optional CSV.",
    )
    parser.add_argument(
        "--speed-profile-csv",
        type=Path,
        default=None,
        metavar="FILE",
        help="With --speed-profile: write first episode (t_sec, speed_m_s) to this CSV for plotting.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    from stable_baselines3 import PPO

    # Load model once
    model = PPO.load(str(args.model), device=args.device)

    if args.sync_seed:
        secret = os.environ.get("SWARM_VALIDATOR_SECRET_KEY")
        if not secret:
            print("--sync-seed requires SWARM_VALIDATOR_SECRET_KEY in environment.", file=sys.stderr)
            sys.exit(1)
        from swarm.validator.seed_manager import SynchronizedSeedManager
        from swarm.constants import SEED_WINDOW_MINUTES
        seed_manager = SynchronizedSeedManager(secret, window_minutes=SEED_WINDOW_MINUTES)
        seed, win_start, win_end = seed_manager.generate_seed()
        seeds = [seed + i for i in range(args.num_episodes)]
        print(f"Using synchronized seed window {win_start}–{win_end} UTC (base seed={seed})")
    else:
        seeds = [args.seed_start + i for i in range(args.num_episodes)]

    challenge_type = args.challenge_type
    speed_profile = getattr(args, "speed_profile", False)
    speed_profile_csv = getattr(args, "speed_profile_csv", None)
    results = []
    all_speeds_by_episode = []
    for i, seed in enumerate(seeds):
        if challenge_type is not None:
            task = random_task_with_type(sim_dt=SIM_DT, seed=seed, challenge_type=challenge_type)
        else:
            task = random_task(sim_dt=SIM_DT, seed=seed)
        result, avg_speed, speeds = _run_episode_speed_limit(task=task, uid=0, model=model, gui=False)
        results.append((result, task.challenge_type, avg_speed))
        if speed_profile and speeds:
            all_speeds_by_episode.append((i, speeds))
        print(f"  Episode {i+1}/{len(seeds)} seed={seed} type={task.challenge_type} "
              f"success={result.success} time={result.time_sec:.2f}s score={result.score:.3f} avg_speed={avg_speed:.3f} m/s")

    # Validator-like summary
    n = len(results)
    successes = sum(1 for r, _, _ in results if r.success)
    scores = [r.score for r, _, _ in results]
    times = [r.time_sec for r, _, _ in results if r.success]
    mean_score = sum(scores) / n if n else 0.0
    mean_time_success = sum(times) / len(times) if times else 0.0

    # Speed profile: how speed changes over time within episodes
    if speed_profile and all_speeds_by_episode:
        import numpy as np
        print()
        print("SPEED PROFILE (mean speed by quarter of episode)")
        print("-" * 56)
        for ep_idx, speeds in all_speeds_by_episode:
            arr = np.array(speeds, dtype=np.float64)
            n = len(arr)
            if n == 0:
                continue
            q = n // 4
            q1 = arr[:q].mean() if q else 0.0
            q2 = arr[q : 2 * q].mean() if q else 0.0
            q3 = arr[2 * q : 3 * q].mean() if q else 0.0
            q4 = arr[3 * q :].mean() if q else 0.0
            print(f"  Episode {ep_idx + 1}: Q1={q1:.3f} Q2={q2:.3f} Q3={q3:.3f} Q4={q4:.3f} m/s  (min={arr.min():.3f} max={arr.max():.3f})")
        # Write first episode to CSV if requested
        if speed_profile_csv and all_speeds_by_episode:
            ep_idx, speeds = all_speeds_by_episode[0]
            speed_profile_csv.parent.mkdir(parents=True, exist_ok=True)
            with open(speed_profile_csv, "w") as f:
                f.write("t_sec,speed_m_s\n")
                for step, s in enumerate(speeds):
                    f.write(f"{step * SIM_DT:.4f},{s:.4f}\n")
            print(f"  First episode speeds written to {speed_profile_csv}")

    print()
    print("=" * 56)
    print("VALIDATOR-STYLE SUMMARY (same scoring & task distribution)")
    print("=" * 56)
    print(f"Episodes       : {n}")
    print(f"Success rate   : {successes}/{n} ({100.0 * successes / n:.1f}%)")
    print(f"Mean score     : {mean_score:.3f}")
    print(f"Mean time (ok) : {mean_time_success:.2f} s" + (" (no successes)" if not times else ""))
    print("=" * 56)
    print()
    print("To test the exact submission validators run (RPC agent in Docker):")
    print("  1. Put your policy in swarm/submission_template and ensure drone_agent loads it.")
    print("  2. python tests/test_rpc.py --folder swarm/submission_template --seed 42")
    print("  3. Or run the validator locally with Docker to match production.")


if __name__ == "__main__":
    main()
