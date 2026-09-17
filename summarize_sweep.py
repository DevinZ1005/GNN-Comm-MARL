#!/usr/bin/env python3
"""
Summarize real sweep results across all 5 seeds for Dense, Attention, and Random modes.
Parses logs/real_*.log, computes mean +/- std, and formats results.

Usage:
    ./venv314/bin/python summarize_sweep.py
"""

import glob
import re
import numpy as np

ITER_LINE_RE = re.compile(
    r"^Iter\s+(\d+)\s*\|\s*Reward Mean:\s*([\d\.\-]+)\s*\|\s*Ep Len:\s*([\d\.\-]+)\s*\|\s*Policy Loss:\s*([\d\.\-]+)\s*\|\s*Drop Frac:\s*([\d\.\-]+)"
)

def parse_log_file(filepath):
    last_valid = None
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            m = ITER_LINE_RE.match(line.strip())
            if m:
                last_valid = {
                    "iter": int(m.group(1)),
                    "reward_mean": float(m.group(2)),
                    "ep_len": float(m.group(3)),
                    "policy_loss": float(m.group(4)),
                    "drop_frac": float(m.group(5)),
                }
    return last_valid

def main():
    print("HISTORICAL TRAINING LOGS ONLY: not held-out evaluation or valid cross-version evidence.")
    print("Use compare_evaluations.py for version 2 matched evaluation results.")
    modes = ["dense", "attn", "rand"]
    seeds = [0, 1, 2, 3, 4]
    
    results = {}
    
    print("=" * 80)
    print(f"{'Condition':<12} | {'Seed':<5} | {'Iter':<6} | {'Reward Mean':<12} | {'Drop Frac':<10} | {'Policy Loss':<12}")
    print("-" * 80)

    for mode in modes:
        results[mode] = []
        for seed in seeds:
            log_path = f"logs/real_{mode}_s{seed}.log"
            data = parse_log_file(log_path) if glob.glob(log_path) else None
            if data and data["iter"] < 300:
                print(f"{mode:<12} | {seed:<5} | {data['iter']:<6} | incomplete; excluded from summary")
                continue
            if data:
                results[mode].append(data)
                print(f"{mode:<12} | {seed:<5} | {data['iter']:<6} | {data['reward_mean']:<12.2f} | {data['drop_frac']:<10.4f} | {data['policy_loss']:<12.4f}")
            else:
                print(f"{mode:<12} | {seed:<5} | {'N/A':<6} | {'(pending)':<12} | {'--':<10} | {'--':<12}")

    print("=" * 80)
    print("\n--- Summary Statistics (Mean ± Std) ---")
    print("-" * 80)
    print(f"{'Condition':<12} | {'Seeds Done':<10} | {'Mean Reward (±std)':<25} | {'Mean Drop Frac':<15}")
    print("-" * 80)

    for mode in modes:
        rewards = [d["reward_mean"] for d in results[mode]]
        drop_fracs = [d["drop_frac"] for d in results[mode]]
        n_done = len(rewards)
        if n_done > 0:
            mean_r = np.mean(rewards)
            std_r = np.std(rewards, ddof=1) if n_done > 1 else 0.0
            mean_df = np.mean(drop_fracs)
            print(f"{mode:<12} | {f'{n_done}/5':<10} | {f'{mean_r:.2f} ± {std_r:.2f}':<25} | {f'{mean_df:.4f}':<15}")
        else:
            print(f"{mode:<12} | 0/5        | {'(no data)':<25} | {'--':<15}")
    print("=" * 80)

if __name__ == "__main__":
    main()
