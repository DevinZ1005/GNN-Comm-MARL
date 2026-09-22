"""Print a matched experiment schedule; --execute runs jobs sequentially.

Existing outputs are never overwritten. Use a fresh root for a new experiment.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--backend", required=True, choices=["kinematic", "pybullet"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--budgets", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--num-robots", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--comm-radius", type=float, default=1.8)
    parser.add_argument("--goal-observers", type=int, default=-1)
    parser.add_argument("--goal-spawn-mode", choices=["coupled", "independent"], default="coupled")
    parser.add_argument("--observation-frame", choices=["world", "robot"], default="world")
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--vf-clip-param", type=float, default=10.0)
    parser.add_argument("--conditions", nargs="+", choices=["dense", "no_comm", "attention",
                        "random", "distance", "gumbel"],
                        default=["dense", "no_comm", "attention", "random", "distance", "gumbel"])
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--train-batch-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gnn-num-layers", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--development-episodes", type=int, default=0,
                        help="Evaluate each final export after training; zero disables this.")
    parser.add_argument("--development-seed-start", type=int, default=80000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if any(len(set(values)) != len(values) for values in (args.seeds, args.budgets, args.conditions)):
        parser.error("duplicate seeds, budgets or conditions")
    if args.goal_observers > 0 and args.goal_observers < args.num_robots and args.goal_spawn_mode != "independent":
        parser.error("private-goal sweeps require --goal-spawn-mode independent")
    if (min(args.iterations, args.max_steps, args.epochs, args.gnn_num_layers,
            args.eval_interval, args.eval_episodes) < 1 or args.train_batch_size < 2
            or args.workers < 0 or args.development_episodes < 0):
        parser.error("invalid training or evaluation budget")
    if not math.isfinite(args.reward_scale) or args.reward_scale <= 0:
        parser.error("reward-scale must be finite and positive")
    if not math.isfinite(args.vf_clip_param) or args.vf_clip_param <= 0:
        parser.error("vf-clip-param must be finite and positive")
    if any(k < 1 or k >= args.num_robots for k in args.budgets):
        parser.error("budgets must be between 1 and N-1")
    conditions = [(name, flags) for name, flags in [("dense", []), ("no_comm", ["--no-comm"])]
                  if name in args.conditions]
    for mode in ("attention", "random", "distance", "gumbel"):
        if mode in args.conditions:
            conditions.extend((f"{mode}_k{k}", ["--topk-mode", mode, "--top-k", str(k)]) for k in args.budgets)
    jobs = []
    for seed in args.seeds:
        for name, flags in conditions:
            run = Path(args.root).resolve() / f"{name}_s{seed}"
            command = [sys.executable, "-u", str(Path(__file__).with_name("train.py")),
                       "--backend", args.backend, "--num-robots", str(args.num_robots),
                       "--comm-radius", str(args.comm_radius), "--goal-observers", str(args.goal_observers),
                       "--seed", str(seed),
                       "--num-workers", str(args.workers), "--max-iterations", str(args.iterations),
                       "--checkpoint-dir", str(run)] + flags
            for key in ("goal_spawn_mode", "observation_frame", "max_steps", "train_batch_size",
                        "epochs", "gnn_num_layers", "eval_interval", "eval_episodes", "reward_scale", "vf_clip_param"):
                command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
            if run.exists() or run.with_suffix(".log").exists():
                raise FileExistsError(f"Existing run: {run}; choose a fresh root")
            evaluation = None
            if args.development_episodes:
                evaluation = [sys.executable, "-u", str(Path(__file__).with_name("evaluate.py")),
                    str(run / f"policy_{args.iterations:05d}.pt"), "--episodes", str(args.development_episodes),
                    "--seed-start", str(args.development_seed_start), "--output", str(run / "development.json")]
            jobs.append((run, command, evaluation))
    sources = [Path(__file__).with_name(name) for name in
               ("env_core.py", "gnn_comm_layer.py", "marl_agent.py", "train.py", "evaluate.py",
                "ppo_diagnostics.py")]
    source_hashes = lambda: {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    expected_sources = source_hashes()
    if args.execute:
        root = Path(args.root)
        root.mkdir(parents=True, exist_ok=True)
        with (root / "schedule.json").open("x") as stream:
            json.dump({"arguments": vars(args), "source_hashes": expected_sources,
                       "jobs": [{"run": str(run), "train": command, "evaluate": evaluation}
                                for run, command, evaluation in jobs]}, stream, indent=2)
    for run, command, evaluation in jobs:
        print(shlex.join(command), flush=True)
        if evaluation:
            print(shlex.join(evaluation), flush=True)
        if args.execute:
            if source_hashes() != expected_sources:
                raise RuntimeError("Source changed during sweep; stopping to avoid mixing implementations")
            with run.with_suffix(".log").open("x") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            if source_hashes() != expected_sources:
                raise RuntimeError("Source changed during training; do not pool this run with other results")
            if evaluation:
                subprocess.run(evaluation, check=True)


if __name__ == "__main__":
    main()
