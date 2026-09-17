"""Print a matched experiment schedule; --execute runs jobs sequentially.

Existing outputs are never overwritten. Use a fresh root for a new experiment.
"""
import argparse
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
    parser.add_argument("--goal-observers", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.budgets)) != len(args.budgets):
        parser.error("duplicate seeds or budgets")
    if any(k < 1 or k >= args.num_robots for k in args.budgets):
        parser.error("budgets must be between 1 and N-1")
    conditions = [("dense", []), ("no_comm", ["--no-comm"])]
    for mode in ("attention", "random", "distance", "gumbel"):
        conditions.extend((f"{mode}_k{k}", ["--topk-mode", mode, "--top-k", str(k)]) for k in args.budgets)
    jobs = []
    for seed in args.seeds:
        for name, flags in conditions:
            run = Path(args.root).resolve() / f"{name}_s{seed}"
            command = [sys.executable, str(Path(__file__).with_name("train.py")),
                       "--backend", args.backend, "--num-robots", str(args.num_robots),
                       "--comm-radius", str(args.comm_radius), "--goal-observers", str(args.goal_observers),
                       "--seed", str(seed),
                       "--num-workers", str(args.workers), "--max-iterations", str(args.iterations),
                       "--checkpoint-dir", str(run)] + flags
            if run.exists():
                raise FileExistsError(f"Existing run: {run}; choose a fresh root")
            jobs.append((run, command))
    for run, command in jobs:
        print(shlex.join(command), flush=True)
        if args.execute:
            run.parent.mkdir(parents=True, exist_ok=True)
            with run.with_suffix(".log").open("x") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
