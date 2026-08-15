#!/usr/bin/env python3
"""
Print the last N logged training iterations for each run log file.

Usage:
    python show_last_iters.py                     # defaults: logs/*.log, last 10 lines
    python show_last_iters.py logs/real_*.log      # specific glob (shell-expanded)
    python show_last_iters.py --n 20               # change how many lines per file
    python show_last_iters.py --pattern "logs/real_*.log" --n 5
"""
import argparse
import glob
import re
import sys

ITER_LINE_RE = re.compile(r"^Iter\s+\d+\s*\|")


def last_iter_lines(filepath: str, n: int):
    lines = []
    try:
        with open(filepath, "r", errors="ignore") as f:
            for line in f:
                line = line.rstrip("\n")
                if ITER_LINE_RE.match(line):
                    lines.append(line)
    except FileNotFoundError:
        return None
    return lines[-n:]


def main():
    parser = argparse.ArgumentParser(description="Show the last N training-iteration lines per log file.")
    parser.add_argument("files", nargs="*", help="Explicit log file paths (shell-expanded globs work too).")
    parser.add_argument("--pattern", default="logs/*.log", help="Glob pattern used if no files are given.")
    parser.add_argument("--n", type=int, default=10, help="Number of trailing iterations to show per file.")
    args = parser.parse_args()

    filepaths = args.files if args.files else sorted(glob.glob(args.pattern))

    if not filepaths:
        print(f"No log files found (pattern: {args.pattern!r}).", file=sys.stderr)
        sys.exit(1)

    for path in filepaths:
        lines = last_iter_lines(path, args.n)
        print(f"\n=== {path} ===")
        if lines is None:
            print("  (file not found)")
        elif not lines:
            print("  (no 'Iter ...' lines found yet)")
        else:
            for line in lines:
                print(f"  {line}")


if __name__ == "__main__":
    main()
