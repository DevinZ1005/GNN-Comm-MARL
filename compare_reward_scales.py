"""Paired comparison allowing only an explicitly specified reward-scale change."""
import argparse
import json
from pathlib import Path

from compare_evaluations import paired_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", nargs="+", required=True)
    parser.add_argument("--right", nargs="+", required=True)
    parser.add_argument("--scales", nargs=2, type=float, required=True, metavar=("LEFT", "RIGHT"))
    parser.add_argument("--metric", default="success", choices=["success", "final_payload_distance",
        "payload_progress", "raw_return_per_agent", "logical_deliveries", "action_effort_per_agent"])
    parser.add_argument("--output")
    args = parser.parse_args()
    result = paired_comparison(args.left, args.right, args.metric, reward_scales=args.scales)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        with Path(args.output).open("x") as stream:
            stream.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
