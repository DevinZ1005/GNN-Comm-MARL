"""Compare matched held-out evaluations using training seeds as replicates."""
import argparse
import json
from pathlib import Path

import numpy as np


def paired_comparison(left, right, metric, samples=10000):
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    def index(paths):
        runs = {}
        for path in paths:
            data = json.loads(Path(path).read_text())
            seed = data["training_seed"]
            if seed in runs:
                raise ValueError(f"Duplicate training seed: {seed}")
            if not data["episodes"]:
                raise ValueError("Empty evaluation")
            runs[seed] = data
        return runs
    a, b = index(left), index(right)
    if set(a) != set(b) or len(a) < 2:
        raise ValueError("Need the same set of at least two independent training seeds")
    differences, left_means, right_means = [], [], []
    reference = None
    reference_model = None
    reference_episodes = None
    for seed in sorted(a):
        first, second = a[seed], b[seed]
        keys = ("env_config", "protocol", "iteration", "env_steps", "training_protocol",
                "source_hashes", "evaluation_source_hashes")
        for key in keys:
            if first[key] != second[key]:
                raise ValueError(f"Unmatched {key} for training seed {seed}")
        protocol = {key: first[key] for key in keys}
        fixed_model = lambda data: {key: value for key, value in data["model_config"].items()
                                    if key not in ("top_k", "topk_mode", "no_comm", "gumbel_temperature")}
        if fixed_model(first) != fixed_model(second):
            raise ValueError("Unmatched model architecture")
        if reference_model is not None and reference_model != fixed_model(first):
            raise ValueError("Mixed model architecture across training seeds")
        reference_model = fixed_model(first)
        if reference is not None and reference != protocol:
            raise ValueError("Mixed evaluation settings across training seeds")
        reference = protocol
        ids_a = [e["episode_seed"] for e in first["episodes"]]
        ids_b = [e["episode_seed"] for e in second["episodes"]]
        if ids_a != ids_b or len(set(ids_a)) != len(ids_a):
            raise ValueError("Evaluation episode seeds must match and be unique")
        if reference_episodes is not None and reference_episodes != ids_a:
            raise ValueError("Mixed evaluation episode seeds across training seeds")
        reference_episodes = ids_a
        left_means.append(float(np.mean([e[metric] for e in first["episodes"]])))
        right_means.append(float(np.mean([e[metric] for e in second["episodes"]])))
        differences.append(left_means[-1] - right_means[-1])
    values = np.asarray(differences)
    rng = np.random.default_rng(149)
    bootstrap = rng.choice(values, (samples, len(values)), replace=True).mean(axis=1)
    return {"metric": metric, "direction": "left minus right", "training_seeds": len(values),
            "seed_ids": sorted(a), "left_means": left_means, "right_means": right_means,
            "paired_differences": differences, "mean_difference": float(values.mean()),
            "bootstrap_95_interval": np.quantile(bootstrap, [.025, .975]).tolist(),
            "paired_standardized_effect": (float(values.mean() / values.std(ddof=1))
                                           if values.std(ddof=1) > 0 else None),
            "caution": "Small seed counts give unstable intervals; this is one prespecified comparison, not multiple-testing correction."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", nargs="+", required=True)
    parser.add_argument("--right", nargs="+", required=True)
    parser.add_argument("--metric", default="success", choices=["success", "final_payload_distance",
        "payload_progress", "return_per_agent", "logical_deliveries", "action_effort_per_agent"])
    parser.add_argument("--output", help="Save the comparison to a new JSON file")
    args = parser.parse_args()
    result = paired_comparison(args.left, args.right, args.metric)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        with Path(args.output).open("x") as stream:
            stream.write(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
