"""Compare matched held-out evaluations using training seeds as replicates."""
import argparse
import json
from pathlib import Path

import numpy as np


def paired_comparison(left, right, metric, samples=10000, *, reward_scales=None, value_clips=None):
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    if value_clips is not None:
        if reward_scales is not None:
            raise ValueError("Only one intervention may change at a time")
        if (len(value_clips) != 2 or value_clips[0] == value_clips[1]
                or any(not np.isfinite(value) or value <= 0 for value in value_clips)):
            raise ValueError("Specify two distinct finite positive value clips")
    if reward_scales is not None:
        if (len(reward_scales) != 2 or reward_scales[0] == reward_scales[1]
                or any(not np.isfinite(scale) or scale <= 0 for scale in reward_scales)):
            raise ValueError("Specify two distinct finite positive reward scales")
        if metric == "return_per_agent":
            raise ValueError("Compare raw_return_per_agent across reward scales")
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
        if value_clips is not None:
            for data, expected in zip((first, second), value_clips):
                if data["training_protocol"]["optimization_config"].get("vf_clip_param") != expected:
                    raise ValueError(f"Unexpected vf_clip_param for training seed {seed}")
            if (first["model_config"] != second["model_config"]
                    or first.get("active_top_k") != second.get("active_top_k")):
                raise ValueError("Value-clip comparisons require identical routing and architecture")
        if reward_scales is not None:
            for data, expected in zip((first, second), reward_scales):
                if data["env_config"].get("reward_scale") != expected:
                    raise ValueError(f"Unexpected reward_scale for training seed {seed}")
            if (first["model_config"] != second["model_config"]
                    or first.get("active_top_k") != second.get("active_top_k")):
                raise ValueError("Reward-scale comparisons require identical routing and architecture")
        def matched_value(data, key):
            value = data[key]
            if key == "env_config" and reward_scales is not None:
                return {name: item for name, item in value.items() if name != "reward_scale"}
            if key == "training_protocol" and value_clips is not None:
                return dict(value, optimization_config={name: item for name, item in
                    value["optimization_config"].items() if name != "vf_clip_param"})
            return value
        keys = ("env_config", "protocol", "iteration", "env_steps", "training_protocol",
                "source_hashes", "evaluation_source_hashes")
        for key in keys:
            if matched_value(first, key) != matched_value(second, key):
                raise ValueError(f"Unmatched {key} for training seed {seed}")
        protocol = {key: matched_value(first, key) for key in keys}
        fixed_model = lambda data: {key: value for key, value in data["model_config"].items()
                                    if reward_scales is not None or value_clips is not None or key not in
                                    ("top_k", "topk_mode", "no_comm", "gumbel_temperature")}
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
    result = {"metric": metric, "direction": "left minus right", "training_seeds": len(values),
            "seed_ids": sorted(a), "left_means": left_means, "right_means": right_means,
            "paired_differences": differences, "mean_difference": float(values.mean()),
            "bootstrap_95_interval": np.quantile(bootstrap, [.025, .975]).tolist(),
            "paired_standardized_effect": (float(values.mean() / values.std(ddof=1))
                                           if values.std(ddof=1) > 0 else None),
            "caution": "Small seed counts give unstable intervals; this is one prespecified comparison, not multiple-testing correction."}
    if reward_scales is not None:
        result["intervention"] = {"field": "env_config.reward_scale",
                                  "left": reward_scales[0], "right": reward_scales[1]}
    if value_clips is not None:
        result["intervention"] = {"field": "training_protocol.optimization_config.vf_clip_param",
                                  "left": value_clips[0], "right": value_clips[1]}
    return result


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
