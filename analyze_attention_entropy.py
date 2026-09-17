"""Measure first-layer routing scores on trained-policy trajectories.

Entropy measures concentration, not message usefulness or causality.
Requires a complete portable export; legacy partial weights are rejected.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from env_core import MultiRobotPhysicsEnv
from evaluate import load_export, observation_batch


@torch.no_grad()
def analyze(path, episodes=10, seed_start=60000):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    model, payload = load_export(path)
    if model.no_comm:
        raise ValueError("No-communication policies have no routing scores")
    env = MultiRobotPhysicsEnv(dict(payload["env_config"], render_mode="headless"))
    entropies, margins, overlaps, churns, eligible = [], [], [], [], []
    selected_counts = np.zeros((env.num_robots, env.num_robots), dtype=int)
    available_counts = np.zeros_like(selected_counts)
    try:
        for seed in range(seed_start, seed_start + episodes):
            obs, _ = env.reset(seed=seed)
            previous = None
            for _ in range(env.max_steps):
                batch = observation_batch(obs)
                logits, _ = model.forward({"obs": batch}, [], None)
                layer = model.gnn_layer.gat_layers[0]
                valid = batch["adj_matrix"][0].bool()
                scores = layer.last_attention_scores[0]
                selected = layer.last_selected_mask[0]
                selected_counts += selected.numpy()
                available_counts += valid.numpy().astype(int)
                for node in range(env.num_robots):
                    values = scores[node][valid[node]]
                    degree = len(values)
                    k = degree if model.top_k is None else min(model.top_k, degree)
                    eligible.append(degree > k)
                    if degree > 1:
                        probs = values.softmax(dim=0)
                        entropies.append(float(-(probs * probs.clamp_min(1e-12).log()).sum() / np.log(degree)))
                    if 0 < k < degree:
                        ordered = values.sort(descending=True).values
                        margins.append(float(ordered[k - 1] - ordered[k]))
                        random_scores = batch["random_comm_mask"][0, node].masked_fill(~valid[node], -float("inf"))
                        random_set = torch.zeros_like(valid[node]).scatter(0, random_scores.topk(k).indices, True)
                        overlaps.append(float((random_set & selected[node]).sum()) / k)
                if previous is not None:
                    union = (previous | selected).sum().item()
                    churns.append(float((previous ^ selected).sum()) / union if union else 0.0)
                previous = selected.clone()
                actions = {agent: logits[i, :2].numpy().clip(-1, 1) for i, agent in enumerate(obs)}
                obs, _, term, trunc, _ = env.step(actions)
                if term["__all__"] or trunc["__all__"]:
                    break
    finally:
        env.close()
    mean = lambda values: float(np.mean(values)) if values else None
    return {"checkpoint": str(Path(path).resolve()), "episodes": episodes, "seed_start": seed_start,
            "normalized_neighbor_score_entropy": mean(entropies),
            "topk_boundary_score_margin": mean(margins),
            "overlap_with_observation_random_topk": mean(overlaps),
            "selection_churn": mean(churns), "fraction_nodes_with_degree_above_k": mean(eligible),
            "selected_counts": selected_counts.tolist(), "available_counts": available_counts.tolist(),
            "interpretation_limit": "Concentration and selection stability do not establish usefulness. Scores are averaged across heads; self loops excluded."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=60000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = analyze(args.checkpoint, args.episodes, args.seed_start)
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("_counts")}, indent=2))


if __name__ == "__main__":
    main()
