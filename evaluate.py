"""Seeded task evaluation of portable exports, independent of Ray workers.

Communication counts describe logical directed deliveries, not network traffic.
The implementation still scores and constructs messages for dense node pairs.
"""
import argparse
from collections import deque
import json
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from env_core import MultiRobotPhysicsEnv
from marl_agent import GNNMARLModel


def load_export(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 2:
        raise ValueError("Expected a version 2 portable export from train.py")
    model = GNNMARLModel(None, None, 4, {"custom_model_config": payload["model_config"]}, "evaluation")
    model.load_state_dict(payload["weights"], strict=True)
    model.set_top_k(payload["active_top_k"])
    model.eval()
    return model, payload


def evaluation_hashes():
    return {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("env_core.py", "gnn_comm_layer.py", "marl_agent.py", "evaluate.py")}


def observation_batch(obs):
    # Insertion order matches robot indices even when N >= 10.
    agents = list(obs)
    return {key: torch.from_numpy(np.stack([obs[a][key] for a in agents]))
            for key in obs[agents[0]]}


@torch.no_grad()
def action_sensitivity(model, batch):
    """Direct first-layer message intervention; NOT task-return causal value."""
    base, _ = model.forward({"obs": batch}, [], None)
    layer = model.gnn_layer.gat_layers[0]
    # Robot-relative graphs differ across focal actors. Inspect each actor's own
    # receiver row, rather than all receiver rows from robot zero's graph.
    indices = batch["node_index"].reshape(-1).long()
    batch_rows = torch.arange(len(indices), device=indices.device)
    selected = layer.last_selected_mask[batch_rows, indices].clone()
    scores = layer.last_attention_scores[batch_rows, indices].clone()
    rows = []
    for actor_row, sender in selected.nonzero().tolist():
        receiver = int(indices[actor_row])
        layer.message_ablation = (receiver, sender)
        try:
            altered, _ = model.forward({"obs": batch}, [], None)
            # Gaussian means and log stds are recorded separately; sensitivity
            # alone does not establish whether a message is beneficial.
            rows.append({"receiver": receiver, "sender": sender,
                         "attention_score": float(scores[actor_row, sender]),
                         "mean_action_change": float(torch.linalg.vector_norm(
                             altered[actor_row, :2] - base[actor_row, :2])),
                         "log_std_change": float(torch.linalg.vector_norm(
                             altered[actor_row, 2:] - base[actor_row, 2:]))})
        finally:
            layer.message_ablation = None
    model.forward({"obs": batch}, [], None)
    return rows


@torch.no_grad()
def evaluate(model, env_config, seeds, *, candidate_loss=0.0, delay=0, noise=0.0,
             diagnostic_interval=0, trajectory_callback=None):
    if not 0 <= candidate_loss <= 1 or delay < 0 or noise < 0:
        raise ValueError("Invalid robustness settings")
    env = MultiRobotPhysicsEnv(env_config)
    previous_mode = model.training
    model.eval()
    episodes, diagnostics = [], []
    try:
        for seed in seeds:
            rng = np.random.default_rng(np.random.SeedSequence([int(seed), 519]))
            obs, _ = env.reset(seed=int(seed))
            history = deque(maxlen=delay + 1)
            reward, selected, available, duration = 0.0, 0.0, 0.0, 0.0
            raw_reward = 0.0
            for step in range(env.max_steps):
                batch = observation_batch(obs)
                history.append(batch["node_features"].clone())
                batch["node_features"] = history[0].clone()
                if noise:
                    perturbation = torch.from_numpy(rng.normal(0, noise,
                        batch["node_features"].shape[1:]).astype(np.float32))
                    batch["node_features"] += perturbation.unsqueeze(0)
                    batch["local_obs"] += perturbation
                if candidate_loss:
                    keep = torch.from_numpy((rng.random(batch["adj_matrix"].shape[1:])
                                             >= candidate_loss).astype(np.float32))
                    batch["adj_matrix"] *= keep.unsqueeze(0)
                start = time.perf_counter()
                logits, _ = model.forward({"obs": batch}, [], None)
                duration += time.perf_counter() - start
                if not model.no_comm:
                    for layer in model.gnn_layer.gat_layers:
                        selected += layer.last_selected_edges / env.num_robots
                        available += layer.last_available_edges / env.num_robots
                if diagnostic_interval and step % diagnostic_interval == 0 and not model.no_comm:
                    diagnostics.extend(dict(row, episode_seed=int(seed), step=step)
                                       for row in action_sensitivity(model, batch))
                # RLlib's normalized Gaussian mean is mapped to [-1,1], then clipped.
                actions = {agent: logits[i, :2].cpu().numpy().clip(-1, 1)
                           for i, agent in enumerate(obs)}
                before = env._get_payload_position().copy() if trajectory_callback else None
                obs, rewards, terminated, truncated, infos = env.step(actions)
                reward += sum(rewards.values()) / env.num_robots
                raw_reward += sum(info["raw_reward"] for info in infos.values()) / env.num_robots
                if trajectory_callback is not None:
                    trajectory_callback(int(seed), step + 1, env, actions, before, infos)
                if terminated["__all__"] or truncated["__all__"]:
                    break
            info = next(iter(infos.values()))
            episodes.append({"episode_seed": int(seed), "success": info["success"],
                             "final_payload_distance": info["payload_dist"],
                             "payload_progress": info["payload_progress"],
                             "steps": step + 1, "return_per_agent": reward,
                             "raw_return_per_agent": raw_reward,
                             "action_effort_per_agent": info["action_effort"],
                             "final_active_contacts": info["active_contacts"],
                             "final_mean_robot_payload_distance": info["mean_robot_payload_distance"],
                             "logical_deliveries": selected,
                             "available_directed_edges_across_rounds": available,
                             "inference_seconds": duration,
                             "logical_payload_bytes": selected * model.local_hidden_dim * 4})
    finally:
        env.close()
        model.train(previous_mode)
    return episodes, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=100000)
    parser.add_argument("--num-robots", type=int)
    parser.add_argument("--comm-radius", type=float)
    parser.add_argument("--candidate-loss", type=float, default=0)
    parser.add_argument("--delay", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--diagnostic-interval", type=int, default=0)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes must be positive")
    torch.set_num_threads(1)
    model, payload = load_export(args.checkpoint)
    config = dict(payload["env_config"], render_mode="headless")
    for key in ("num_robots", "comm_radius"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    episodes, diagnostics = evaluate(model, config,
        range(args.seed_start, args.seed_start + args.episodes),
        candidate_loss=args.candidate_loss, delay=args.delay, noise=args.noise,
        diagnostic_interval=args.diagnostic_interval)
    result = {"format_version": 2, "checkpoint": str(Path(args.checkpoint).resolve()),
              "training_seed": payload["training_seed"], "iteration": payload["iteration"],
              "env_steps": payload["env_steps"], "training_protocol": payload["training_protocol"],
              "model_config": payload["model_config"], "active_top_k": payload["active_top_k"],
              "env_config": config, "source_hashes": payload["source_hashes"],
              "evaluation_source_hashes": evaluation_hashes(),
              "protocol": {"candidate_loss": args.candidate_loss, "delay": args.delay,
                           "noise": args.noise, "action_selection": "clipped_gaussian_mean"},
              "episodes": episodes, "message_sensitivity": diagnostics}
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"Success: {np.mean([r['success'] for r in episodes]):.3f}; "
          f"distance: {np.mean([r['final_payload_distance'] for r in episodes]):.3f}")


if __name__ == "__main__":
    main()
