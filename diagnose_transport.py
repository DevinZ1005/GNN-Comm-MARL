"""Record deterministic development trajectories and test extra time without retraining."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate import evaluate, evaluation_hashes, load_export


def trajectory_row(seed, step, env, actions, before, infos):
    payload = env._get_payload_position().copy()
    elapsed = env.dt * env.physics_steps_per_env_step
    velocity = (payload[:2] - before[:2]) / elapsed
    delta = env.goal_pos[:2] - before[:2]
    direction = delta / max(float(np.linalg.norm(delta)), 1e-8)
    distances = np.linalg.norm(env.robot_positions[:, :2] - payload[:2], axis=1)
    weights = np.maximum(0, 1 - distances / env.contact_radius)
    robot_speed = np.linalg.norm(env.robot_velocities[:, :2], axis=1)
    weighted_speed = float(np.dot(weights, robot_speed))
    resultant = float(np.linalg.norm((weights[:, None] * env.robot_velocities[:, :2]).sum(axis=0)))
    info = next(iter(infos.values()))
    return {"episode_seed": seed, "step": step,
            "payload_distance": info["payload_dist"],
            "payload_speed": float(np.linalg.norm(velocity)),
            "goal_velocity": float(np.dot(velocity, direction)),
            "active_contacts": info["active_contacts"],
            "velocity_agreement": resultant / weighted_speed if weighted_speed > 1e-8 else 0.0,
            "mean_robot_speed": float(robot_speed.mean()),
            "mean_robot_payload_distance": info["mean_robot_payload_distance"],
            "action_saturation_fraction": float(np.mean(np.abs(list(actions.values())) >= 0.999)),
            "payload_position": payload.tolist(), "goal_position": env.goal_pos.tolist(),
            "robot_positions": env.robot_positions.tolist(),
            "robot_headings": env.robot_headings.tolist(),
            "actions": {key: value.tolist() for key, value in actions.items()}}


def summarize_trajectory(rows, success, min_contacts, tail_steps=50):
    tail = rows[-tail_steps:]
    return {"success": bool(success), "steps": len(rows),
            "minimum_distance": min(row["payload_distance"] for row in rows),
            "final_distance": rows[-1]["payload_distance"],
            "tail_mean_goal_velocity": float(np.mean([r["goal_velocity"] for r in tail])),
            "tail_contact_shortfall_fraction": float(np.mean([
                r["active_contacts"] < min_contacts for r in tail])),
            "tail_mean_velocity_agreement": float(np.mean([r["velocity_agreement"] for r in tail])),
            "tail_mean_robot_speed": float(np.mean([r["mean_robot_speed"] for r in tail]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed-start", type=int, default=80000)
    parser.add_argument("--extended-steps", type=int, default=750)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error("output already exists; choose a new path")
    model, payload = load_export(args.checkpoint)
    config = dict(payload["env_config"], render_mode="headless")
    original_steps = config["max_steps"]
    if args.episodes < 1 or args.extended_steps <= original_steps:
        parser.error("episodes must be positive and extended steps must exceed training horizon")
    torch.set_num_threads(1)
    traces = {seed: [] for seed in range(args.seed_start, args.seed_start + args.episodes)}

    def record(seed, step, env, actions, before, infos):
        traces[seed].append(trajectory_row(seed, step, env, actions, before, infos))

    # max_steps affects truncation only; the prefix is the original deterministic rollout.
    extended_config = dict(config, max_steps=args.extended_steps)
    episodes, _ = evaluate(model, extended_config, traces, trajectory_callback=record)
    summaries = []
    for episode in episodes:
        seed = episode["episode_seed"]
        rows = traces[seed]
        base_success = episode["success"] and episode["steps"] <= original_steps
        summaries.append({"episode_seed": seed,
            "original": summarize_trajectory(rows[:original_steps], base_success,
                                              config.get("min_payload_contacts", 2)),
            "extended": summarize_trajectory(rows, episode["success"],
                                              config.get("min_payload_contacts", 2))})
    aggregate = {"episodes": len(episodes), "original_successes": sum(
        s["original"]["success"] for s in summaries), "extended_successes": sum(
        s["extended"]["success"] for s in summaries)}
    result = {"checkpoint": str(Path(args.checkpoint).resolve()),
              "iteration": payload["iteration"], "training_seed": payload["training_seed"],
              "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              "training_source_hashes": payload["source_hashes"],
              "evaluation_source_hashes": evaluation_hashes(),
              "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "original_env_config": config, "extended_env_config": extended_config,
              "protocol": "Development only; clipped Gaussian means; extra time without retraining",
              "aggregate": aggregate, "summaries": summaries, "trajectories": traces}
    with Path(args.output).open("x") as stream:
        json.dump(result, stream, allow_nan=False)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
