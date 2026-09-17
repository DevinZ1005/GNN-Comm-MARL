"""Validate transport feasibility and graph structure before RL experiments."""
import argparse
import json

import numpy as np

from env_core import MultiRobotPhysicsEnv


def _angle_error(target, heading):
    return (target - heading + np.pi) % (2 * np.pi) - np.pi


def scripted_actions(env):
    payload = env._get_payload_position()
    goal_direction = np.arctan2(env.goal_pos[1] - payload[1], env.goal_pos[0] - payload[0])
    actions = {}
    for index, agent in enumerate(env._agent_ids):
        displacement = payload[:2] - env.robot_positions[index, :2]
        distance = float(np.linalg.norm(displacement))
        target = (np.arctan2(displacement[1], displacement[0])
                  if distance > 0.9 * env.contact_radius else goal_direction)
        error = _angle_error(target, env.robot_headings[index])
        forward = max(0.0, float(np.cos(error)))
        turn = float(np.clip(error, -1.0, 1.0))
        actions[agent] = np.clip([forward + turn, forward - turn], -1, 1).astype(np.float32)
    return actions


def random_actions(env, rng):
    return {agent: rng.uniform(-1, 1, 2).astype(np.float32) for agent in env._agent_ids}


def rollout(config, seeds, controller):
    env = MultiRobotPhysicsEnv(config)
    rows = []
    try:
        for seed in seeds:
            rng = np.random.default_rng(np.random.SeedSequence([seed, 913]))
            _, _ = env.reset(seed=seed)
            degrees = env.adj_matrix.sum(axis=1)
            initial_edges = float(env.adj_matrix.sum())
            initial_partial = bool(np.any(degrees < env.num_robots - 1))
            for step in range(env.max_steps):
                actions = controller(env) if controller is scripted_actions else controller(env, rng)
                _, _, terminated, truncated, infos = env.step(actions)
                if terminated["__all__"] or truncated["__all__"]:
                    break
            info = infos["robot_0"]
            rows.append({"seed": seed, "success": info["success"], "steps": step + 1,
                         "final_payload_distance": info["payload_dist"],
                         "payload_progress": info["payload_progress"],
                         "initial_directed_edges": initial_edges,
                         "initial_partial_graph": initial_partial,
                         "initial_mean_degree": float(degrees.mean())})
    finally:
        env.close()
    return rows


def validate(config, episodes=20, seed_start=70000):
    seeds = range(seed_start, seed_start + episodes)
    scripted = rollout(config, seeds, scripted_actions)
    random = rollout(config, seeds, random_actions)
    summarize = lambda rows: {
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "mean_final_payload_distance": float(np.mean([row["final_payload_distance"] for row in rows])),
        "mean_payload_progress": float(np.mean([row["payload_progress"] for row in rows])),
    }
    return {"environment": config, "episodes_per_controller": episodes,
            "scripted": summarize(scripted), "random": summarize(random),
            "fraction_initial_partial_graphs": float(np.mean(
                [row["initial_partial_graph"] for row in scripted])),
            "mean_initial_degree": float(np.mean([row["initial_mean_degree"] for row in scripted])),
            "acceptance": {
                "scripted_transport_feasible": all(row["success"] for row in scripted),
                "random_not_solving_task": summarize(random)["success_rate"] <= 0.1,
                "proximity_graph_not_always_complete": any(row["initial_partial_graph"] for row in scripted),
            }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-robots", type=int, default=8)
    parser.add_argument("--comm-radius", type=float, default=1.8)
    parser.add_argument("--goal-observers", type=int, default=-1)
    parser.add_argument("--output")
    args = parser.parse_args()
    config = {"backend": "kinematic", "num_robots": args.num_robots,
              "comm_radius": args.comm_radius, "goal_observers": args.goal_observers}
    result = validate(config, args.episodes)
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        with open(args.output, "x") as stream:
            stream.write(encoded + "\n")
    print(encoded)
    if not all(result["acceptance"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
