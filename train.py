"""
Distributed Multi-Agent Training Entrypoint (`train.py`).

This script configures and launches distributed Multi-Agent Proximal Policy Optimization (MAPPO)
using Ray RLlib. It registers the custom physics environment (`MultiRobotPhysicsEnv`) and the
dynamic topological GNN policy (`GNNMARLModel`).

Training Strategy & Parameter Sharing:
To maximize sample efficiency across identical homogeneous robots, we utilize decentralized
execution with parameter sharing under a shared policy (`shared_gnn_policy`). Each robot evaluates
its local actions using the identical GNN weights, with symmetry broken via unique local sensor inputs
and neighborhood communication embeddings.
"""

import argparse
import os
import sys
from typing import Dict, Any

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from env_core import MultiRobotPhysicsEnv
from marl_agent import GNNMARLModel


def parse_arguments() -> argparse.Namespace:
    """Parse command line options for distributed training configuration."""
    parser = argparse.ArgumentParser(description="Train Multi-Robot GNN Coordination with Ray RLlib.")
    parser.add_argument("--num-robots", type=int, default=4, help="Number of robots in simulation.")
    parser.add_argument("--comm-radius", type=float, default=6.0, help="Euclidean communication radius R_comm.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of parallel rollout workers.")
    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum training epochs.")
    parser.add_argument("--render", action="store_true", help="Run with PyBullet GUI enabled.")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Directory to save model weights.")
    parser.add_argument("--train-batch-size", type=int, default=1000, help="PPO train batch size (total env steps per iteration).")
    parser.add_argument("--top-k", type=int, default=None, help="Top-K communication sparsification budget.")
    parser.add_argument("--topk-mode", type=str, choices=["attention", "random"], default="attention", help="Top-K neighbor selection strategy.")
    parser.add_argument("--top-k-anneal-steps", type=int, default=None, help="Number of training iterations to anneal top_k from dense to target.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--gnn-num-layers", type=int, default=2, help="Number of GAT message-passing layers.")
    return parser.parse_args()


class CommunicationSparsificationCallback(DefaultCallbacks):
    """Callback to extract communication sparsification metrics after each training iteration."""

    def on_train_result(self, *, algorithm, result: Dict[str, Any], **kwargs: Any) -> None:
        policy = algorithm.get_policy("shared_gnn_policy")
        if policy is not None and hasattr(policy, "model") and hasattr(policy.model, "get_drop_frac"):
            result["drop_frac"] = policy.model.get_drop_frac()
        else:
            result["drop_frac"] = float("nan")

        # Drive top-k annealing from the training iteration count.
        # This is centralized in the callback so all workers stay in sync.
        config = algorithm.config
        custom_cfg = config.get("model", {}).get("custom_model_config", {})
        target_top_k = custom_cfg.get("top_k", None)
        anneal_steps = custom_cfg.get("top_k_anneal_steps", None)
        if target_top_k is not None and anneal_steps is not None:
            iteration = result.get("training_iteration", 0)
            if iteration >= anneal_steps:
                current_top_k = target_top_k
            else:
                # Linearly anneal from dense (None -> large K) down to target_top_k
                num_robots = config.get("env_config", {}).get("num_robots", 4)
                max_k = max(1, num_robots - 1)
                progress = iteration / float(anneal_steps)
                current_top_k = int(round(max_k - progress * (max_k - target_top_k)))
                current_top_k = max(target_top_k, current_top_k)
                if current_top_k >= max_k:
                    current_top_k = None

            # Apply to the learner (local) policy model
            if policy is not None and hasattr(policy.model, "set_top_k"):
                policy.model.set_top_k(current_top_k)

            # Apply to all remote env runner policy models
            def _set_worker_top_k(env_runner):
                p = env_runner.get_policy("shared_gnn_policy")
                if p is not None and hasattr(p.model, "set_top_k"):
                    p.model.set_top_k(current_top_k)

            if hasattr(algorithm, "env_runner_group") and algorithm.env_runner_group is not None:
                algorithm.env_runner_group.foreach_env_runner(_set_worker_top_k, local_env_runner=False)


def env_creator(env_config: Dict[str, Any]) -> MultiRobotPhysicsEnv:
    """Factory function for registering MultiRobotPhysicsEnv with RLlib."""
    return MultiRobotPhysicsEnv(env_config)


def main() -> None:
    """Main execution loop setting up RLlib algorithms and executing MAPPO training."""
    args = parse_arguments()

    # 1. Initialize Ray cluster (local instance or distributed cluster)
    ray.init(ignore_reinit_error=True)

    # 2. Register custom environment and GNN model class
    register_env("MultiRobotPhysicsEnv-v0", env_creator)
    ModelCatalog.register_custom_model("GNNMARLModel", GNNMARLModel)

    # Instantiate a temporary environment to extract canonical observation and action spaces
    tmp_env = MultiRobotPhysicsEnv({"num_robots": args.num_robots, "comm_radius": args.comm_radius})
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space

    # 3. Configure Multi-Agent Proximal Policy Optimization (PPO)
    config = (
        PPOConfig()
        .debugging(seed=args.seed)
        .environment(
            env="MultiRobotPhysicsEnv-v0",
            env_config={
                "num_robots": args.num_robots,
                "comm_radius": args.comm_radius,
                "render_mode": "gui" if args.render else "headless",
                "max_steps": 500
            }
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .env_runners(
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=1
        )
        .callbacks(CommunicationSparsificationCallback)
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            entropy_coeff=0.01,
            train_batch_size=args.train_batch_size,
            model={
                "custom_model": "GNNMARLModel",
                "custom_model_config": {
                    "raw_obs_dim": tmp_env.raw_obs_dim,
                    "edge_dim": tmp_env.edge_dim,
                    "comm_latent_dim": 64,
                    "local_hidden_dim": 128,
                    "gnn_num_layers": args.gnn_num_layers,
                    "gnn_num_heads": 4,
                    "top_k": args.top_k,
                    "topk_mode": args.topk_mode,
                    "top_k_anneal_steps": args.top_k_anneal_steps,
                }
            }
        )
        .multi_agent(
            policies={
                "shared_gnn_policy": (None, obs_space, act_space, {})
            },
            policy_mapping_fn=lambda *args, **kwargs: "shared_gnn_policy"
        )
        .resources(
            num_gpus=0
        )
    )

    import torch as _torch
    _num_gpus = 1 if _torch.cuda.is_available() else 0
    config.sgd_minibatch_size = min(128, args.train_batch_size)
    config.num_epochs = 10
    config.num_gpus_per_learner = _num_gpus
    config.num_gpus_per_env_runner = 0

    # 4. Build the RLlib algorithm instance
    print("Building Ray RLlib MAPPO algorithm with Dynamic Topological GNN architecture...")
    algo = config.build_algo()

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 5. Execute training iterations with telemetry reporting
    print(f"Starting training for {args.max_iterations} iterations across {args.num_workers} workers...")
    try:
        for iteration in range(1, args.max_iterations + 1):
            results = algo.train()
            
            # Extract key performance indicators
            # Try new Ray location (results["env_runners"]) first, fall back to flat keys
            env_runners = results.get("env_runners", {})
            reward_mean = env_runners.get("episode_return_mean",
                          results.get("episode_reward_mean", float("nan")))
            episode_len = env_runners.get("episode_len_mean",
                          results.get("episode_len_mean", float("nan")))
            policy_loss = results.get("info", {}).get("learner", {}).get("shared_gnn_policy", {}).get("learner_stats", {}).get("policy_loss", float("nan"))
            drop_frac = results.get("drop_frac", float("nan"))

            print(f"Iter {iteration:03d} | Reward Mean: {reward_mean:8.2f} | Ep Len: {episode_len:5.1f} | Policy Loss: {policy_loss:6.4f} | Drop Frac: {drop_frac:6.4f}")

            # Periodic checkpointing every 10 iterations
            if iteration % 10 == 0 or iteration == args.max_iterations:
                checkpoint_path = algo.save(args.checkpoint_dir)
                print(f"Saved policy checkpoint -> {checkpoint_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving emergency checkpoint...")
        checkpoint_path = algo.save(os.path.join(args.checkpoint_dir, "emergency_checkpoint"))
        print(f"Emergency checkpoint saved -> {checkpoint_path}")
    finally:
        algo.stop()
        ray.shutdown()
        print("Ray training session closed cleanly.")


if __name__ == "__main__":
    main()
