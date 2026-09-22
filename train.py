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
import json
import hashlib
import inspect
import math
from pathlib import Path
import platform
from typing import Dict, Any

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from env_core import MultiRobotPhysicsEnv
from marl_agent import GNNMARLModel
from ppo_diagnostics import DIAGNOSTIC_KEYS, InstrumentedPPOTorchPolicy


def parse_arguments() -> argparse.Namespace:
    """Parse command line options for distributed training configuration."""
    parser = argparse.ArgumentParser(description="Train Multi-Robot GNN Coordination with Ray RLlib.")
    parser.add_argument("--num-robots", type=int, default=4, help="Number of robots in simulation.")
    parser.add_argument("--comm-radius", type=float, default=1.8, help="Euclidean communication radius R_comm.")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of parallel rollout workers.")
    parser.add_argument("--max-iterations", type=int, default=100, help="Maximum training epochs.")
    parser.add_argument("--render", action="store_true", help="Run with PyBullet GUI enabled.")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints", help="Directory to save model weights.")
    parser.add_argument("--train-batch-size", type=int, default=1000, help="PPO train batch size (total env steps per iteration).")
    parser.add_argument("--top-k", type=int, default=None, help="Top-K communication sparsification budget.")
    parser.add_argument("--topk-mode", type=str, choices=["attention", "gumbel", "random", "distance"], default="attention", help="Top-K neighbor selection strategy.")
    parser.add_argument("--gumbel-temperature", type=float, default=1.0, help="Temperature for Gumbel-Softmax relaxation (topk_mode='gumbel'). Lower=harder selection.")
    parser.add_argument("--top-k-anneal-steps", type=int, default=None, help="Number of training iterations to anneal top_k from dense to target.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility")
    parser.add_argument("--gnn-num-layers", type=int, default=2, help="Number of GAT message-passing layers.")
    parser.add_argument("--observation-frame", choices=["world", "robot"], default="world",
                        help="Actor coordinate frame; robot mode requires fresh training.")
    parser.add_argument("--no-comm", action="store_true", help="Zero actor communication; retain the shared centralized critic design.")
    parser.add_argument("--backend", choices=["pybullet", "kinematic"], default="pybullet")
    parser.add_argument("--reward-version", choices=["transport_v2", "legacy"], default="transport_v2")
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="Positive multiplier for all learning rewards; physics and raw metrics unchanged.")
    parser.add_argument("--vf-clip-param", type=float, default=10.0,
                        help="Positive cap on squared critic error in the installed PPO loss (not gradient clipping).")
    parser.add_argument("--goal-distance", type=float, default=4.0)
    parser.add_argument("--goal-spawn-mode", choices=["coupled", "independent"], default="coupled",
                        help="Use independent for private-goal studies; coupled preserves old scenarios.")
    parser.add_argument("--spawn-radius", type=float, default=1.0)
    parser.add_argument("--spawn-jitter", type=float, default=0.1)
    parser.add_argument("--goal-observers", type=int, default=-1,
                        help="Robots given the private goal vector; -1 gives it to all robots.")
    parser.add_argument("--min-payload-contacts", type=int, default=2)
    parser.add_argument("--approach-scale", type=float, default=1.0)
    parser.add_argument("--energy-scale", type=float, default=0.005)
    parser.add_argument("--contribution-scale", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--random-mask-scope", choices=["step", "episode"], default="step")
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-seed-start", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--resume-from", type=str,
                        help="Ray checkpoint to restore; max-iterations is the additional iteration count.")
    return parser.parse_args()


class CommunicationSparsificationCallback(DefaultCallbacks):
    """Callback to extract communication sparsification metrics after each training iteration."""

    def on_episode_end(self, *, episode, **kwargs):
        infos = [episode.last_info_for(agent) for agent in episode.get_agents()]
        info = next((value for value in infos if value and "success" in value), None)
        if info:
            for key in ("success", "payload_dist", "payload_progress", "action_effort"):
                episode.custom_metrics[key] = float(info[key])

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


def learner_metrics(results):
    """Persist available old-stack learner statistics; missing/nonfinite is null."""
    stats = results.get("info", {}).get("learner", {}).get(
        "shared_gnn_policy", {}).get("learner_stats", {})
    metrics = {}
    for key in ("total_loss", "policy_loss", "vf_loss", "vf_explained_var",
                "entropy", "kl", "cur_lr", "cur_kl_coeff", "entropy_coeff") + DIAGNOSTIC_KEYS:
        value = stats.get(key)
        metrics[key] = float(value) if value is not None and math.isfinite(float(value)) else None
    terminals = metrics["vf_terminal_samples_per_minibatch"]
    saturated = metrics["vf_saturated_terminal_samples_per_minibatch"]
    metrics["vf_terminal_saturation_fraction"] = (
        saturated / terminals if terminals and saturated is not None else None)
    return metrics


def validated_training_steps(results, previous_steps=None):
    """Reject empty/invalid learner updates before exporting an apparent iteration."""
    steps = results.get("num_env_steps_sampled_lifetime")
    stats = learner_metrics(results)
    if (steps is None or not math.isfinite(float(steps)) or int(steps) != steps
            or steps <= (previous_steps if previous_steps is not None else 0)
            or any(stats[key] is None for key in ("policy_loss", "vf_loss", "total_loss"))):
        raise RuntimeError("PPO made no valid training progress; check Ray workers and learner metrics")
    return int(steps)


def main() -> None:
    """Main execution loop setting up RLlib algorithms and executing MAPPO training."""
    args = parse_arguments()
    if (args.max_iterations < 1 or args.max_steps < 1 or args.eval_interval < 1
            or args.eval_episodes < 1 or args.epochs < 1 or args.torch_threads < 1
            or args.train_batch_size < 2 or args.num_workers < 0
            or not math.isfinite(args.reward_scale) or args.reward_scale <= 0
            or not math.isfinite(args.vf_clip_param) or args.vf_clip_param <= 0
            or (args.top_k is not None and args.top_k < 0)
            or args.gumbel_temperature <= 0
            or (args.top_k_anneal_steps is not None and args.top_k_anneal_steps < 1)):
        raise ValueError("Invalid training configuration")
    run_dir = Path(args.checkpoint_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Use a new checkpoint directory; preserving existing run: {run_dir}")
    env_config = {key: getattr(args, key) for key in (
        "num_robots", "comm_radius", "backend", "reward_version", "goal_distance",
        "spawn_radius", "spawn_jitter", "goal_observers", "min_payload_contacts",
        "approach_scale", "energy_scale", "contribution_scale", "max_steps", "random_mask_scope",
        "goal_spawn_mode", "reward_scale")}
    env_config["render_mode"] = "gui" if args.render else "headless"
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ("env_core.py", "gnn_comm_layer.py", "marl_agent.py", "train.py", "evaluate.py",
                                  "ppo_diagnostics.py")}
    value_loss_source_hash = hashlib.sha256(
        Path(inspect.getfile(PPOTorchPolicy)).read_bytes()).hexdigest()
    resume_path = None
    if args.resume_from:
        resume_path = Path(args.resume_from).resolve()
        if not resume_path.is_dir():
            raise FileNotFoundError(resume_path)
        previous = json.loads((resume_path.parent / "manifest.json").read_text())
        mutable_args = {"resume_from", "checkpoint_dir", "max_iterations",
                        "eval_interval", "eval_episodes", "eval_seed_start"}
        mismatches = [key for key, value in vars(args).items()
                      if key not in mutable_args and previous["arguments"].get(key) != value]
        mismatches += [name for name in ("env_core.py", "gnn_comm_layer.py", "marl_agent.py", "ppo_diagnostics.py")
                       if previous["source_hashes"].get(name) != source_hashes[name]]
        if previous.get("value_loss_source_sha256") != value_loss_source_hash:
            mismatches.append("value_loss_source_sha256")
        if mismatches:
            raise ValueError(f"Resume must preserve training settings and task/model code: {mismatches}")

    # 1. Initialize Ray cluster (local instance or distributed cluster)

    # 2. Register custom environment and GNN model class
    register_env("MultiRobotPhysicsEnv-v0", env_creator)
    ModelCatalog.register_custom_model("GNNMARLModel", GNNMARLModel)

    # Instantiate a temporary environment to extract canonical observation and action spaces
    tmp_env = MultiRobotPhysicsEnv(env_config)
    obs_space = tmp_env.observation_space
    act_space = tmp_env.action_space
    tmp_env.close()
    ray.init(ignore_reinit_error=True)

    # 3. Configure Multi-Agent Proximal Policy Optimization (PPO)
    config = (
        PPOConfig()
        .debugging(seed=args.seed)
        .environment(
            env="MultiRobotPhysicsEnv-v0",
            env_config=env_config
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False
        )
        .env_runners(
            num_env_runners=args.num_workers,
            num_envs_per_env_runner=1,
            rollout_fragment_length="auto"
        )
        .callbacks(CommunicationSparsificationCallback)
        .training(
            lr=3e-4,
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            vf_clip_param=args.vf_clip_param,
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
                    "gumbel_temperature": args.gumbel_temperature,
                    "top_k_anneal_steps": args.top_k_anneal_steps,
                    "no_comm": args.no_comm,
                    "observation_frame": args.observation_frame,
                }
            }
        )
        .multi_agent(
            policies={
                "shared_gnn_policy": (InstrumentedPPOTorchPolicy, obs_space, act_space, {})
            },
            policy_mapping_fn=lambda *args, **kwargs: "shared_gnn_policy"
        )
        .resources(
            num_gpus=0
        )
    )

    import torch as _torch
    _torch.set_num_threads(args.torch_threads)
    config.extra_python_environs_for_worker = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    _num_gpus = args.gpus
    config.num_gpus = _num_gpus
    config.sgd_minibatch_size = min(128, args.train_batch_size)
    config.num_epochs = args.epochs
    config.minibatch_size = min(128, args.train_batch_size)
    config.num_gpus_per_learner = _num_gpus
    config.num_gpus_per_env_runner = 0
    optimization_config = {key: getattr(config, key) for key in
        ("lr", "gamma", "lambda_", "clip_param", "vf_loss_coeff", "vf_clip_param",
         "entropy_coeff", "kl_coeff", "kl_target")}
    if resume_path is not None and previous.get("optimization_config") != optimization_config:
        ray.shutdown()
        raise ValueError("Resume must preserve optimization_config")

    # 4. Build the RLlib algorithm instance
    print("Building Ray RLlib MAPPO algorithm with Dynamic Topological GNN architecture...")
    try:
        algo = config.build_algo()
    except BaseException:
        ray.shutdown()
        raise
    if resume_path is not None:
        try:
            algo.restore(str(resume_path))
            if (algo.config.env_config != env_config or
                    algo.config.model["custom_model_config"] != config.model["custom_model_config"]):
                raise ValueError("Restored checkpoint configuration differs from requested configuration")
            CommunicationSparsificationCallback().on_train_result(
                algorithm=algo, result={"training_iteration": algo.iteration})
        except BaseException:
            algo.stop()
            ray.shutdown()
            raise
        print(f"Restored training state from {resume_path}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    from evaluate import evaluate
    model_cfg = config.model["custom_model_config"]
    with (run_dir / "manifest.json").open("x") as stream:
        json.dump({"format_version": 2, "arguments": vars(args), "env_config": env_config,
                   "model_config": model_cfg, "source_hashes": source_hashes,
                   "optimization_config": optimization_config,
                   "value_loss_source_sha256": value_loss_source_hash,
                   "policy_class": "InstrumentedPPOTorchPolicy",
                   "python": platform.python_version(), "torch": str(_torch.__version__),
                   "ray": ray.__version__}, stream, indent=2)

    # 5. Execute training iterations with telemetry reporting
    print(f"Starting training for {args.max_iterations} iterations across {args.num_workers} workers...")
    previous_steps = None
    try:
        for local_iteration in range(1, args.max_iterations + 1):
            results = algo.train()
            previous_steps = validated_training_steps(results, previous_steps)
            iteration = int(results.get("training_iteration", local_iteration))
            
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
            record = {"iteration": iteration, "training_seed": args.seed,
                      "env_steps": previous_steps,
                      "training_return": float(reward_mean), "learner_batch_drop_frac": float(drop_frac),
                      "reward_scale": args.reward_scale, "learner_metrics": learner_metrics(results)}
            if iteration % args.eval_interval == 0 or local_iteration == args.max_iterations:
                live_model = algo.get_policy("shared_gnn_policy").model
                export = {"format_version": 2, "weights": {k: v.detach().cpu() for k, v in live_model.state_dict().items()},
                          "model_config": model_cfg, "env_config": env_config,
                          "active_top_k": live_model.gnn_layer.top_k, "iteration": iteration,
                          "training_seed": args.seed, "source_hashes": source_hashes,
                          "env_steps": record["env_steps"],
                          "training_protocol": {
                              **{key: getattr(args, key) for key in
                                 ("train_batch_size", "epochs", "num_workers", "top_k_anneal_steps")},
                              "optimization_config": optimization_config,
                              "policy_class": "InstrumentedPPOTorchPolicy",
                              "value_loss_source_sha256": value_loss_source_hash}}
                export_path = run_dir / f"policy_{iteration:05d}.pt"
                _torch.save(export, export_path)
                from evaluate import load_export
                eval_model, _ = load_export(export_path)
                evaluation_config = dict(env_config, render_mode="headless")
                rows, _ = evaluate(eval_model, evaluation_config,
                    range(args.eval_seed_start, args.eval_seed_start + args.eval_episodes))
                record["validation_episodes"] = rows
                print(f"Validation success: {sum(row['success'] for row in rows) / len(rows):.3f}")
            # Missing training metrics are explicit nulls, never nonstandard JSON NaN.
            record = {key: (None if isinstance(value, float) and not math.isfinite(value) else value)
                      for key, value in record.items()}
            with (run_dir / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")

            # Periodic checkpointing every 10 iterations
            if iteration % 10 == 0 or local_iteration == args.max_iterations:
                checkpoint_path = algo.save(str(run_dir / f"ray_{iteration:05d}"))
                print(f"Saved policy checkpoint -> {checkpoint_path.checkpoint.path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving emergency checkpoint...")
        checkpoint_path = algo.save(os.path.join(args.checkpoint_dir, "emergency_checkpoint"))
        print(f"Emergency checkpoint saved -> {checkpoint_path.checkpoint.path}")
    finally:
        algo.stop()
        ray.shutdown()
        print("Ray training session closed cleanly.")


if __name__ == "__main__":
    main()
