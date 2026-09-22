"""Regression tests for scientific controls; run with python -m unittest test_research."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import json
import shlex
import subprocess
import sys

import numpy as np
import torch

from env_core import MultiRobotPhysicsEnv
from gnn_comm_layer import EdgeConditionedGATLayer
from marl_agent import GNNMARLModel, robot_frame_features
from evaluate import evaluate, load_export, observation_batch, action_sensitivity
from compare_evaluations import paired_comparison
from validate_task import validate
from train import learner_metrics


class EnvironmentTests(unittest.TestCase):
    def make_env(self, **config):
        env = MultiRobotPhysicsEnv(dict(backend="kinematic", **config))
        self.addCleanup(env.close)
        return env

    def test_seed_isolation(self):
        a, b = self.make_env(), self.make_env()
        initial, _ = a.reset(seed=7)
        expected, _ = b.reset(seed=7)
        for key in initial["robot_0"]:
            np.testing.assert_array_equal(initial["robot_0"][key], expected["robot_0"][key])
        a.reset(seed=888)
        b_obs = b.step({})[0]
        c = self.make_env()
        c.reset(seed=7)
        c_obs = c.step({})[0]
        np.testing.assert_array_equal(b_obs["robot_0"]["random_comm_mask"], c_obs["robot_0"]["random_comm_mask"])

    def test_stationary_has_no_positive_reward(self):
        env = self.make_env(spawn_radius=0.8)
        env.reset(seed=0)
        _, rewards, _, _, _ = env.step({})
        self.assertTrue(all(value < 0 for value in rewards.values()))

    def test_reward_scale_rejects_invalid_values(self):
        for scale in (0, -1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, "reward_scale"):
                self.make_env(reward_scale=scale)

    def test_reward_scale_preserves_physics_components_and_episode_endings(self):
        for version in ("transport_v2", "legacy"):
            for success in (True, False):
                with self.subTest(version=version, success=success):
                    config = dict(reward_version=version, spawn_radius=0.6, spawn_jitter=0,
                                  goal_distance=2, max_steps=100 if success else 3)
                    envs = [self.make_env(**config), self.make_env(**config, reward_scale=1),
                            self.make_env(**config, reward_scale=0.01)]
                    observations = [env.reset(seed=4)[0] for env in envs]
                    for other in observations[1:]:
                        for agent in other:
                            for key in other[agent]:
                                np.testing.assert_array_equal(other[agent][key], observations[0][agent][key])
                    for env in envs:
                        env.robot_headings[:] = np.arctan2(env.goal_pos[1], env.goal_pos[0])
                    for _ in range(config["max_steps"]):
                        actions = {agent: [1, 1] for agent in envs[0]._agent_ids}
                        results = [env.step(actions) for env in envs]
                        obs, rewards, term, trunc, infos = results[0]
                        for index, (other_obs, other_rewards, other_term, other_trunc, other_info) in enumerate(results[1:], 1):
                            self.assertEqual(other_term, term)
                            self.assertEqual(other_trunc, trunc)
                            np.testing.assert_array_equal(envs[index]._get_payload_position(), envs[0]._get_payload_position())
                            for agent in obs:
                                for key in obs[agent]:
                                    np.testing.assert_array_equal(other_obs[agent][key], obs[agent][key])
                                self.assertEqual(other_info[agent]["raw_reward_components"], infos[agent]["raw_reward_components"])
                                self.assertEqual(other_info[agent]["raw_reward"], rewards[agent])
                                self.assertEqual(other_rewards[agent], rewards[agent] * envs[index].reward_scale)
                                self.assertEqual(other_info[agent]["learning_reward"], other_rewards[agent])
                                self.assertAlmostEqual(sum(other_info[agent]["raw_reward_components"].values()), rewards[agent])
                                self.assertAlmostEqual(sum(other_info[agent]["scaled_reward_components"].values()), other_rewards[agent])
                        if term["__all__"] or trunc["__all__"]:
                            break
                    self.assertEqual(term["__all__"], success)
                    self.assertEqual(trunc["__all__"], not success)
                    self.assertEqual(infos["robot_0"]["raw_reward_components"]["completion"], 100 if success else 0)
                    if success and version == "transport_v2":
                        # Checks target scale at V=0, not a claim about trained predictions.
                        for row, expected_zero in ((results[0], True), (results[2], False)):
                            value = torch.tensor(0.0, requires_grad=True)
                            loss = (value - row[1]["robot_0"]).square().clamp(0, 10)
                            loss.backward()
                            self.assertEqual(value.grad.item() == 0, expected_zero)

    def test_action_clipping_and_elapsed_time(self):
        a, b = self.make_env(), self.make_env()
        a.reset(seed=1)
        b.reset(seed=1)
        a.step({agent: [100, 100] for agent in a._agent_ids})
        b.step({agent: [1, 1] for agent in b._agent_ids})
        np.testing.assert_allclose(a.robot_positions, b.robot_positions)
        self.assertAlmostEqual(float(np.linalg.norm(a.robot_velocities[0])), 1.5, places=5)

    def test_mask_scope(self):
        for scope in ("step", "episode"):
            env = self.make_env(random_mask_scope=scope)
            before, _ = env.reset(seed=2)
            after = env.step({})[0]
            same = np.array_equal(before["robot_0"]["random_comm_mask"], after["robot_0"]["random_comm_mask"])
            self.assertEqual(same, scope == "episode")

    def test_scripted_transport_succeeds(self):
        # A feasibility check for the simplified drag model, not a learned policy.
        env = self.make_env(num_robots=4, spawn_radius=0.6, spawn_jitter=0,
                            goal_distance=2, max_steps=100)
        env.reset(seed=4)
        heading = np.arctan2(env.goal_pos[1], env.goal_pos[0])
        env.robot_headings[:] = heading
        for _ in range(env.max_steps):
            _, _, term, trunc, infos = env.step({a: [1, 1] for a in env._agent_ids})
            if term["__all__"] or trunc["__all__"]:
                break
        self.assertTrue(term["__all__"])
        self.assertFalse(trunc["__all__"])
        self.assertTrue(infos["robot_0"]["success"])

    def test_default_task_is_feasible_but_not_trivial(self):
        result = validate({"backend": "kinematic", "num_robots": 8,
                           "comm_radius": 1.8}, episodes=5, seed_start=91000)
        self.assertTrue(all(result["acceptance"].values()))

    def test_private_goal_observation(self):
        env = self.make_env(num_robots=8, goal_observers=1)
        obs, _ = env.reset(seed=3)
        matrix = obs["robot_0"]["node_features"]
        self.assertEqual(int(matrix[:, 26].sum()), 1)
        self.assertEqual(int(np.count_nonzero(np.linalg.norm(matrix[:, 18:21], axis=1))), 1)

    def test_goal_scenarios_vary(self):
        env = self.make_env()
        env.reset(seed=1)
        goal = env.goal_pos.copy()
        env.reset(seed=2)
        self.assertFalse(np.array_equal(goal, env.goal_pos))

    def test_goal_spawn_independence_and_legacy_replay(self):
        default = self.make_env(num_robots=8)
        coupled = self.make_env(num_robots=8, goal_spawn_mode="coupled")
        original, _ = default.reset(seed=81)
        explicit, _ = coupled.reset(seed=81)
        for key in original["robot_0"]:
            np.testing.assert_array_equal(original["robot_0"][key], explicit["robot_0"][key])
        env = self.make_env(num_robots=8, goal_spawn_mode="independent", spawn_jitter=0,
                            goal_observers=1)
        offsets = []
        for seed in range(128):
            obs, _ = env.reset(seed=seed)
            goal_angle = np.arctan2(env.goal_pos[1], env.goal_pos[0])
            ring_angle = np.arctan2(env.robot_positions[0, 1], env.robot_positions[0, 0])
            offsets.append(np.exp(1j * (goal_angle - ring_angle)))
            features = obs["robot_0"]["node_features"]
            self.assertTrue(np.all(features[features[:, 26] == 0, 18:21] == 0))
        # The old cue gives a fixed zero offset (resultant magnitude one).
        self.assertLess(abs(np.mean(offsets)), 0.2)
        a, _ = env.reset(seed=17)
        b, _ = env.reset(seed=17)
        np.testing.assert_array_equal(a["robot_0"]["node_features"], b["robot_0"]["node_features"])

    def test_transport_contribution_has_correct_direction(self):
        toward, away = self.make_env(spawn_jitter=0), self.make_env(spawn_jitter=0)
        toward.reset(seed=8)
        away.reset(seed=8)
        heading = np.arctan2(toward.goal_pos[1], toward.goal_pos[0])
        toward.robot_headings[:] = heading
        away.robot_headings[:] = heading + np.pi
        action = {agent: [1, 1] for agent in toward._agent_ids}
        toward_info = toward.step(action)[4]["robot_0"]
        away_info = away.step(action)[4]["robot_0"]
        self.assertGreater(toward_info["transport_contribution"], 0)
        self.assertLess(away_info["transport_contribution"], 0)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.nodes = torch.randn(1, 4, 16)
        self.edges = torch.randn(1, 4, 4, 8)
        self.adj = torch.tensor([[[0, 1, 1, 1], [1, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0]]]).float()
        self.random = torch.rand(1, 4, 4)

    def layer(self, mode, k):
        return EdgeConditionedGATLayer(node_dim=16, edge_dim=8, hidden_dim=16,
                                      num_heads=2, top_k=k, topk_mode=mode)

    def test_budget_isolation_finite_and_replay(self):
        for mode in ("attention", "random", "distance", "gumbel"):
            for k in (0, 1, 2, 8, None):
                with self.subTest(mode=mode, k=k):
                    layer = self.layer(mode, k)
                    output, _ = layer(self.nodes, self.adj, self.edges, self.random)
                    output2, _ = layer(self.nodes, self.adj, self.edges, self.random)
                    torch.testing.assert_close(output, output2, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(output).all())
                    counts = layer.last_selected_mask.sum(2)
                    expected = self.adj.sum(2) if k is None else self.adj.sum(2).clamp_max(k)
                    torch.testing.assert_close(counts.float(), expected)
                    self.assertFalse((layer.last_selected_mask & ~self.adj.bool()).any())
                    output.square().sum().backward()
                    self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in layer.parameters()))

    def test_distance_selects_nearest(self):
        layer = self.layer("distance", 1)
        self.edges[0, 0, :, 6] = torch.tensor([0, 3, 1, 2])
        layer(self.nodes, self.adj, self.edges, self.random)
        self.assertTrue(layer.last_selected_mask[0, 0, 2])

    def test_gumbel_unselected_score_gradient(self):
        layer = self.layer("gumbel", 1)
        edges = self.edges.clone().requires_grad_()
        output, _ = layer(self.nodes, self.adj, edges, self.random)
        output[0, 0, 0].backward()
        unselected = self.adj.bool() & ~layer.last_selected_mask
        self.assertGreater(float(edges.grad[unselected].abs().sum()), 0)

    def test_fixed_topk_mask_across_layers(self):
        env = MultiRobotPhysicsEnv({"backend": "kinematic", "num_robots": 8, "comm_radius": 3.8})
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=2)
        for mode in ("random", "attention", "gumbel"):
            model = GNNMARLModel(None, None, 4, {"custom_model_config": {"top_k": 2, "topk_mode": mode}}, "test")
            model.forward({"obs": observation_batch(obs)}, [], None)
            layers = model.gnn_layer.gat_layers
            torch.testing.assert_close(layers[0].last_selected_mask, layers[1].last_selected_mask)


class SweepTests(unittest.TestCase):
    def test_matched_schedule_and_private_goal_guard(self):
        with tempfile.TemporaryDirectory() as root:
            base = [sys.executable, str(Path(__file__).with_name("run_sweep.py")),
                    "--root", root, "--backend", "kinematic", "--seeds", "104", "105",
                    "--conditions", "dense", "no_comm", "--observation-frame", "robot",
                    "--iterations", "40", "--max-steps", "250", "--train-batch-size", "512",
                    "--epochs", "2", "--gnn-num-layers", "1", "--development-episodes", "50",
                    "--reward-scale", "0.01"]
            result = subprocess.run(base, text=True, capture_output=True, check=True)
            commands = [shlex.split(line) for line in result.stdout.splitlines()]
            training = [cmd for cmd in commands if any(v.endswith("train.py") for v in cmd)]
            self.assertEqual(len(training), 4)
            self.assertEqual(len(commands), 8)
            for cmd in training:
                self.assertEqual(cmd[cmd.index("--observation-frame") + 1], "robot")
                self.assertEqual(cmd[cmd.index("--max-steps") + 1], "250")
                self.assertEqual(cmd[cmd.index("--train-batch-size") + 1], "512")
                self.assertEqual(cmd[cmd.index("--reward-scale") + 1], "0.01")
            self.assertEqual(sum("--no-comm" in cmd for cmd in training), 2)
            rejected = subprocess.run(base + ["--goal-observers", "1"], text=True, capture_output=True)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("private-goal sweeps require", rejected.stderr)
            accepted = subprocess.run(base + ["--goal-observers", "1", "--goal-spawn-mode", "independent"],
                                      text=True, capture_output=True)
            self.assertEqual(accepted.returncode, 0)
            self.assertEqual(list(Path(root).iterdir()), [])
            for value in ("0", "nan", "inf"):
                invalid = subprocess.run(base + ["--reward-scale", value], text=True, capture_output=True)
                self.assertNotEqual(invalid.returncode, 0)
                self.assertIn("reward-scale must be finite and positive", invalid.stderr)


class TrainingTests(unittest.TestCase):
    def test_learner_metrics_preserve_zero_and_null_nonfinite(self):
        result = {"info": {"learner": {"shared_gnn_policy": {"learner_stats": {
            "vf_loss": 0, "vf_explained_var": float("nan"), "kl": float("inf"), "cur_lr": 3e-4}}}}}
        stats = learner_metrics(result)
        self.assertEqual(stats["vf_loss"], 0.0)
        self.assertEqual(stats["cur_lr"], 3e-4)
        self.assertIsNone(stats["vf_explained_var"])
        self.assertIsNone(stats["kl"])
        self.assertIsNone(stats["entropy"])
        self.assertTrue(all(value is None for value in learner_metrics({}).values()))
        json.dumps(stats, allow_nan=False)

    def test_invalid_training_scale_rejected_before_ray(self):
        import train
        for value in ("0", "nan", "inf", "-1"):
            with patch.object(sys, "argv", ["train.py", "--reward-scale", value]), patch.object(train.ray, "init") as init:
                with self.assertRaisesRegex(ValueError, "Invalid training configuration"):
                    train.main()
                init.assert_not_called()

    def test_resume_rejects_reward_scale_change_before_ray(self):
        import hashlib
        import inspect
        import train
        with tempfile.TemporaryDirectory() as root:
            checkpoint = Path(root) / "ray_00001"
            checkpoint.mkdir()
            argv = ["train.py", "--backend", "kinematic", "--resume-from", str(checkpoint),
                    "--checkpoint-dir", str(Path(root) / "continued"), "--reward-scale", "0.01"]
            with patch.object(sys, "argv", argv):
                args = vars(train.parse_arguments())
                previous = {"arguments": dict(args, reward_scale=1.0),
                    "source_hashes": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ("env_core.py", "gnn_comm_layer.py", "marl_agent.py")},
                    "value_loss_source_sha256": hashlib.sha256(Path(inspect.getfile(train.PPOTorchPolicy)).read_bytes()).hexdigest()}
                (Path(root) / "manifest.json").write_text(json.dumps(previous))
                with patch.object(train.ray, "init") as init:
                    with self.assertRaisesRegex(ValueError, "reward_scale"):
                        train.main()
                    init.assert_not_called()


class EvaluationTests(unittest.TestCase):
    def test_reward_scale_export_and_evaluation_compatibility(self):
        config = {"observation_frame": "robot", "gnn_num_layers": 1}
        model = GNNMARLModel(None, None, 4, {"custom_model_config": config}, "test")
        env_config = {"backend": "kinematic", "max_steps": 5}
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for scale in (None, 1.0, 0.01):
                saved_config = dict(env_config)
                if scale is not None:
                    saved_config["reward_scale"] = scale
                path = Path(tmp) / f"policy_{scale}.pt"
                torch.save({"format_version": 2, "weights": model.state_dict(),
                            "model_config": config, "active_top_k": None, "env_config": saved_config}, path)
                restored, payload = load_export(path)
                episodes, _ = evaluate(restored, payload["env_config"], [80001])
                row = episodes[0]
                self.assertAlmostEqual(row["return_per_agent"], row["raw_return_per_agent"] * (scale or 1.0))
                row.pop("return_per_agent")
                row.pop("inference_seconds")
                rows.append(row)
            self.assertEqual(rows[0], rows[1])
            self.assertEqual(rows[0], rows[2])

    def test_hidden_goal_only_reaches_actors_through_local_observation_or_edges(self):
        torch.manual_seed(23)
        env = MultiRobotPhysicsEnv({"backend": "kinematic", "num_robots": 8,
            "goal_observers": 1, "goal_spawn_mode": "independent"})
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=77)
        before = observation_batch(obs)
        observer = int(np.flatnonzero(env.goal_observer_mask)[0])
        uninformed = before["local_obs"][:, 26] == 0
        reachable = before["adj_matrix"][0, :, observer].bool()
        self.assertTrue((uninformed & reachable).any())
        self.assertTrue((uninformed & ~reachable).any())

        # Counterfactual snapshot: only the private goal changes, no physics step.
        env.goal_pos[:2] *= -1
        features = env._extract_all_node_features()
        for i, agent in enumerate(obs):
            obs[agent].update(local_obs=features[i].copy(), node_features=features.copy())
        after = observation_batch(obs)
        torch.testing.assert_close(before["local_obs"][uninformed], after["local_obs"][uninformed])

        for no_comm, remove_edges in ((True, False), (False, True), (False, False)):
            model = GNNMARLModel(None, None, 4, {"custom_model_config": {
                "observation_frame": "robot", "gnn_num_layers": 1, "no_comm": no_comm}}, "test")
            model.eval()
            a, b = dict(before), dict(after)
            if remove_edges:
                a["adj_matrix"] = torch.zeros_like(before["adj_matrix"])
                b["adj_matrix"] = torch.zeros_like(after["adj_matrix"])
            original, _ = model.forward({"obs": a}, [], None)
            changed, _ = model.forward({"obs": b}, [], None)
            protected = uninformed if no_comm or remove_edges else uninformed & ~reachable
            torch.testing.assert_close(original[protected], changed[protected], atol=1e-7, rtol=1e-6)
            if not no_comm and not remove_edges:
                self.assertGreater(float((original[uninformed & reachable] -
                                          changed[uninformed & reachable]).detach().abs().max()), 1e-6)

    def test_entropy_uses_each_actors_routing_rows(self):
        import analyze_attention_entropy as analysis
        torch.manual_seed(44)
        config = {"observation_frame": "robot", "top_k": 1, "gnn_num_layers": 1}
        model = GNNMARLModel(None, None, 4, {"custom_model_config": config}, "test")
        env_config = {"backend": "kinematic", "num_robots": 8, "max_steps": 3}
        expected = np.zeros((8, 8), dtype=int)

        def record(*args):
            layer = model.gnn_layer.gat_layers[0]
            rows = torch.arange(8)
            expected[:] += layer.last_selected_mask[rows, rows].cpu().numpy()

        evaluate(model, env_config, [18], trajectory_callback=record)
        payload = {"env_config": env_config, "model_config": config,
                   "training_seed": 44, "iteration": 0}
        with patch.object(analysis, "load_export", return_value=(model, payload)):
            result = analysis.analyze("unused.pt", episodes=1, seed_start=18)
        np.testing.assert_array_equal(result["selected_counts"], expected)
        self.assertEqual(result["model_config"]["observation_frame"], "robot")

    def test_robot_frame_rotation_translation_and_private_goals(self):
        env = MultiRobotPhysicsEnv({"backend": "kinematic", "num_robots": 4,
                                    "goal_observers": 1})
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=14)
        original = observation_batch(obs)
        framed = robot_frame_features(original["local_obs"], original["node_features"],
                                      original["edge_features"])
        self.assertTrue(torch.all(framed[0][:, :2] == 0))
        self.assertTrue(torch.all(framed[0][original["local_obs"][:, 26] == 0, 18:21] == 0))
        angle = 1.37
        rotation = np.array([[np.cos(angle), -np.sin(angle)],
                             [np.sin(angle), np.cos(angle)]], dtype=np.float32)
        for positions in (env.robot_positions, env.payload_pos[None], env.goal_pos[None]):
            positions[:, :2] = positions[:, :2] @ rotation.T + [2.0, -3.0]
        env.robot_velocities[:, :2] = env.robot_velocities[:, :2] @ rotation.T
        env.robot_headings += angle
        env._update_kinematics_and_graph()
        features = env._extract_all_node_features()
        for i, agent in enumerate(obs):
            obs[agent].update(local_obs=features[i].copy(), node_features=features.copy(),
                              adj_matrix=env.adj_matrix.copy(), edge_features=env.edge_features.copy())
        rotated = observation_batch(obs)
        for mode, top_k, no_comm in (("attention", None, False), ("attention", 2, False),
                                    ("random", 2, False), ("distance", 2, False),
                                    ("gumbel", 2, False), ("attention", None, True)):
            model = GNNMARLModel(None, None, 4, {"custom_model_config": {
                "observation_frame": "robot", "top_k": top_k, "topk_mode": mode,
                "no_comm": no_comm}}, "test")
            model.eval()
            before, _ = model.forward({"obs": original}, [], None)
            value = model.value_function().clone()
            after, _ = model.forward({"obs": rotated}, [], None)
            torch.testing.assert_close(before, after, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(value, model.value_function(), atol=2e-6, rtol=2e-5)
            (after.square().mean() + model.value_function().square().mean()).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()
                                if p.grad is not None))

    def test_trajectory_recording_and_horizon_prefix(self):
        from diagnose_transport import trajectory_row, summarize_trajectory
        model = GNNMARLModel(None, None, 4, {"custom_model_config": {}}, "test")
        config = {"backend": "kinematic", "max_steps": 3}
        plain, _ = evaluate(model, config, [12])
        traces = []
        recorded, _ = evaluate(model, config, [12], trajectory_callback=lambda *args:
                               traces.append(trajectory_row(*args)))
        plain[0].pop("inference_seconds")
        recorded[0].pop("inference_seconds")
        self.assertEqual(plain, recorded)
        longer = []
        evaluate(model, dict(config, max_steps=6), [12], trajectory_callback=lambda *args:
                 longer.append(trajectory_row(*args)))
        self.assertEqual(traces, longer[:3])
        self.assertEqual([r["step"] for r in traces], [1, 2, 3])
        summary = summarize_trajectory(traces, False, 2)
        self.assertEqual(summary["final_distance"], recorded[0]["final_payload_distance"])
        json.dumps(summary, allow_nan=False)

    def test_paired_statistics_and_reject_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            left, right = [], []
            for seed in (0, 1):
                for group, value, paths in (("left", 1, left), ("right", 0, right)):
                    data = {"training_seed": seed, "env_config": {}, "protocol": {},
                            "iteration": 10, "env_steps": 100, "training_protocol": {},
                            "source_hashes": {}, "evaluation_source_hashes": {}, "model_config": {},
                            "episodes": [{"episode_seed": 100, "success": value}]}
                    path = Path(tmp) / f"{group}{seed}.json"
                    path.write_text(json.dumps(data))
                    paths.append(path)
            result = paired_comparison(left, right, "success")
            self.assertEqual(result["mean_difference"], 1)
            self.assertEqual(result["seed_ids"], [0, 1])
            self.assertEqual(result["left_means"], [1, 1])
            self.assertEqual(result["bootstrap_95_interval"], [1, 1])
            data = json.loads(right[0].read_text())
            data["env_steps"] = 99
            right[0].write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                paired_comparison(left, right, "success")

    def test_comparison_rejects_cross_seed_architecture_and_scenario_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            groups = [[], []]
            for seed in (0, 1):
                for condition in (0, 1):
                    data = {"training_seed": seed, "env_config": {}, "protocol": {},
                            "iteration": 40, "env_steps": 20480, "training_protocol": {},
                            "source_hashes": {}, "evaluation_source_hashes": {},
                            "model_config": {"observation_frame": "robot" if seed == 0 else "world"},
                            "episodes": [{"episode_seed": 80000, "success": True}]}
                    path = Path(tmp) / f"{condition}_{seed}.json"
                    path.write_text(json.dumps(data))
                    groups[condition].append(path)
            with self.assertRaisesRegex(ValueError, "Mixed model architecture"):
                paired_comparison(*groups, "success")
            for path in (groups[0][1], groups[1][1]):
                data = json.loads(path.read_text())
                data["model_config"]["observation_frame"] = "robot"
                data["episodes"][0]["episode_seed"] = 90000
                path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Mixed evaluation episode seeds"):
                paired_comparison(*groups, "success")

    def test_robustness_and_larger_team(self):
        model = GNNMARLModel(None, None, 4, {"custom_model_config": {"top_k": 2}}, "test")
        rows, _ = evaluate(model, {"backend": "kinematic", "num_robots": 12, "max_steps": 3},
                           [10], candidate_loss=1, delay=2, noise=0.1)
        self.assertEqual(rows[0]["logical_deliveries"], 0)
        self.assertTrue(np.isfinite(rows[0]["return_per_agent"]))

    def test_export_roundtrip_and_repeatability(self):
        config = {"top_k": 1, "topk_mode": "gumbel", "observation_frame": "robot"}
        model = GNNMARLModel(None, None, 4, {"custom_model_config": config}, "test")
        env_config = {"backend": "kinematic", "num_robots": 4, "max_steps": 3}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "policy.pt"
            torch.save({"format_version": 2, "weights": model.state_dict(),
                        "model_config": config, "active_top_k": 1}, path)
            restored, _ = load_export(path)
            a, _ = evaluate(restored, env_config, [100])
            b, _ = evaluate(restored, env_config, [100])
            a[0].pop("inference_seconds")
            b[0].pop("inference_seconds")
            self.assertEqual(a, b)
            self.assertLessEqual(a[0]["logical_deliveries"], 4 * 1 * 2 * 3)

    def test_no_comm_actor_invariance_and_critic_information(self):
        model = GNNMARLModel(None, None, 4, {"custom_model_config": {"no_comm": True}}, "test")
        env = MultiRobotPhysicsEnv({"backend": "kinematic"})
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=0)
        batch = observation_batch(obs)
        before, _ = model.forward({"obs": batch}, [], None)
        old_value = model.value_function().clone()
        batch["node_features"] += 20
        after, _ = model.forward({"obs": batch}, [], None)
        torch.testing.assert_close(before, after)
        self.assertFalse(torch.equal(old_value, model.value_function()))

    def test_ablation_clears_intervention(self):
        model = GNNMARLModel(None, None, 4, {"custom_model_config": {
            "top_k": 1, "observation_frame": "robot"}}, "test")
        env = MultiRobotPhysicsEnv({"backend": "kinematic"})
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=0)
        batch = observation_batch(obs)
        before, _ = model.forward({"obs": batch}, [], None)
        layer = model.gnn_layer.gat_layers[0]
        expected = {(i, j): float(layer.last_attention_scores[i, i, j])
                    for i in range(env.num_robots)
                    for j in range(env.num_robots) if layer.last_selected_mask[i, i, j]}
        rows = action_sensitivity(model, batch)
        self.assertTrue(rows)
        self.assertEqual({(r["receiver"], r["sender"]): r["attention_score"] for r in rows}, expected)
        after, _ = model.forward({"obs": batch}, [], None)
        torch.testing.assert_close(before, after)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
