"""Regression tests for scientific controls; run with python -m unittest test_research."""
import tempfile
from pathlib import Path
import unittest
import json

import numpy as np
import torch

from env_core import MultiRobotPhysicsEnv
from gnn_comm_layer import EdgeConditionedGATLayer
from marl_agent import GNNMARLModel
from evaluate import evaluate, load_export, observation_batch, action_sensitivity
from compare_evaluations import paired_comparison
from validate_task import validate


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


class EvaluationTests(unittest.TestCase):
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
            self.assertEqual(result["bootstrap_95_interval"], [1, 1])
            data = json.loads(right[0].read_text())
            data["env_steps"] = 99
            right[0].write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                paired_comparison(left, right, "success")

    def test_robustness_and_larger_team(self):
        model = GNNMARLModel(None, None, 4, {"custom_model_config": {"top_k": 2}}, "test")
        rows, _ = evaluate(model, {"backend": "kinematic", "num_robots": 12, "max_steps": 3},
                           [10], candidate_loss=1, delay=2, noise=0.1)
        self.assertEqual(rows[0]["logical_deliveries"], 0)
        self.assertTrue(np.isfinite(rows[0]["return_per_agent"]))

    def test_export_roundtrip_and_repeatability(self):
        config = {"top_k": 1, "topk_mode": "gumbel"}
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
        model = GNNMARLModel(None, None, 4, {"custom_model_config": {"top_k": 1}}, "test")
        env = MultiRobotPhysicsEnv({"backend": "kinematic"})
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=0)
        batch = observation_batch(obs)
        before, _ = model.forward({"obs": batch}, [], None)
        rows = action_sensitivity(model, batch)
        self.assertTrue(rows)
        after, _ = model.forward({"obs": batch}, [], None)
        torch.testing.assert_close(before, after)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
