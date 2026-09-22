"""Verify instrumentation against RLlib's actual loss and parameter gradients."""
import copy
import unittest

import numpy as np
import torch
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
from ray.rllib.policy.sample_batch import SampleBatch

from env_core import MultiRobotPhysicsEnv
from evaluate import observation_batch
from marl_agent import GNNMARLModel
from ppo_diagnostics import DIAGNOSTIC_KEYS, InstrumentedPPOTorchPolicy
from train import learner_metrics


class LossHarness(InstrumentedPPOTorchPolicy):
    def __init__(self, model):
        self.model = model
        self.config = {"use_critic": True, "vf_clip_param": 10.0, "clip_param": 0.2,
                       "vf_loss_coeff": 0.5, "kl_coeff": 0.2}
        self.entropy_coeff = 0.01
        self.kl_coeff = 0.2
        self.cur_lr = 3e-4
        self._loss_initialized = True

    def get_tower_stats(self, name):
        return [self.model.tower_stats[name]]


class DiagnosticTests(unittest.TestCase):
    def test_actual_loss_gradients_and_rng_are_unchanged(self):
        env = MultiRobotPhysicsEnv({"backend": "kinematic", "num_robots": 4})
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=80003)
        for mode in ("dense", "no_comm", "attention", "random", "gumbel"):
            with self.subTest(mode=mode):
                torch.manual_seed(47)
                config = {"gnn_num_layers": 1, "observation_frame": "robot",
                          "no_comm": mode == "no_comm"}
                if mode not in ("dense", "no_comm"):
                    config.update(top_k=2, topk_mode=mode)
                original = GNNMARLModel(None, None, 4, {
                    "_disable_preprocessor_api": True, "custom_model_config": config}, "test")
                measured = copy.deepcopy(original)
                batch = {"obs": observation_batch(obs)}
                with torch.no_grad():
                    logits, _ = original(batch)
                    values = original.value_function().clone()
                    actions = torch.zeros(4, 2)
                    logp = TorchDiagGaussian(logits, original).logp(actions)
                batch.update({SampleBatch.ACTIONS: actions,
                    SampleBatch.ACTION_DIST_INPUTS: logits.clone(),
                    SampleBatch.ACTION_LOGP: logp - torch.tensor([0.5, 1., 1.5, 1.]).log(),
                    Postprocessing.ADVANTAGES: torch.tensor([1., -1., 0.5, -0.5]),
                    Postprocessing.VALUE_TARGETS: values + torch.tensor([100., 1., 10., 0.]),
                    SampleBatch.TERMINATEDS: torch.tensor([True, False, False, False]),
                    SampleBatch.TRUNCATEDS: torch.tensor([False, False, True, False])})
                plain_policy, measured_policy = LossHarness(original), LossHarness(measured)
                baseline = PPOTorchPolicy.loss(plain_policy, original, TorchDiagGaussian, batch)
                rng_state = torch.get_rng_state().clone()
                result = measured_policy.loss(measured, TorchDiagGaussian, batch)
                torch.testing.assert_close(torch.get_rng_state(), rng_state, rtol=0, atol=0)
                torch.testing.assert_close(result, baseline, rtol=0, atol=0)
                baseline.backward()
                result.backward()
                for (name, a), (_, b) in zip(original.named_parameters(), measured.named_parameters()):
                    self.assertEqual(a.grad is None, b.grad is None, name)
                    if a.grad is not None:
                        torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)
                stats = measured_policy.stats_fn(batch)
                self.assertAlmostEqual(stats["vf_unclipped_loss"], (10000 + 1 + 100) / 4, places=2)
                self.assertEqual(stats["vf_saturation_fraction"], 0.5)
                self.assertEqual(stats["policy_clip_fraction"], 0.5)
                self.assertEqual(stats["vf_terminal_samples_per_minibatch"], 1)
                self.assertEqual(stats["vf_saturated_terminal_samples_per_minibatch"], 1)
                self.assertTrue(all(np.isfinite(stats[key]) for key in DIAGNOSTIC_KEYS))
                self.assertTrue(all(not measured.tower_stats[key].requires_grad for key in DIAGNOSTIC_KEYS))

    def test_terminal_ratio_uses_counts_and_missing_is_null(self):
        def metrics(count, saturated):
            return learner_metrics({"info": {"learner": {"shared_gnn_policy": {"learner_stats": {
                "vf_terminal_samples_per_minibatch": count,
                "vf_saturated_terminal_samples_per_minibatch": saturated}}}}})
        self.assertEqual(metrics(0.5, 0.25)["vf_terminal_saturation_fraction"], 0.5)
        self.assertIsNone(metrics(0, 0)["vf_terminal_saturation_fraction"])
        self.assertIsNone(metrics(1, None)["vf_terminal_saturation_fraction"])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
