"""Guard the clip-only intervention, its CLI, and actual PPO value gradients."""
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.torch.torch_action_dist import TorchDiagGaussian
from ray.rllib.policy.sample_batch import SampleBatch

import train
from compare_evaluations import paired_comparison
from env_core import MultiRobotPhysicsEnv
from evaluate import observation_batch
from marl_agent import GNNMARLModel
from test_ppo_diagnostics import LossHarness


class ValueClipTests(unittest.TestCase):
    def test_empty_nonfinite_and_stalled_training_updates_fail(self):
        valid = {"num_env_steps_sampled_lifetime": 512, "info": {"learner": {
            "shared_gnn_policy": {"learner_stats": {"policy_loss": 0., "vf_loss": 1., "total_loss": .5}}}}}
        self.assertEqual(train.validated_training_steps(valid), 512)
        self.assertEqual(train.validated_training_steps(valid, 256), 512)
        for previous in (512, 1024):
            with self.assertRaisesRegex(RuntimeError, "no valid training progress"):
                train.validated_training_steps(valid, previous)
        with self.assertRaises(RuntimeError):
            train.validated_training_steps({})
        for value in (None, float("nan"), float("inf")):
            broken = copy.deepcopy(valid)
            broken["info"]["learner"]["shared_gnn_policy"]["learner_stats"]["policy_loss"] = value
            with self.assertRaises(RuntimeError):
                train.validated_training_steps(broken)

    def test_default_and_invalid_before_ray(self):
        with patch.object(sys, "argv", ["train.py"]):
            self.assertEqual(train.parse_arguments().vf_clip_param, 10.0)
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(value=value), patch.object(sys, "argv", ["train.py", "--vf-clip-param", value]), patch.object(train.ray, "init") as init:
                with self.assertRaisesRegex(ValueError, "Invalid training"):
                    train.main()
                init.assert_not_called()

    def test_resume_rejects_changed_clip_before_ray(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint = root / "ray_00001"
            checkpoint.mkdir()
            argv = ["train.py", "--resume-from", str(checkpoint), "--checkpoint-dir",
                    str(root / "new"), "--vf-clip-param", "1000000"]
            with patch.object(sys, "argv", argv):
                previous = {"arguments": dict(vars(train.parse_arguments()), vf_clip_param=10.0),
                    "source_hashes": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                        for name in ("env_core.py", "gnn_comm_layer.py", "marl_agent.py", "ppo_diagnostics.py")},
                    "value_loss_source_sha256": hashlib.sha256(Path(inspect.getfile(train.PPOTorchPolicy)).read_bytes()).hexdigest()}
                (root / "manifest.json").write_text(json.dumps(previous))
                with patch.object(train.ray, "init") as init:
                    with self.assertRaisesRegex(ValueError, "vf_clip_param"):
                        train.main()
                    init.assert_not_called()

    def test_sweep_propagates_and_validates_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            command = [sys.executable, "run_sweep.py", "--root", folder, "--backend", "kinematic",
                       "--seeds", "110", "--conditions", "dense", "--vf-clip-param"]
            result = subprocess.run(command + ["1000000"], capture_output=True, text=True, check=True)
            self.assertIn("--vf-clip-param 1000000.0", result.stdout)
            for value in ("0", "-1", "nan", "inf"):
                result = subprocess.run(command + [value], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("vf-clip-param must be finite and positive", result.stderr)

    def test_installed_loss_value_derivative(self):
        torch.manual_seed(83)
        env = MultiRobotPhysicsEnv({"backend": "kinematic", "num_robots": 4})
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=80003)
        model = GNNMARLModel(None, None, 4, {"_disable_preprocessor_api": True,
            "custom_model_config": {"gnn_num_layers": 1}}, "clip_test")
        batch = {"obs": observation_batch(obs)}
        with torch.no_grad():
            logits, _ = model(batch)
            targets = model.value_function().clone() + 100
            actions = torch.zeros(4, 2)
            logp = TorchDiagGaussian(logits, model).logp(actions)
        batch.update({SampleBatch.ACTIONS: actions, SampleBatch.ACTION_DIST_INPUTS: logits,
            SampleBatch.ACTION_LOGP: logp, Postprocessing.ADVANTAGES: torch.zeros(4),
            Postprocessing.VALUE_TARGETS: targets, SampleBatch.TERMINATEDS: torch.ones(4, dtype=torch.bool),
            SampleBatch.TRUNCATEDS: torch.zeros(4, dtype=torch.bool)})
        for cap, expected in ((10., 0.), (1000000., -25.)):
            candidate = copy.deepcopy(model)
            policy = LossHarness(candidate)
            policy.config["vf_clip_param"] = cap
            loss = policy.loss(candidate, TorchDiagGaussian, batch)
            values = candidate.value_function()
            derivative, = torch.autograd.grad(loss, values)
            # vf_loss_coeff=.5, four samples: .5 * 2 * (V-target) / 4.
            torch.testing.assert_close(derivative, torch.full_like(values, expected))
            self.assertEqual(float(candidate.tower_stats["vf_saturation_fraction"]), float(cap == 10))

    def test_comparator_allows_only_declared_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            left, right = [], []
            for seed in (110, 111):
                for group, clip, paths in (("a", 1000000., left), ("b", 10., right)):
                    data = {"training_seed": seed, "env_config": {"reward_scale": 1.0},
                        "model_config": {"no_comm": False, "top_k": None}, "active_top_k": None,
                        "protocol": {}, "iteration": 80, "env_steps": 40960,
                        "training_protocol": {"epochs": 2, "optimization_config": {"vf_clip_param": clip, "lr": 3e-4}},
                        "source_hashes": {"train.py": "fixed"}, "evaluation_source_hashes": {},
                        "episodes": [{"episode_seed": 80000, "success": True}]}
                    path = Path(folder) / f"{group}_{seed}.json"
                    path.write_text(json.dumps(data))
                    paths.append(path)
            with self.assertRaisesRegex(ValueError, "Unmatched training_protocol"):
                paired_comparison(left, right, "success")
            result = paired_comparison(left, right, "success", value_clips=(1000000., 10.))
            self.assertEqual(result["mean_difference"], 0)
            original = json.loads(right[1].read_text())
            changes = [(("env_config", "reward_scale"), .01),
                (("training_protocol", "optimization_config", "lr"), .001),
                (("training_protocol", "optimization_config", "vf_clip_param"), 100.),
                (("training_protocol", "epochs"), 3), (("model_config", "no_comm"), True),
                (("active_top_k",), 2), (("source_hashes", "train.py"), "changed"),
                (("env_steps",), 81920)]
            for keys, value in changes:
                altered = copy.deepcopy(original)
                node = altered
                for key in keys[:-1]:
                    node = node[key]
                node[keys[-1]] = value
                right[1].write_text(json.dumps(altered))
                with self.subTest(keys=keys), self.assertRaises(ValueError):
                    paired_comparison(left, right, "success", value_clips=(1000000., 10.))
            right[1].write_text(json.dumps(original))
            for clips in ((10., 10.), (float("nan"), 10.), (0., 10.)):
                with self.assertRaises(ValueError):
                    paired_comparison(left, right, "success", value_clips=clips)
            with self.assertRaisesRegex(ValueError, "one intervention"):
                paired_comparison(left, right, "success", value_clips=(1000000., 10.), reward_scales=(.01, 1.))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
