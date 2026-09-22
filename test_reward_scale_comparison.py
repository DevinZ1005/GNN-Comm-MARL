"""The scale-intervention path must reject every other protocol change."""
import json
from pathlib import Path
import tempfile
import unittest

from compare_evaluations import paired_comparison


class ScaleComparisonTests(unittest.TestCase):
    def test_explicit_intervention_and_all_other_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            left, right = [], []
            for seed in (107, 108):
                for group, scale, paths in (("left", 0.01, left), ("right", 1.0, right)):
                    data = {"training_seed": seed, "env_config": {"reward_scale": scale, "max_steps": 250},
                        "model_config": {"top_k": None, "no_comm": False}, "active_top_k": None,
                        "protocol": {}, "iteration": 80, "env_steps": 40960,
                        "training_protocol": {}, "source_hashes": {"train.py": "frozen"},
                        "evaluation_source_hashes": {}, "episodes": [{"episode_seed": 80000, "success": scale == 0.01}]}
                    path = Path(folder) / f"{group}_{seed}.json"
                    path.write_text(json.dumps(data))
                    paths.append(path)
            with self.assertRaisesRegex(ValueError, "Unmatched env_config"):
                paired_comparison(left, right, "success")
            result = paired_comparison(left, right, "success", reward_scales=(0.01, 1.0))
            self.assertEqual(result["mean_difference"], 1)
            self.assertEqual(result["intervention"]["left"], 0.01)
            original = json.loads(right[1].read_text())
            mutations = [("env_config", "max_steps", 750), ("env_config", "reward_scale", 0.1),
                ("model_config", "top_k", 2), ("source_hashes", "train.py", "changed"),
                ("training_protocol", "epochs", 3)]
            for section, key, value in mutations:
                altered = json.loads(json.dumps(original))
                altered[section][key] = value
                right[1].write_text(json.dumps(altered))
                with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                    paired_comparison(left, right, "success", reward_scales=(0.01, 1.0))
            right[1].write_text(json.dumps(original))
            with self.assertRaisesRegex(ValueError, "raw_return"):
                paired_comparison(left, right, "return_per_agent", reward_scales=(0.01, 1.0))


if __name__ == "__main__":
    unittest.main()
