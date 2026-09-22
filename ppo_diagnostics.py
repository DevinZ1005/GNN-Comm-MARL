"""Read-only diagnostics on the actual feed-forward PPO loss minibatch.

RLlib averages tower statistics across loss minibatches. Sample-count fields are
therefore mean counts per minibatch, including repeated PPO epochs, not unique
rollout transitions. Ratios of the aggregated terminal counts avoid treating
minibatches without terminal samples as zero terminal saturation.
"""
import torch

from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch


DIAGNOSTIC_KEYS = (
    "vf_unclipped_loss", "vf_error_abs_mean", "vf_saturation_fraction",
    "vf_target_abs_mean", "vf_prediction_abs_mean", "policy_clip_fraction",
    "vf_samples_per_minibatch", "vf_terminal_samples_per_minibatch",
    "vf_saturated_terminal_samples_per_minibatch",
)


class InstrumentedPPOTorchPolicy(PPOTorchPolicy):
    """Use RLlib's loss unchanged, then inspect its cached values and logits."""

    def loss(self, model, dist_class, train_batch):
        if model.get_initial_state():
            raise ValueError("PPO diagnostics currently require a feed-forward model")
        if not self.config["use_critic"]:
            raise ValueError("PPO value diagnostics require use_critic=True")
        loss = super().loss(model, dist_class, train_batch)
        with torch.no_grad():
            values = model.value_function().detach()
            targets = train_batch[Postprocessing.VALUE_TARGETS].detach()
            errors = values - targets
            squared = errors.square()
            saturated = squared > self.config["vf_clip_param"]
            terminal = train_batch[SampleBatch.TERMINATEDS].bool()
            logits = model.last_output().detach()
            distribution = dist_class(logits, model)
            ratio = torch.exp(distribution.logp(train_batch[SampleBatch.ACTIONS])
                              - train_batch[SampleBatch.ACTION_LOGP])
            clipped = (ratio < 1 - self.config["clip_param"]) | (
                ratio > 1 + self.config["clip_param"])
            measurements = (
                squared.mean(), errors.abs().mean(), saturated.float().mean(),
                targets.abs().mean(), values.abs().mean(), clipped.float().mean(),
                values.new_tensor(values.numel()), terminal.float().sum(),
                (terminal & saturated).float().sum(),
            )
            model.tower_stats.update(zip(DIAGNOSTIC_KEYS, measurements))
        return loss

    def stats_fn(self, train_batch):
        stats = super().stats_fn(train_batch)
        for key in DIAGNOSTIC_KEYS:
            stats[key] = torch.stack(self.get_tower_stats(key)).mean().cpu().item()
        return stats
