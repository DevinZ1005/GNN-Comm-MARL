# Feedback Review and First Implementation

September 18, 2026. AI-assisted engineering record, not a competition submission or student-authored scientific interpretation.

## Review Verdict

The feedback's sequencing is reasonable: resolve or rule out the value-learning issue, repeat communication controls, freeze a manageable experiment, then investigate routing and message importance. Its numerical quality ratings and statements about winning potential or novelty are subjective, not validated evidence.

The source, saved private-goal reports, and installed RLlib loss support the reported development results and the value-loss concern. They do not establish that scaling will fix training, that communication is strictly necessary, or that attention is causally faithful. No final-test scenarios were used in this implementation work.

| Recommendation or claim | Assessment | Decision |
| --- | --- | --- |
| Prioritize critic/value scaling | Supported by the installed loss and terminal replay; cause of instability remains unproven | Implement one opt-in reward multiplier; retain the same value clip |
| Change one treatment at a time | Sound for attributing improvements | Do not simultaneously change architecture, reward ratios, horizon, or loss formula |
| Repeat dense/no-message controls | Required before interpreting the main routing comparison | Use fresh matched training runs, not the smoke test |
| Dense must reach 80% in each seed | Sensible existing engineering gate, not a universal scientific requirement | Keep it for the next pilot unless changed with a recorded prior rationale; retain failing seeds |
| Private goal means information must spread through messages | Too strong | Explicit messages helped these pilots; physical motion can also reveal goal information |
| Freeze eight robots, one observer, independent orientations, robot frame, one layer | Reasonable candidate, not yet a proven stable final architecture | Preserve it while testing scaling; freeze after reliability checks |
| Start with attention/random at K=2 | Reasonable scope control | Defer broad K/depth/noise/Gumbel matrices |
| Causal faithfulness differs from routing performance | Correct | Keep distinct questions, estimands, and experiments |
| Faithfulness was definitely the original primary question | Not established by this conversation's description, which explicitly states attention versus random | Do not silently replace the primary question; retain faithfulness as an extension for the student to choose and justify |
| Correlation with removal loss answers faithfulness | Potentially informative, but underspecified | Define score, intervention, continuation, sampling, and uncertainty first |
| A faithfulness extension establishes novelty | Not demonstrated | Requires independent literature review; no novelty claim adopted |
| Keep reserved scenarios untouched | Sound | Use development seeds only and retain an access record |
| Defer hardware/PyBullet | Reasonable scope decision | Keep current claims kinematic |
| September repair, October routing, November mechanisms | Useful aspiration, not a validated time estimate | Use readiness gates; leave time for student-authored materials |

## First Implemented Change

`--reward-scale` now exists in `train.py` and `run_sweep.py`. The default is **1.0**, preserving historical reward magnitude. A fresh run can explicitly choose **0.01** as a candidate treatment. Nonpositive or nonfinite scales are rejected before training starts.

The multiplier applies after every reward term, including terminal completion. It does not change reset RNG, observations, graph, physics, action clipping, success, or truncation. Per-agent info contains raw/scaled reward components, raw reward, learning reward, and scale. Evaluation keeps `return_per_agent` as the learning return and adds `raw_return_per_agent`. Existing episode component totals remain raw.

Scale is captured in environment configuration, arguments, schedules, and exports. Resume rejects changes to scale, relevant task/model source, recorded optimizer settings, and the installed value-loss source. Old portable policies still load; an absent scale means 1.0. Resuming historical runs through changed source remains intentionally blocked.

The trainer explicitly pins the existing value-loss clip at **10.0**; it does not increase it or alter the loss. Manifests and portable training protocols record optimizer settings and the installed PPO policy-source SHA-256. JSONL now persists available RLlib learner statistics: total/policy/value losses, explained variance, entropy, KL, learning rate, KL coefficient, and entropy coefficient. Missing/nonfinite values are null, not zero or invalid JSON.

**Still missing:** unclipped value loss and saturation fraction on actual optimizer minibatches, policy clip fraction, a complete counterfactual rollout system, and matched full-length scaling experiments. The available `vf_loss` is already clipped and cannot alone reveal the fraction of saturated samples.

## Verification

- Main regression suite: **31 tests passed**, up from 25; all four random-routing acceptance checks also passed.
- Edited executable modules compiled.
- Tests compare default scale, explicit 1.0, and 0.01 through matching successful and truncated episodes for both reward versions.
- Coverage includes observation arrays, payload positions, raw/scaled arithmetic, terminal bonus, invalid settings, sweep forwarding, export/evaluation compatibility, and pre-Ray rejection of scale changes on resume.
- A terminal-target check differentiates the installed loss expression at V=0: the raw target saturates and the scaled target does not. This is not a trained-critic reliability result.
- An actual one-iteration Ray/PPO smoke run completed with finite available learner statistics, saved checkpoints, and clean shutdown.
- An unchanged-configuration restore completed iteration 2 (256 cumulative environment steps) and shut down cleanly. This is continuation of the same software check, not an independent seed.
- The historical private-goal dense-104 policy reproduced every saved non-timing metric on the two checked development scenarios at the default scale.
- The archived source files match the smoke manifest's source hashes.

Smoke artifacts: [manifest](pilot_runs/reward_scale_smoke_20260918/manifest.json), [metrics](pilot_runs/reward_scale_smoke_20260918/metrics.jsonl). Settings: seed 106, eight robots, one observer, independent orientations, robot frame, dense one-layer GNN, scale 0.01, zero remote workers, batch 128, one epoch, 128 environment steps, 20-step horizon, validation 80000-80001. Both short episodes failed. This is a software check, not evidence of improved transport.

Preserved [source snapshot](pilot_runs/reward_scale_smoke_20260918/source_snapshot.tar.gz) and [resumed metrics](pilot_runs/reward_scale_smoke_20260918_resumed/metrics.jsonl) support reproduction. The resumed policy also had zero successes in its two short validation episodes.

## Next Controlled Development Check

Add actual-minibatch value-error diagnostics or an equivalently audited instrumentation path before attributing a learning improvement to reduced saturation. Preserve the original loss and test that instrumentation does not change outputs or gradients.

A candidate development design is a 2-by-2 check: scale 1.0 versus 0.01, each with dense and trained no-message actors, on the same fresh seed list. Keep the prior 250-step task, 80 iterations / 40,960 environment steps, batch 512, two epochs, four workers, one GNN layer, and final-scheduled-checkpoint rule. Raw and scaled treatments should share the new source version. Predetermine seeds, gates, and failed-run handling before launch; two seeds would still be preliminary.

The strict same-protocol comparator should continue rejecting reward-scale differences. A separate audited intervention comparison must allow only the intended difference. Do not edit old manifests or globally relax matching checks.

If scaling improves reliability and communication still helps, replicate before freezing the main routing experiment. If it does not, preserve the negative result and inspect critic errors, update size, reachability, and trajectories. Do not assume flat value loss was the only failure mechanism.

## Requirements for a Later Faithfulness Study

`C_ij = normal_future_return - intervened_future_return` can define a simulator intervention effect. It is not a universal causal property of a message. Its meaning depends on these choices:

1. **Score:** distinguish pre-selection ranking, normalized attention after masking, and per-head versus aggregated scores. Normalized weights across different neighborhood sizes need not be directly comparable.
2. **Intervention:** one message at one step, persistent edge removal, replacement, and pre-selection candidate removal differ. Zeroing a message with fixed weights differs from renormalizing. Replacement can preserve K; deletion does not.
3. **State/randomness:** paired branches require identical full starting state and future exogenous draws, including routing randomness. Preserve RNG and policy/history state. A no-op branch must exactly reproduce baseline behavior. Start with tested kinematic branching rather than arbitrary backend cloning.
4. **Outcome:** define continuation horizon, discount, terminal handling, and whether the outcome is raw return, success, or distance. Scaling must not accidentally redefine message value.
5. **Sampling:** selected-only analysis applies to retained messages, not all candidates. Fix sampling before observing effects; handle ties and zero effects explicitly.
6. **Dependence:** edges share states, states share episodes, and episodes share trained policies. Thousands of edges are not thousands of learning replications. Use within-state ranking measures and clustered/seed-level uncertainty as appropriate.
7. **Interactions:** individually redundant messages can be jointly essential. Single-edge deletion is incomplete; a negative effect can mean the message was harmful in that context.
8. **Distribution shift:** removal may be off-distribution. Describe a policy-specific simulated intervention, not real-world causal validity.

Concrete code detail: `evaluate.action_sensitivity` currently records `attention_score` from `last_attention_scores`, the raw mean-over-head ranking score. The layer separately stores normalized per-head `last_attention_weights`. The existing diagnostic's score is not an alpha probability such as 0.82; a faithfulness implementation must choose and label these quantities explicitly.

Attention/random training compares learning under routing regimes. Fixed-policy interventions test reliance or usefulness within a trained policy. Neither proves the other. Keep the three-experiment structure as a scope-controlled option, not a guaranteed stronger result or a requirement to manufacture an attention advantage.

## Git Status

`git status` reports that this directory is not a Git repository. Its `.git` directory is empty and read-only here. No Git commit was created, no repository was reinitialized, and no history was replaced. Changes are in the working files; actual commits require access to the real writable repository metadata. Suggested grouping once available: reward-scaling/configuration/tests, then review/documentation.
