# Project Status, Change History, and Roadmap to TERRA NYC

**Snapshot:** September 18, 2026. **Target cycle:** TERRA NYC 2027. **Scope:** inherited multi-robot simulator, subsequent AI-assisted engineering, development experiments, remaining implementation work, and research decisions for the student to evaluate.

This is an **AI-assisted engineering handoff**, not a student-authored research plan, paper, abstract, poster, conclusion, or competition submission. Use it to understand and audit the software. Independently decide, document, and explain your scientific reasoning with your sponsor. The fair's AI guidance and disclosure requirements matter; see Section 10 before using any of this material in competition work.

The change history below is reconstructed from the current implementation, saved protocols, reports, manifests, and our work records. It is a functional history, not a verified line-by-line diff against the inherited repository or a claim that every existing line was written during this collaboration. Existing simulation, PPO, and neural-network foundations must retain their original attribution. Saved experiment artifacts take precedence over remembered conversation summaries.

**Later September 18 update:** opt-in uniform reward scaling is implemented and smoke-tested. It is not yet a demonstrated reliability fix. See the [feedback review](FEEDBACK_REVIEW_20260918.md) for the recommendation audit, implementation details, and requirements for a separate causal-faithfulness extension. Historical experiment results below are unchanged.

## Contents

**September 21 update:** actual PPO minibatch diagnostics and a fixed eight-run
reward-scale pilot are complete, with one explicitly documented evaluation-only
recovery and missing final-iteration diagnostics. Dense success at scale 1.0 was
88%/48%; at scale 0.01 it was 12%/30% (seeds 107/108). Lower saturation did not
improve transport. Keep scale 1.0; reliability still fails the two-seed gate.
See the [full results and next actions](pilot_runs/reward_scale_controls_20260921/RESULTS.md).
This update supersedes the September 18 pending-work descriptions below.

**Later September 21 update:** the default-preserving value-clip option and
strict comparator are implemented. The completed [clip-only pilot](pilot_runs/value_clip_controls_20260921_local/RESULTS.md)
found dense success 88%/14% at clip 10 versus 26%/8% at clip 1,000,000
(seeds 110/111), with no-message success 0% throughout. The larger cap also failed
as a reliability fix. A failed LAN-addressed attempt is preserved separately;
all eight local-only retry runs passed consistency checks. Forty unit tests and
four routing checks passed. Defaults remain reward scale 1.0 and clip 10.
Next: diagnose late-policy degradation using saved checkpoints and paired
development trajectories before choosing another training intervention.

1. [Current status](#1-current-status)
2. [How the code works](#2-how-the-code-works)
3. [Implemented changes](#3-implemented-changes)
4. [Experiment history](#4-experiment-history)
5. [Immediate engineering priority](#5-immediate-engineering-priority)
6. [Remaining code backlog](#6-remaining-code-backlog)
7. [Research decision gates](#7-research-decision-gates)
8. [Experimental and statistical discipline](#8-experimental-and-statistical-discipline)
9. [Timeline to TERRA](#9-timeline-to-terra)
10. [Eligibility, authorship, and preparation](#10-eligibility-authorship-and-preparation)
11. [Reproduction and artifact index](#11-reproduction-and-artifact-index)
12. [Next-session checklist](#12-next-session-checklist)

## 1. Current Status

### The question we are trying to make testable

At a fixed communication budget, does selecting neighbors by learned attention improve cooperative payload transport compared with selecting neighbors randomly? Dense in-range communication is the high-budget reference; a trained no-message actor tests whether explicit messages help at all.

This is a useful research question, but attaching top-K to a GAT is not, by itself, a demonstrated novel contribution. A defensible project needs a correct task, fair controls, independent training replications, a mechanism analysis, and accurately bounded claims. There is no evidence-based way to promise a TERRA win or ISEF selection.

### Status at a glance

| Area | Current evidence | Status |
| --- | --- | --- |
| Environment and routing correctness | Seeded replay, graph-budget checks, explicit backends, boundary tests | Substantially improved; tests are not a full proof |
| Basic learned transport | Robot-relative all-goal dense policies achieved 100%, 82%, and 100% in three development seeds | Learnability substantially improved |
| Need for communication in all-goal task | Matched no-message actors achieved 92% and 100% | All-goal task does not establish a communication benefit |
| Private-goal communication control | Dense 22% and 74%; trained no-message 0% and 0% | Message benefit observed in two development pairs |
| Private-goal dense reliability | Neither seed reached the prespecified 80% gate | Not ready for the main sparsification sweep |
| Critic learning | Actual minibatch saturation reduced by scale 0.01, but dense success fell in both new seeds | Scaling pilot rejected as a reliability fix; default remains 1.0 |
| Attention versus random on corrected private-goal task | No completed matched study | Open question |
| Physical realism | Reported experiments use the kinematic backend | PyBullet transport and real hardware unvalidated |
| Communication savings | Selected-edge delivery accounting exists | No demonstrated real-network bandwidth or runtime savings |
| Statistical evidence | Two paired training seeds in the latest study | Development evidence, not a final significance result |

**What we are improving now:** the shared environment, representation, training reliability, and evaluation foundation. Dense is the main diagnostic condition and no-message is its control. These improvements should be carried into attention, random, and other conditions consistently. We are not tuning attention until it wins.

**Most important correction to the early story:** a fully connected candidate graph can still offer a top-K choice whenever degree exceeds K. Full connectivity removes spatial selectivity; it does not automatically remove neighbor selection. A separate issue is whether observations and task dynamics make messages useful in the first place.

## 2. How the Code Works

### Execution path and ownership

```text
run_sweep.py -> train.py -> RLlib PPO + shared policy
                              |
                       env_core.py observations
                              |
                       marl_agent.py
                        /          \
            local actor + GNN    separate global critic encoder
                    |
            gnn_comm_layer.py
                    |
            two continuous wheel commands per robot

portable policy export -> evaluate.py -> episode records
                                      -> compare_evaluations.py
                                      -> diagnostics and plots
```

| File | Responsibility and relevant changes |
| --- | --- |
| [env_core.py](env_core.py) | Dynamics, seeded resets, graph construction, goal visibility, rewards, termination, observation-stored routing randomness |
| [gnn_comm_layer.py](gnn_comm_layer.py) | Edge-conditioned multi-head attention, candidate masking, budgeted routing, shared routing across layers, routing diagnostics |
| [marl_agent.py](marl_agent.py) | Shared actor, separate critic encoder, robot-relative transforms, no-message switch, message sensitivity |
| [train.py](train.py) | PPO configuration, seeds/resources, manifests, strict resume, portable exports, validation, structured metrics |
| [evaluate.py](evaluate.py) | Strict model loading, deterministic mean actions, scenario replay, robustness interventions, delivery accounting |
| [run_sweep.py](run_sweep.py) | Matched condition/seed schedules, consistent argument forwarding, source guards, optional final development evaluations |
| [compare_evaluations.py](compare_evaluations.py) | Compatibility checks, per-training-seed aggregation, paired differences and bootstrap summaries |
| [validate_task.py](validate_task.py) | Privileged scripted-controller versus random-action feasibility checks |
| [diagnose_transport.py](diagnose_transport.py) | Full trajectories, contact/alignment summaries, original-horizon versus extended-horizon checks |
| [plot_transport_diagnostics.py](plot_transport_diagnostics.py) | Figures from saved trajectory diagnostics |
| [analyze_attention_entropy.py](analyze_attention_entropy.py) | Routing concentration, margins, churn, degree, and overlap diagnostics |
| [test_research.py](test_research.py) | Main regression suite, currently 31 tests |
| [test_random_mode_determinism.py](test_random_mode_determinism.py) | Separate random-routing replay acceptance checks |
| [summarize_sweep.py](summarize_sweep.py), [experiments.md](experiments.md) | Historical records/utilities, not the authoritative current research result |

### Environment and observations

Current pilot settings are eight robots, communication radius 1.8, spawn radius 1.0, spawn jitter 0.1, initial goal distance 4.0, success distance strictly below 0.5, and a 250-step episode limit. These are **pilot settings**, not all CLI defaults. For example, the trainer defaults to four robots, 500 steps, world coordinates, and PyBullet unless overridden.

Each raw node observation has 27 entries:

| Python slice | Content |
| --- | --- |
| `0:3` | Position |
| `3:6` | Velocity |
| `6:18` | Twelve lidar readings |
| `18:21` | Relative goal vector, zero for uninformed robots |
| `21:24` | Relative payload vector |
| `24:26` | Sine and cosine of heading |
| `26` | Whether this robot directly observes the goal |

Each agent's observation dictionary also contains all node features, adjacency, eight-dimensional edge features, its node index, and a stored random-score matrix. This is a centralized simulator representation. The actor is constrained through the implemented local/GNN paths, not through a real network API. The global critic has a separate encoder; it can see privileged training information without directly feeding that embedding into the actor. Parameter sharing and centralized training do not imply that actors have unrestricted global information at execution.

The current trainer uses RLlib's old API stack, learning rate `3e-4`, discount `0.99`, GAE lambda `0.95`, PPO policy clip `0.2`, value-loss coefficient `0.5`, and entropy coefficient `0.01`. The communication latent is 64-dimensional, local hidden width is 128, and attention uses four heads. Pilots override depth to one layer, batch size to 512, and epochs to two. With that batch size, the configured minibatch size is 128. These settings and the installed loss implementation should be preserved in an exact-version reproduction.

`--goal-observers -1` exposes the goal to all robots. A positive count chooses that many observers per episode and masks everyone else's goal vector. `--goal-spawn-mode independent` independently samples goal direction and spawn rotation. Historical `coupled` mode remains available for old seeded scenarios, but is inappropriate for the current private-goal comparison because spawn geometry supplies a goal-related cue.

`--observation-frame robot` expresses the actor's graph relative to its own XY pose and heading. Positions, velocities, goal/payload vectors, neighbor headings, and edge geometry transform consistently; body-relative lidar stays body-relative. The reference frame uses the robot's own pose, not the hidden goal. This is an opt-in model representation and requires fresh training.

### Dynamics and reward

The default simulation interval is `10 * (1/240)` seconds per environment step. The kinematic backend integrates wheel-like commands and moves the payload using proximity-weighted robot velocities when at least two robots are within the contact radius. These are proximity counts, not measured contact forces. The PyBullet implementation uses force-driven sphere bodies and needs its own feasibility validation; it is not a validated differential-drive hardware model.

For `transport_v2`, the per-agent reward is approximately the following exact component structure, with defaults shown:

```text
r_i = 10 * (previous_payload_goal_distance - current_payload_goal_distance)
    +  1 * (previous_mean_robot_payload_distance - current_mean_robot_payload_distance)
    + 0.1 * proximity_weight_i * dot(robot_velocity_i, current_goal_unit_vector)
    - 0.005 * squared_action_norm_i
    - 0.01
    + 100 * success_indicator
```

The individual contribution term applies only when robot i is within the contact radius and the minimum proximity count is met. The completion bonus is terminal. Progress and approach are shared shaping components. Recurring connectivity and contact bonuses are removed in `transport_v2`; `legacy` remains for controlled historical investigations but does not recreate every detail of old runs. Distance-difference shaping is not guaranteed policy-invariant potential shaping with discount below one.

### Communication semantics

| Condition | Implemented meaning | Important limitation |
| --- | --- | --- |
| Dense | Aggregate all eligible in-range neighbors, plus self | Not necessarily globally fully connected |
| Attention top-K | Rank valid non-self neighbors by learned attention scores | Discrete selected indices have no derivative, although retained attention values and shared score parameters learn |
| Random top-K | Rank valid neighbors by stored random scores | Temporal score scope is a distinct experimental choice |
| Distance top-K | Select nearest valid neighbors | Useful geometric control, not learned selection |
| Gumbel top-K | Hard forward budget with relaxed straight-through gradient | Biased estimator; no established performance advantage |
| `top_k=0` | Self-only GNN path | Different capacity/path from bypassing the actor GNN |
| `no_comm` | Zero the actor communication latent; retain centralized critic | Physical coupling and implicit information through motion remain |

K counts non-self neighbors. A robot with fewer than K valid neighbors receives fewer than K messages. Self is preserved without using a communication slot. Every GNN layer reuses the first routing decision; extra layers permit propagation along that selected graph, not an independent new graph each layer.

Random scores default to changing at each environment step, but are stored in the observation so repeated PPO optimization of that sample sees the same stochastic input. `--random-mask-scope episode` keeps scores fixed for an episode; changing proximity can still change selected neighbors. Per-step resampling is a legitimate condition when implemented consistently. Unrecorded redraws during forward/replay were the bug, not stochasticity itself.

**Cost boundary:** scoring and message construction still operate on dense tensors before masking. Reported logical bytes assume float32 hidden vectors on retained directed edges. They exclude discovery, selection/score exchange, packet overhead, retries, and broadcast effects. Since ranking can inspect candidate features before dropping them, selected-message count is not the complete cost or information-access budget of a deployed selector.

## 3. Implemented Changes

### A. Correctness and research infrastructure

| Change | Problem addressed | What now exists / what it does not prove |
| --- | --- | --- |
| Explicit backend choice and missing-PyBullet error | Silent fallback could make nominally identical runs use different dynamics | Backend is intentional and recorded; Bullet realism is still unvalidated |
| Kinematic time integration correction | Earlier fallback displacement used inconsistent elapsed time | Timing tests and explicit simulation interval; old timing-affected results remain historical |
| Configurable spawn geometry, radius, goal distance, heading/jitter | Early scenarios were insufficiently diagnostic of spatial communication | Partial graphs can be checked directly; useful selection still requires degree greater than K |
| Environment-local seeded RNG | Reset and routing reproducibility could depend on unrelated global draws | Reproducible fixed-seed scenarios and stochastic inputs |
| Valid-neighbor masking before top-K | Invalid/out-of-range candidates could displace valid neighbors | Budget, invalid-edge, isolated-node, and self-path checks |
| Replay-stable random routing | Repeated forward passes could use different random selections for the same PPO sample | Observation-stored scores; explicit step/episode semantics |
| Routing reuse across GNN layers | Per-layer selection could obscure the effective communication protocol | Shared first-layer routing across the stack |
| Gumbel hard-forward/relaxed-backward path | Earlier logic did not provide the intended routing-learning surrogate | Gradient and hard-budget checks; estimator remains biased |
| True no-message actor control and nearest-distance control | Dense versus sparse alone cannot establish communication necessity or learned-selection value | Additional baselines; K=0 and no-message remain distinct |
| Separate global critic encoder | Critic information/path could confound execution-time information restrictions | Same central-critic design across conditions; hidden-goal boundary tested |
| 27-feature observation with heading and observer flag | Control lacked a consistent heading representation and explicit goal-visibility status | Old 26-feature weights are incompatible, not silently adapted |
| `transport_v2` reward | Repeated contact/connectivity income could inflate return without transport | Progress, costs, terminal success, later approach/contribution shaping |
| Strict portable exports | Partial loading or fallback model construction could yield misleading evaluations | Full weights/configuration required; no silent random-policy substitute |
| Run manifests and structured records | Logs alone did not identify task, budget, source, or environment | Source hashes, library versions, arguments, model/env config, JSONL metrics |
| Fresh-output guards and disabled old launchers | Reused paths could overwrite or truncate evidence | New run roots required; historical logs retained |
| Standalone seeded evaluation | Training return conflated optimization and actual completion | Per-episode success/distance/effort/time/delivery measurements |
| Matched-seed comparison tool | Episode counts could be mistaken for independent learning replications | Strict task/budget/provenance checks and seed-level paired summaries |
| Robustness and routing diagnostics | A single reward average could not explain communication behavior | Candidate loss, stale features, noise, team-size/radius tests, entropy and action sensitivity |

### B. Transport diagnosis and learnability

1. Added approach shaping and individual goalward contribution shaping to help credit assignment. This changed the development objective; later studies must keep it matched across communication conditions.
2. Added strict Ray checkpoint continuation. The parent manifest and relevant source hashes are checked before restore, the destination must be fresh, and restored configuration is verified. `--max-iterations` means additional iterations when resuming. Continuation is not a new independent seed, and exact environment/RNG continuity is not guaranteed.
3. Fixed final checkpoint/export handling so a nonperiodic last iteration, such as 81, still produces both a portable policy and Ray checkpoint.
4. Added trajectory recording of robot/payload states, headings, actions, contact counts, progress, and velocity alignment. Recording is tested not to change ordinary evaluation.
5. Added extended-horizon diagnostics and verified identity of the original-horizon prefix. This separated policies that simply needed more time from policies that drifted or stalled.
6. Implemented actor-relative observations after world-frame failures showed persistent directional/control problems. Regression coverage checks rotation/translation invariance, finite gradients, goal masking, and export round trips across communication modes.
7. Corrected message-action sensitivity to read each actor's own receiver row. Robot-relative graphs differ between actors, so a single shared graph row is not a valid substitute.

### C. Communication-relevant task and matched controls

1. Added independent goal/spawn orientations while preserving the coupled historical option. The sweep runner rejects private-goal schedules that retain the coupled cue.
2. Expanded the sweep CLI to forward observation frame, layer count, horizon, PPO batch/epochs, validation schedule, and development evaluation settings consistently. Condition subsets are supported.
3. Added source checks before and after jobs so edits during a sweep cannot silently enter a pooled comparison. The saved schedule records the actual commands and source hashes.
4. Fixed entropy/routing analysis to use the correct per-actor graph coordinates and save configuration/provenance.
5. Strengthened comparisons to reject mixed architectures and mixed evaluation scenario lists across training seeds, not just mismatches within one pair. JSON comparison output is supported.
6. Completed matched all-goal dense/no-message replications, then fresh private-goal dense/no-message controls with a prespecified message-removal evaluation.
7. Added a hidden-goal counterfactual regression test: change only the goal while holding the physical snapshot fixed. Uninformed no-message actors and actors with all non-self edges removed do not change their actions; one-layer dense actors outside the observer's direct neighborhood also remain unchanged.
8. Added explicit message-value-path analysis and terminal critic-target replay audits, with artifact-consistency checks and reproducible figures.

### D. What has not been changed or established

- Uniform reward scaling now exists, defaulting to 1.0; the existing value clip is explicitly pinned at 10.0. There is no demonstrated training-reliability improvement or alternate value-clip treatment yet.
- No corrected-task attention/random sweep has been completed; the old random-wins story is not a valid final result for this version.
- No communication-cost reduction in a real network or sparse runtime implementation has been demonstrated.
- No physical robot or validated PyBullet transport experiment supports these results.
- No untouched final test has been reported. Seeds 50000 and 80000 are development resources.
- No causal conclusion about individual neighbor usefulness follows from attention entropy or one-step action sensitivity.
- Some inherited file headers/docstrings describe stronger convergence/physics claims than the implemented evidence supports. Those comments are not experimental validation and should be cleaned up in a scoped documentation pass.

## 4. Experiment History

All success counts below are **development evaluations**. A scenario seed is not a training replicate. All reported learned-transport results here use the kinematic backend. Different tasks, horizons, representations, source versions, and training budgets must not be pooled as one experiment.

### Historical and early development runs

The inherited/early sweeps had multiple validity problems: routing masks, random replay, environment timing, reward/task design, and checkpoint evaluation changed over the repair sequence. [experiments.md](experiments.md) retains earlier records and pending-work notes; those notes are not the current project status. Preserve these artifacts as history rather than relabeling them as corrected experiments.

Two early learnability pilots, `dense_all_goal_s101` and `dense_learnability_s102`, were stopped after 2 and 28 iterations, respectively, without validation successes. They are failed/unfinished development attempts, not missing observations to quietly exclude from a favorable final sweep.

### World coordinates versus robot-relative coordinates

All rows below use training seed 103, eight robots, all-goal observations, a one-layer dense GNN, a 250-step evaluation horizon, and the same 50 development scenarios, 80000-80049.

| Run | Representation | Iterations / environment steps | Success | Mean final distance |
| --- | --- | ---: | ---: | ---: |
| `dense_credit_s103` | World | 40 / 20,480 | 3/50 = 6% | 2.495 |
| `dense_credit_s103_continued` | World, continued from above | 81 total / 41,472 | 8/50 = 16% | 1.610 |
| `dense_robot_frame_s103` | Robot-relative, fresh training | 40 / 20,480 | 50/50 = 100% | 0.493 |

The robot-relative run completed in a mean 118.78 steps, maximum 136. Its repeated validation counts at iterations 10/20/30/40 were 14/20, 19/20, 18/20, and 20/20. The matched-budget comparison is the first versus third row; the continued world-frame row used more training. This supports a major representation-related improvement in this pilot, not universal convergence or a communication advantage.

For the continued world-frame policy, extending evaluation from 250 to 750 steps changed successes from 8/50 to 14/50. However, 23 of the 42 original failures finished over 0.1 farther away with extra time, 18 had negative final-window goalward velocity, and five spent most of that window below the proximity-contact threshold. These overlapping diagnostics motivated investigating control, not just increasing the horizon.

Records: [continuation](pilot_runs/dense_credit_s103_continued/DEVELOPMENT.md), [robot-relative pilot](pilot_runs/dense_robot_frame_s103/DEVELOPMENT.md).

### All-goal replication and no-message control

Fresh seeds 104 and 105; 40 iterations / 20,480 environment steps each; robot-relative observations; historical coupled all-goal task; same 250-step horizon and 50 development scenarios.

| Training seed | Dense success | No-message success | Dense mean distance | No-message mean distance |
| --- | ---: | ---: | ---: | ---: |
| 104 | 41/50 = 82% | 46/50 = 92% | 0.555 | 0.667 |
| 105 | 50/50 = 100% | 50/50 = 100% | 0.491 | 0.488 |

The mean paired success difference, dense minus no-message, was -5 percentage points. The two-seed bootstrap range of -10 to 0 points is descriptive, not robust evidence of inferiority or equivalence. Seed 104 illustrates a metric tradeoff: dense had fewer successes but smaller average final distance. Mean episode steps include timeouts and are not success-conditioned completion times.

An exploratory dense-104 rollout extended to 750 steps reached 49/50 while reproducing the original 250-step prefix exactly. That is an evaluation-condition change, not a new trained-policy result. Separately, an independent-goal feasibility check at 250 steps gave the privileged scripted controller 20/20 successes, random actions 0/20, and initially partial graphs in 20/20 scenarios. A privileged controller is not an RL policy or a proof that messages are necessary.

Record: [all-goal replication report](pilot_runs/robot_frame_replication_20260917/RESULTS.md).

### Private-goal controls: latest completed experiment

Protocol: exactly one randomly designated goal observer among eight robots; independent goal/spawn rotations; robot-relative observations; one GNN layer; radius 1.8; 250-step horizon; **80 iterations / 40,960 environment steps** for every run. PPO batch 512, two epochs, four workers. Fresh matched training seeds 104/105. Final scheduled iteration-80 checkpoints, not retrospectively chosen peaks, were evaluated on scenarios 80000-80049.

| Seed | Dense | Trained no-message | Same dense checkpoint, non-self messages removed |
| --- | ---: | ---: | ---: |
| 104 | 11/50 = 22% | 0/50 = 0% | 0/50 = 0% |
| 105 | 37/50 = 74% | 0/50 = 0% | 5/50 = 10% |

| Seed | Dense mean distance | No-message mean distance | Dense with messages removed |
| --- | ---: | ---: | ---: |
| 104 | 1.656 | 3.775 | 3.874 |
| 105 | 0.822 | 3.781 | 4.857 |

The mean paired dense advantage was 48 percentage points. The two-seed bootstrap interval of 22-74 points reflects these two observed differences; it is not a persuasive final population-level significance claim. Removing messages reduced the same dense policies' success by 22 and 64 points. Removal changes the policy's input distribution and aggregation, so this is evidence of message dependence in these pilots, not a clean estimate of every neighbor's causal value.

The prespecified practical gates required each dense seed to reach at least 80% success and exceed its no-message control by at least 20 points. **Both communication-gap gates passed; both reliability gates failed.** These are engineering thresholds, not hypothesis tests.

Training remained unstable: dense seed 105 achieved 20/20 repeated validation successes at iterations 50 and 60, dropped to 9/20 at 70, and finished at 17/20. Dense seed 104 peaked at 11/20 and finished at 10/20. Reporting the best intermediate result as the final result would misrepresent the fixed-checkpoint protocol.

Information-path diagnostics found mean explicit value-path reachability to uninformed robots of 0.768 and 0.914. Seed-104 failures still averaged 0.730 reachability and did not have insufficient post-action proximity contacts. Missing every contact or every goal-information path cannot by itself explain those failures. Available information is not necessarily used well; physical motion can also carry information.

Records: [prespecified protocol](pilot_runs/private_goal_controls_20260917/PROTOCOL.md), [results](pilot_runs/private_goal_controls_20260917/RESULTS.md), [machine-readable summary](pilot_runs/private_goal_controls_20260917/summary.json).

## 5. Immediate Engineering Priority

### Identified issue: reward and critic-loss scales

The installed old-stack RLlib PPO implementation computes:

```text
value_loss = clamp((V - target)^2, 0, vf_clip_param)
vf_clip_param = 10.0  # installed default, now explicitly pinned at the same value
```

Above an absolute prediction error of `sqrt(10)`, about 3.162, this clipped loss has zero derivative with respect to V. This is clipping the squared loss itself in this implementation, not simply limiting a gradient or clipping a prediction around an old prediction. Verify this again if changing RLlib versions.

At a true successful terminal transition there is no future bootstrap, so its target is the immediate reward. The historical runs used an unscaled 100-point completion bonus. Replay of their final private-goal policies found:

| Seed | Terminal agent samples | Mean critic prediction | Mean terminal target | Zero gradient from clipped value term |
| --- | ---: | ---: | ---: | ---: |
| 104 | 88 | 11.14 | 100.21 | 100% |
| 105 | 296 | 18.71 | 100.28 | 100% |

The 384 samples come from eight agents in each of 48 successful episodes; they are not 384 independent training replications. This replay audit measures actual evaluation rewards and cached pre-action critic predictions, then differentiates the installed loss expression. It does **not** reconstruct the original PPO training minibatches or prove the cause of instability. Other losses can still update shared parameters.

Evidence: [audit script](pilot_runs/private_goal_controls_20260917/audit_value_scale.py), [seed-104 audit](pilot_runs/private_goal_controls_20260917/dense_s104/value_scale_audit.json), [seed-105 audit](pilot_runs/private_goal_controls_20260917/dense_s105/value_scale_audit.json). Installed source: `venv314/lib/python3.14/site-packages/ray/rllib/algorithms/ppo/ppo_torch_policy.py`, with the default in `ppo.py` in the same directory.

### Implemented candidate and remaining validation

The following numbered record describes September 18. As of September 21,
items 4, 6 and 7 have been followed by actual minibatch instrumentation, stock
loss/gradient/RNG equivalence tests, smoke/resume checks and the fixed pilot.
The suite now has 34 unit tests plus four routing acceptance checks. The
[pilot report](pilot_runs/reward_scale_controls_20260921/RESULTS.md) records the
negative scaling result. Next candidate: a separately controlled value-clip
intervention at raw reward scale, with no claim that it will solve reliability.

1. Implemented uniform `--reward-scale`, default 1.0. A candidate treatment is 0.01, making the completion bonus 1.0. The value clip remains 10.0; no simultaneous loss or architecture change was introduced.
2. Preserved physical dynamics, observations, reset draws, horizon, termination, and relative reward-component weights. Raw/scaled component dictionaries are available in per-agent info; evaluation records raw and learning returns separately.
3. Added scale to environment/trainer CLI, sweep generation, manifests, exports, and resume checks. Existing strict configuration comparisons reject mismatched scales. Portable exports lacking the field retain scale 1.0; historical manifests are not rewritten.
4. Added structured logging of RLlib's available losses, explained variance, entropy, KL, and optimizer metadata/source provenance. Unclipped value loss, value saturation fraction, and policy clip fraction on actual optimizer minibatches remain missing. Clipped loss alone cannot diagnose saturation.
5. Added tests for identical observations/physics, scaled rewards including completion and truncation, finite positive scales, export compatibility, and pre-Ray resume rejection. The main suite now has 31 passing tests.
6. Checked terminal-target gradient arithmetic at V=0 and completed a one-iteration PPO smoke test. These are software checks, not proof of useful trained critic gradients; actual minibatch inspection and full-length fresh training remain necessary.
7. Freeze a small development comparison before launching it. Train fresh dense and no-message controls with matched settings and predetermined endpoints. Compare reliability, value diagnostics, and task success rather than raw return magnitudes across reward scales.
8. Retain all outcomes, including a failed candidate. Only expand the comparison if the intervention improves the intended learning behavior without changing task semantics or introducing a new confound.

Uniform positive scaling preserves reward-component ratios and the ideal ordering of discounted policy returns under fixed assumptions. It does **not** leave PPO optimization unchanged: critic targets, clipping, loss balance, and regularization interactions change. There is no guarantee that 0.01, a larger clip, or either candidate will solve transport.

**Acceptance gate:** regression tests pass; provenance captures the setting; relevant critic targets are not systematically trapped in a flat loss region; fresh matched pilots show stable useful transport; the communication control remains meaningful. Do not call the problem fixed merely because training runs without exceptions.

## 6. Remaining Code Backlog

| Priority | Work | Completion evidence |
| --- | --- | --- |
| P0 | Validate the implemented scale candidate | Software tests passed; actual minibatch saturation diagnostics and matched fresh pilots remain |
| P0 | Audit software environment reproducibility | Tested version specification for the working Python/Ray/Torch stack; record installed versions and loss-source behavior; do not assume old `requirements.txt` recreates this environment |
| P0 | Preserve source snapshots and results | Immutable run artifacts, verified backups, readable version-control history when available; no destructive repair of existing metadata |
| P1 | Improve training telemetry | Value/advantage statistics, entropy/KL/clip fraction, reward components, actual steps and wall time in structured logs |
| P1 | Complete experimental CLI forwarding | Reward scale is forwarded; random-mask scope is still available only in the trainer, not the sweep runner |
| P1 | Audit decision opportunities and effective budgets | Degree distributions, fraction of decisions with degree greater than K, retained counts and messages per step/round across conditions |
| P1 | Audit selector information access | Test influence through unselected candidates and document features needed before selection; distinguish value paths from selection-dependent information |
| P1 | Mechanism diagnostics | Reproducible selected-edge interventions with fixed initial states; distinguish action sensitivity from closed-loop return effects |
| P1 | Final-analysis validation | Strict paired data checks, failed-run policy, clearly defined intervals, multiplicity plan, and test-set access record |
| P2 | Depth/reachability experiment if needed | Controlled one-layer versus deeper comparison, same depth across main routing conditions, costs reported per round |
| P2 | Observer-aware geometric heuristic if justified | Uses only allowed information and the same budget; any observer-status discovery cost disclosed |
| P2 | Robustness experiment harness | Prespecified held-out team sizes/radii/link availability/delay settings, intervention semantics saved in each output |
| P2 | True sparse execution or deployable protocol | Costs include candidate discovery/scoring; measured end-to-end runtime or network traffic, not just masked-edge counts |
| Optional | PyBullet validation | Scripted feasibility, meaningful contacts/forces, timing/reset checks, fresh training and matched controls in that backend |
| Optional | Physical robot demonstration | Separate safety review and validated transfer; not a late replacement for a sound simulation study |

Do not patch installed RLlib source as an undocumented shortcut. Prefer an explicit configuration or a narrowly scoped, versioned custom loss when justified and tested. A dependency upgrade is a protocol change, not housekeeping during a frozen sweep.

Some proposed comparisons intentionally change configuration, source, or architecture and should be rejected by the strict same-protocol comparator. Build an explicit, audited analysis path for those interventions; do not weaken the main comparator until everything passes.

## 7. Research Decision Gates

These are candidate engineering/research checkpoints for discussion, not an authored competition research plan. The student should choose and justify the scientific protocol independently, record decisions before the relevant runs, and discuss permitted AI assistance with the sponsor/SRC.

### Gate A: task integrity and learnability

- Verify that goal masking and independent spawn/goal sampling remain correct after every relevant change.
- Verify that agents have genuine selection opportunities: eligible non-self degree exceeds K often enough, and graph topology is neither always complete nor usually empty if spatial selectivity is part of the claim.
- Establish privileged-controller feasibility and a substantially weaker random-action control without confusing either with learned performance.
- Establish stable dense learned transport across fresh seeds and record the full learning curves, not just a favorable checkpoint.
- Avoid simultaneously changing reward, representation, episode length, task geometry, and PPO settings. A curriculum or longer training budget may be reasonable, but it needs its own documented comparison.

### Gate B: communication usefulness

Repeat matched dense/no-message controls after the training fix. Preserve the current practical reliability/gap thresholds for the immediate replication unless a new threshold is justified and recorded before running. Add fresh training seeds rather than treating repeated evaluations of seeds 104/105 as additional replications.

The current private-goal task is promising because its message and no-message controls separate, but physical payload/robot motion can transmit goal information indirectly. The claim should concern the incremental benefit of explicit messages under the implemented conditions, not absolute impossibility without communication.

If dense remains unstable, prioritize failure trajectories, critic learning, and information reachability. With one message-passing layer and a feed-forward policy, a temporarily remote observer may not directly inform every robot. Recurrence or deeper propagation are possible future design choices, not automatic fixes. They alter capacity, latency, and costs and must be matched across routing methods.

### Gate C: answer the primary attention-versus-random question

Once the controls are credible, freeze a manageable main comparison. A reasonable candidate is attention versus random at K=2, with dense and no-message references. K=1 and K=3, distance selection, or Gumbel can be secondary studies if resources allow. There is no requirement to run every available option before obtaining a useful result.

Keep the observation frame, goal visibility, task geometry, horizon, rewards, critic, network widths/depth, PPO settings, training budget, validation rule, and scenario list matched. Apply a common tuning budget rather than giving the preferred condition substantially more optimization effort.

Random and attention top-K share the downstream learned message/attention machinery; the primary difference is how the retained subset is chosen. This helps isolate selection from the mere presence of learned aggregation. A K=0 self-only path can separately assess the architectural difference introduced by the `no_comm` bypass.

Choose random temporal scope in advance. Stepwise random scores and episode-fixed scores answer different questions. If both are studied, label them separately. Also measure selection churn for attention: different temporal correlation may contribute to a difference even when K matches.

### Gate D: distinguish a performance difference from a mechanism

Routing performance and causal faithfulness are separate questions. The reviewed feedback proposes making faithfulness the eventual scientific endpoint, but this is a scope decision for the student, not a conclusion forced by the current results. A useful extension needs precise intervention definitions and paired counterfactual validation; see the [review's requirements](FEEDBACK_REVIEW_20260918.md#requirements-for-a-later-faithfulness-study). Do not build a large counterfactual system before the learning controls are credible.

| Diagnostic question | Existing support | Additional work / caveat |
| --- | --- | --- |
| Does explicit communication help? | Dense/no-message and same-policy removal | Replicate beyond two training seeds; removal is distribution shift |
| Are scores concentrated? | Entropy and score-margin analysis | Concentration is not usefulness |
| Does changing a message change the action? | Fixed-routing message sensitivity | Action changes need not improve return |
| Do selected value paths reach informed agents? | Private-goal path diagnostics | Reachability is not utilization; selection scores may create other dependencies |
| Are selected neighbors better than alternatives? | Not established | Prespecified replacements/removals, matched initial states, repeated trained policies, closed-loop outcomes |
| Is the ranking itself useful? | Not established | Compare learned ranking with shuffled/random ranking while keeping budget and policy context explicit |
| Are savings real? | Logical selected-delivery accounting | Include score/discovery traffic or narrow the claim to logical message sparsity |

Evaluate ranking manipulations as interventions on a fixed trained policy separately from training separate conditions. The former measures reliance/sensitivity under changed inputs; the latter measures learnability under each communication regime. They are not interchangeable estimands.

### Gate E: robustness and scientific scope

After the main comparison is frozen, a small prespecified robustness set can ask whether the effect survives a changed team size, changed radius, unavailable links, or stale information. The current `candidate_loss` removes directed links before selection, allowing rerouting; `delay` stales node features while keeping current geometry/local observations; `noise` perturbs raw observation features. Do not label these complete packet-loss, network-delay, or sensor models.

A scientifically useful outcome can be a reliable attention advantage, a reliable random advantage, a context-dependent difference, or an inconclusive comparison with honest uncertainty. Lack of statistical significance is not evidence of equivalence. Do not search tasks until attention wins and report only that task.

The potentially distinguishing contribution is careful explanation of **when selection helps and why**, with information restrictions and costs made explicit. Whether that contribution is genuinely novel requires the student's own literature review and verification of original sources. Keep a reading matrix covering each work's task, information assumptions, routing budget, random control, learning method, costs, and limitations; this document is not a supplied competition bibliography.

## 8. Experimental and Statistical Discipline

### Experimental units and provenance

1. The main learning replicate is an independently trained policy seed. Eight robots in one team are not eight replicates; fifty scenarios from one trained policy are not fifty learning replicates.
2. Use paired training seed IDs across conditions and the same evaluation scenarios. Shared seed IDs do not guarantee identical stochastic trajectories or parameter initialization after architectural changes; inspect and describe the actual pairing.
3. Aggregate scenario outcomes within each trained policy, then compare conditions across training seeds. Preserve individual episode rows for audit and possible hierarchical analyses.
4. Report actual environment steps, agent steps where relevant, optimizer epochs, checkpoint iteration, workers, wall time, and architecture. Equal iteration counts are not equal budgets if batch size changes.
5. Record all prespecified seeds and run failures. An infrastructure crash should be classified and handled by a predeclared rerun rule, not silently dropped; nonlearning is an outcome, not an infrastructure failure.
6. Preserve protocol, schedule, manifests, exact source/dependency versions, exports, learning curves, evaluation records, and analysis scripts. Source hashes identify files; without source snapshots they do not reconstruct those files.

### Development versus final testing

| Resource | Current use | Rule |
| --- | --- | --- |
| Validation seeds starting 50000 | Repeated checkpoint inspection | Development only |
| Development seeds 80000-80049 | Diagnosis and every main pilot comparison so far | Already exposed; not untouched evidence |
| Reserved seeds starting 100000 | Intended final evaluation pool; unused in the reported pilots | Do not inspect until task, method, budgets, metrics, and checkpoint rules are fixed |
| Training seeds 103/104/105 | Extensively used for development | Add fresh learning replications; do not count resumes as new seeds |

Reserve the full final scenario list and access record, not merely a starting number. If final-test feedback changes the method, the set has become development data; a new untouched evaluation is required for a fresh confirmatory claim. Fix checkpoint selection before testing. Current studies use the final scheduled checkpoint, not best-of-many test performance.

### Metrics and uncertainty

Primary candidate endpoint: episode success at the fixed horizon. Report percentage-point differences, per-training-seed results, and uncertainty. Secondary endpoints can include final distance, progress, action effort, logical deliveries, and completion-time information with failures handled explicitly. Return is an optimization diagnostic, not a substitute for task success.

Match communication accounting to the question. Equal K gives a cap, not equal realized episode traffic: degree, trajectory, and episode duration differ. Report retained edges per decision/round and total episode deliveries, preferably including a common-duration view where appropriate. Early successful termination naturally changes total traffic.

The current comparator resamples paired training-seed differences with a fixed bootstrap RNG and reports a descriptive 95% interval and paired standardized effect. It does not implement multiplicity correction, a power analysis, a formal equivalence test, or a universal guarantee from a particular seed count. Two-seed intervals are especially fragile. Plan final replication from observed variability, a meaningful effect size, compute limits, and the desired precision; five seeds can be an initial budget, not a promise of adequate power.

For one primary comparison, state its metric, K, checkpoint rule, and analysis before final testing. Label other budgets, robustness settings, and mechanisms secondary/exploratory, or apply an appropriate prespecified multiplicity strategy with statistical guidance. Avoid declaring a discovery because one of many variants happens to cross a threshold. Report absolute rates and intervals even when an effect is not significant.

### Honest claim boundaries

| Supported description | Unsupported leap |
| --- | --- |
| Robot-relative representation substantially improved these all-goal development runs | The controller is universally reliable |
| Private-goal dense outperformed no-message in two development pairs | Communication is mathematically necessary or performance is conclusively generalized |
| Replayed terminal targets fall in a flat value-loss region | This is proven to be the sole cause of PPO instability |
| Top-K retains fewer logical message deliveries where degree exceeds K | End-to-end compute or real bandwidth is reduced by the same percentage |
| A tested routing method beats a matched control under a frozen protocol | Attention generally solves multi-robot communication |
| Kinematic cooperative transport simulation | Validated physical multi-robot transport |

## 9. Timeline to TERRA

### Published schedule checked September 18, 2026

The official home page currently lists applications **October 1-January 11**, online preliminary review **January 12-February 27**, and the final at **NYU Tandon on March 20, 2027**. It also lists ISEF **May 9-14, 2027, Los Angeles**. The application/preliminary ranges omit years; their placement beside the 2027 final suggests the October 2026-January/February 2027 cycle. Confirm the dated deadlines in the active portal with the organizer rather than relying on that inference. [TERRA NYC schedule](https://tnycfair.org/)

The application page calls for required forms, a project plan, research paper, quad chart, and at least five cited sources. It warns that invalid or fabricated citations can disqualify an application. Coordinate teacher registration and signatures well before winter break. [TERRA NYC application requirements](https://tnycfair.org/apply.html)

### Proposed internal milestones, not fair deadlines

| Period | Engineering milestone | Student research / administrative milestone | Exit condition |
| --- | --- | --- | --- |
| September 18-30, 2026 | Scale/loss audit and tested candidate implementation; environment/provenance snapshot | Sponsor reviews current work, eligibility, assistance, approvals, and prior-work boundaries; student begins verified reading notes | Next pilot can run without an avoidable implementation confound |
| October 2026 | Fresh matched dense/no-message pilots; training telemetry; scoped fixes only | Register early when portal opens; student records chosen question, rationale, decisions, and safety/paperwork status | Reliable enough baseline and useful communication control, or an explicit scope reduction |
| Early November | Freeze task, selector semantics, primary K, budgets, source snapshot, checkpoint rule | Student and sponsor review main experimental design and analysis choices | Main experiment no longer being redesigned during measurement |
| November-mid December | Main attention/random comparison with controls and adequate independent training replication | Student analyzes evidence, checks prior-work differences, and keeps failure records; secure signatures before break | Main result and uncertainty available in time for application materials |
| Mid December-early January | Verify final artifacts and figures; limited prespecified analyses, no untracked tuning | Student writes and checks required materials in their own words; verify every reference; submit ahead of the confirmed deadline | Complete application with a traceable evidence chain |
| January 12-February 27, 2027 | Reproduce key results; optional approved, prespecified robustness work clearly separated from submitted data | Prepare for online preliminary review; confirm rules on updating materials; practice defending methods and limitations | Reproducible study and clear ownership, not just a polished demo |
| Late February-March 19 | Freeze presentation artifacts and backups; avoid last-minute framework changes | Student prepares display, explanation, required forms, and judge practice | Every plotted number traceable to raw records |
| March 20, 2027 | No dependence on live training or a fragile demo | TERRA final, if advanced | Accurate presentation of the work actually completed |
| After TERRA, if selected | Reproduction and permitted data collection only | Confirm ISEF instructions and approval boundaries before extending work | No unapproved project redesign |

If the dense reliability gate is still failing in early November, reduce scope before multiplying conditions. A narrower, well-controlled analysis of learnability and communication dependence is more defensible than an unfinished large matrix. The competition's acceptance or award decisions remain outside our control.

## 10. Eligibility, Authorship, and Preparation

### Resolve with the sponsor and fair now

TERRA's application page currently describes eligibility as grades 9-12 attending school in NYC's five boroughs. Confirm your own eligibility, the active portal, and all submission requirements directly; this document does not establish eligibility. [TERRA NYC application](https://tnycfair.org/apply.html)

The 2027 ISEF rules list Forms 1, 1A, 1B, the research plan/addendum, and **Student Support Disclosure Form 2A** for every project. Other forms depend on the work and setting. Prior research may require continuation documentation; the current rules limit judged work to twelve continuous months within the stated January 2026-May 2027 window. After an affiliated fair, methodology changes are restricted. Ask the sponsor/SRC how these apply here. [ISEF 2027 rules](https://www.societyforscience.org/isef/international-rules/rules-for-all-projects/)

The forms page explains timing and exceptions for signatures. Work has already occurred: present its actual dates and assistance history, and ask how to handle any missing documentation. Never backdate approvals. Use current forms, not assumptions from an earlier competition year. [ISEF forms and timing](https://www.societyforscience.org/isef/forms/)

Simulation-only work does not justify automatically skipping review. Adding hardware, human testing, or a new work setting can change the requirements. The sponsor/SRC should determine the applicable review; the code assistant cannot certify compliance. [TERRA guides and forms](https://tnycfair.org/formsandguides.html)

### AI assistance and student ownership

The current ISEF rules hub links an AI-use table dated October 2025. It permits AI-generated code with explicit identification of affected portions and a prompt log. It prohibits AI initially authoring submission documents and producing the student's conclusions/future steps; data interpretation belongs to the student. Local fairs may be stricter. Do not submit this handoff, its interpretations, or its proposed research sequence as your independent work. Ask TERRA/SRC how to document the assistance already provided and what further assistance is permitted. [Official AI-use table](https://sspcdn.blob.core.windows.net/files/Documents/SEP/ISEF/2026/Rules/Generative-AI-Use-Table.pdf), [current rules hub](https://www.societyforscience.org/isef/international-rules/)

Maintain an honest contribution ledger with: inherited components and their original authors/license; changes you designed and implemented; AI-generated or AI-modified portions; help from mentors; commands/experiments actually run; and interpretations you independently checked. Do not retroactively label AI-written modules as entirely student-written. Keep the conversation/prompt history and annotate where you accepted, rejected, tested, or revised assistance.

### Student preparation checklist

- [ ] Explain one complete observation-to-action pass, including what a non-observer does and does not know.
- [ ] Explain PPO at the level needed to defend the policy/critic distinction, reward shaping, stochastic rollout versus mean-action evaluation, and the value-loss scaling issue.
- [ ] Draw the candidate graph, selected graph, self path, and one-layer/multilayer information flow without relying on code generated by someone else.
- [ ] Explain why K=0 differs from no-message, why dense means in-range, and why fully connected does not remove top-K choices.
- [ ] Explain why 50 evaluation episodes are not 50 independent training runs and why two seed pairs are not final significance evidence.
- [ ] Reproduce a result from a saved checkpoint and trace a figure back to raw episode rows and source hashes.
- [ ] State precisely which work was inherited, yours, AI-assisted, and mentor-assisted.
- [ ] Independently read relevant original research, verify bibliographic details, and identify what this experiment adds or fails to add.
- [ ] Prepare your own account of unexpected results, failed approaches, alternative explanations, and limitations.
- [ ] Practice answering what evidence would change your mind, including a reproducible random-selection advantage.

For the display and interview, prioritize a clear question, readable method diagram, per-seed results with uncertainty, one informative failure/mechanism example, accurate cost accounting, and explicit limitations. Presentation cannot compensate for invalid controls or uncertain authorship. Do not claim measured hardware performance using a simulation illustration.

## 11. Reproduction and Artifact Index

### Existing verification commands

Run from the project root using the installed environment. These checks do not start a training sweep:

```bash
venv314/bin/python test_research.py
venv314/bin/python test_random_mode_determinism.py
venv314/bin/python train.py --help
venv314/bin/python run_sweep.py --help
```

The main suite passed 25 tests when this handoff was first written and **31 tests after the scaling implementation**. It covers environment replay/timing/clipping/reward basics, graph routing, gradients, export behavior, robot-frame invariance, trajectory neutrality, sweep settings, comparison guards, hidden-goal boundaries, and scale invariance/compatibility. Passing does not validate PyBullet, real networking, or long-run PPO convergence.

The separate random-routing script passed all four acceptance checks. The original private-goal paired comparison reproduced the reported 0.48 difference. The later implementation includes a one-iteration, 128-environment-step smoke run on development scenarios 80000-80001 with no successes, solely to check software integration. No reserved final-test scenarios were consumed.

### Print a reproduction schedule without launching it

The following prints the unscaled private-goal pilot configuration, not a test of the 0.01 candidate. Omit `--execute` to avoid training. A real execution requires a fresh root and, in restricted environments, permission for Ray's processes/network bindings. Source and metadata have changed since the historical runs, so this is a new execution, not an identical historical artifact.

```bash
venv314/bin/python run_sweep.py \
  --root runs/private_goal_reproduction_new \
  --backend kinematic --seeds 104 105 --conditions dense no_comm \
  --iterations 80 --num-robots 8 --workers 4 --comm-radius 1.8 \
  --goal-observers 1 --goal-spawn-mode independent \
  --observation-frame robot --max-steps 250 \
  --reward-scale 1.0 \
  --train-batch-size 512 --epochs 2 --gnn-num-layers 1 \
  --eval-interval 10 --eval-episodes 20 --development-episodes 50
```

Reproducing the old protocol under later changed source is a new run, not a byte-identical replay of the saved experiment. Use saved source snapshots for exact-version work and record dependency differences. Do not resume old checkpoints across a changed reward/model implementation by bypassing compatibility checks.

### Recompute the existing paired success summary

This reads saved evaluations and prints a comparison without overwriting them:

```bash
venv314/bin/python compare_evaluations.py \
  --left pilot_runs/private_goal_controls_20260917/dense_s104/development.json \
         pilot_runs/private_goal_controls_20260917/dense_s105/development.json \
  --right pilot_runs/private_goal_controls_20260917/no_comm_s104/development.json \
          pilot_runs/private_goal_controls_20260917/no_comm_s105/development.json \
  --metric success
```

### Read these artifacts in order

| Artifact | What it establishes |
| --- | --- |
| [README](README.md) | Current commands, configuration semantics, major limits |
| [World-frame continuation](pilot_runs/dense_credit_s103_continued/DEVELOPMENT.md) | What extra training did and did not improve |
| [Robot-relative pilot](pilot_runs/dense_robot_frame_s103/DEVELOPMENT.md) | Failure diagnosis and matched-budget representation experiment |
| [All-goal protocol](pilot_runs/robot_frame_replication_20260917/PROTOCOL.md) and [results](pilot_runs/robot_frame_replication_20260917/RESULTS.md) | Replication plus no-message control |
| [Independent-goal feasibility](pilot_runs/robot_frame_replication_20260917/independent_task_feasibility.json) | Privileged-script feasibility, not learned communication |
| [Private-goal protocol](pilot_runs/private_goal_controls_20260917/PROTOCOL.md) and [schedule](pilot_runs/private_goal_controls_20260917/schedule.json) | Prespecified gates and exact commands |
| [Private-goal results](pilot_runs/private_goal_controls_20260917/RESULTS.md) and [summary](pilot_runs/private_goal_controls_20260917/summary.json) | Latest matched control evidence and unresolved reliability |
| [Paired success](pilot_runs/private_goal_controls_20260917/comparison_success.json) and [distance](pilot_runs/private_goal_controls_20260917/comparison_distance.json) | Per-seed effects and descriptive bootstrap summaries |
| [Information analysis](pilot_runs/private_goal_controls_20260917/analyze_information.py) | Goal-observation value-path accounting |
| [Value audit](pilot_runs/private_goal_controls_20260917/audit_value_scale.py) | Terminal target/prediction capture and clipped-loss gradient check |
| [Consistency checker](pilot_runs/private_goal_controls_20260917/summarize_results.py) | Artifact consistency and diagnostic replay agreement |
| [Private-goal plotting script](pilot_runs/private_goal_controls_20260917/plot_results.py) and [figure](pilot_runs/private_goal_controls_20260917/private_goal_controls.png) | Reproducible presentation of development data |

Each completed run directory additionally contains its manifest, metrics, portable policies, and Ray checkpoints. Development JSON records the evaluation protocol and episode rows. Do not overwrite those files to make a later experiment appear compatible.

## 12. Next-Session Checklist

### Immediate code work

- [x] Re-read the installed PPO loss and preserve its version/source provenance.
- [x] Implement uniform scaling as an opt-in candidate, keeping the default at 1.0 and value clip at 10.0.
- [x] Carry the setting through configuration, logging, exports, resume, and sweeps.
- [x] Test physics invariance, reward arithmetic, valid settings, terminal-target gradient arithmetic, and old-export compatibility.
- [x] Instrument real learner value errors, not only deterministic terminal replay.
- [x] Write the new pilot protocol before training; use fresh output directories and retain all outcomes.
- [x] Run matched dense/no-message pilots and assess the gate: scaling failed to improve reliability.
- [x] Implement and test a default-preserving value-clip option; freeze a clip-only development protocol before further training.
- [x] Finish the [local-only clip pilot](pilot_runs/value_clip_controls_20260921_local/RESULTS.md), validate all budgets, and report its outcomes. The [first attempt](pilot_runs/value_clip_controls_20260921/FAILURE.md) failed after LAN-addressed Ray connectivity was lost and is excluded from the retry.
- [ ] Diagnose saved dense-policy regressions with paired development trajectories before another hyperparameter intervention.
- [ ] Obtain replicated reliable transport before expanding to the main sparsification conditions.

### Immediate student decisions

- [ ] Confirm TERRA cycle/deadlines, eligibility, sponsor, forms, and treatment of work already performed.
- [ ] Review AI-assistance limits and establish an accurate contribution/prompt log.
- [ ] Independently read and explain the current implementation and results.
- [ ] Decide and justify the primary comparison and acceptable claim scope with the sponsor.
- [ ] Keep final-test scenarios uninspected until the chosen protocol is frozen.

**Bottom line:** the project has progressed from unreliable transport and confounded comparisons to a more auditable simulator, substantially better basic control, and an initial private-goal communication benefit. The next unresolved engineering problem is critic/training reliability. The central learned-selection-versus-random question remains unanswered, and the roadmap should preserve the possibility of any scientifically honest outcome.
