# Multi-Robot Communication Experiments

A research simulator using PyTorch graph attention and Ray RLlib PPO.
It is not a validated hardware controller or a demonstrated state-of-the-art method.

For the consolidated change history, development results, remaining code work, and
competition preparation milestones, see the
[project status and TERRA roadmap](PROJECT_STATUS_AND_TERRA_ROADMAP.md)
(September 18, 2026; AI-assisted engineering documentation, not a competition submission).
The subsequent [feedback review and first implementation](FEEDBACK_REVIEW_20260918.md)
records the opt-in reward-scaling change, verification, and remaining research gates.

## Version 2 changes

- Explicit `pybullet` or `kinematic` backend; missing PyBullet never silently changes physics.
- `transport_v2` rewards distance progress, subtracts action effort and time costs, and
  grants a terminal completion bonus. Development shaping also rewards decreasing mean
  robot-payload distance and individual velocity contributions toward the goal while
  enough robots are near the payload. Recurring connectivity/contact rewards are removed.
  `legacy` retains those reward terms for controlled ablations, but does not recreate old runs.
- `--reward-scale` defaults to 1.0 and uniformly scales every learning reward, including
  completion. A value such as 0.01 is an opt-in development treatment, not a proven fix.
  Physics and raw components are unchanged. Evaluation reports learning
  `return_per_agent` and `raw_return_per_agent`. Use fresh training for scale changes.
- Observations now have 27 features, including sine/cosine heading and a goal-observer
  flag. Earlier 26-feature weights are incompatible. Models use a separate global critic
  encoder in every condition. Export format 2 alone does not identify the task version;
  compare environment configuration, model configuration and source hashes.
- Per-environment seeded RNG; randomized goal direction, initial headings and spawn jitter.
  Goal distance and spawn radius are explicit settings. Default goal distance is now 4.
- Dense, no-communication, attention, random, nearest-distance and Gumbel routing.
  K counts non-self neighbors. K=0 retains only self; this differs from bypassing the GNN.
- Gumbel routing uses observation-stored uniform draws, a hard budget in the forward pass
  and a relaxed surrogate gradient. All layers reuse the first layer's routing gate.
  This is a biased gradient estimator, not a guarantee of better learning.
- Random scores default to changing each environment step, while replay stays fixed.
  `--random-mask-scope episode` holds scores through an episode; selected sets can still
  change when proximity changes.
- Portable strict checkpoints, run manifests with source hashes and library versions,
  structured metrics and fixed-seed validation. Existing run directories are rejected.

Hard attention routing remains a legitimate learned-score baseline: the index operation
has no derivative, but shared score parameters still learn through retained values.
It is not correct to conclude that a hard selector cannot learn.

## Run and verify

Use the project's installed environment, for example:

```bash
venv314/bin/python test_research.py
venv314/bin/python test_ppo_diagnostics.py
venv314/bin/python test_reward_scale_comparison.py
venv314/bin/python test_random_mode_determinism.py
venv314/bin/python validate_task.py
venv314/bin/python train.py --backend kinematic --num-robots 8 --comm-radius 1.8 --top-k 2 --topk-mode attention --checkpoint-dir runs/attention_k2_s0
```

Use `--backend pybullet` only with PyBullet installed. The PyBullet surrogate uses
force-driven sphere bodies, not a validated differential-drive vehicle. Its transport
feasibility requires separate checks. The kinematic backend uses proximity-weighted
velocity drag; by default at least two robots must be within the contact radius before
the payload moves. These are proximity counts, not rigid-body contacts or proof that
two robots contribute force. Spawn radius defaults to 1 and communication radius to 1.8.
Neither backend proves that communication is necessary for the task.

Training defaults to CPU, one Torch thread and two rollout workers. Set `--gpus`,
`--torch-threads` and `--num-workers` explicitly to change resource use.
A short smoke run validates software execution, not learned transport.

`--observation-frame robot` is an opt-in representation experiment. Each actor's
position, vector features, neighbor poses and edge vectors are expressed relative
to its own XY pose; body-relative lidar and masked goals are preserved. This removes
dependence on a common world rotation/translation without changing the task or reward.
It uses the existing dense graph computation, not a deployed communication protocol.
The default `world` frame preserves existing exports. Robot-frame policies require
fresh training; do not reinterpret old world-frame weights as robot-frame weights.
The frame is recorded in the model configuration and portable export.

Continue a Ray checkpoint into a fresh, empty run directory with
`--resume-from OLD_RUN/ray_00040`. In that mode, `--max-iterations` is the
number of additional iterations. The new manifest records the source checkpoint.
Repeat the original training arguments: resume rejects changes to training settings
or environment/model source files. Validation scheduling and output location may change.
Resuming restores RLlib training state but is not guaranteed to reproduce an uninterrupted
run's environment trajectories or random-number stream exactly.

`--goal-observers -1` gives every robot the goal. Positive counts mask the goal vector
for other robots. Use `--goal-spawn-mode independent` for private-goal experiments:
it draws goal direction separately from spawn-ring rotation. The default `coupled`
mode retains historical seeded scenarios and their geometric cue. The sweep generator
defaults to all-goal observations and rejects private-goal sweeps in coupled mode.
Removing this cue alone does not establish that communication is necessary; compare
with no-communication controls before drawing that conclusion.

`validate_task.py` compares a privileged scripted controller with random actions to
check feasibility. Scripted success does not establish PPO learnability. The development
pilots in `pilot_runs` use repeated validation seeds, not an untouched test set.

## PPO value diagnostics

`train.py` and `run_sweep.py` accept `--vf-clip-param` (default `10.0`), a finite
positive cap on squared critic error, not gradient clipping. The chosen value
is preserved in optimizer metadata and portable exports; resume rejects changes.
`compare_value_clips.py --clips LEFT RIGHT` permits only that declared optimizer
change, keeping task, routing, source and all other training settings matched.
Run `venv314/bin/python -m unittest test_value_clips` for the focused checks.
The [clip-only pilot protocol](pilot_runs/value_clip_controls_20260921_local/PROTOCOL.md)
tests 10 against 1,000,000 at reward scale 1.0. A higher cap is experimental,
not a recommended new default.

The [completed clip-only results](pilot_runs/value_clip_controls_20260921_local/RESULTS.md)
show dense success of 88%/14% at clip 10 versus 26%/8% at clip 1,000,000
(seeds 110/111). All eight retry jobs passed artifact checks. Keep clip 10;
the next diagnostic priority is late-policy degradation, not the main sparsification sweep.

Training now rejects non-advancing sample counts and missing/nonfinite learner
losses before exporting an apparent successful iteration. A LAN-addressed Ray
failure motivated this guard; see the preserved [failure audit](pilot_runs/value_clip_controls_20260921/FAILURE.md).
The retry runner binds its single-machine cluster to loopback, checks that
address, and starts all eight conditions fresh. Do not pool the failed attempt.

The [September 21 pilot results](pilot_runs/reward_scale_controls_20260921/RESULTS.md)
show that reward scale 0.01 removed observed value saturation but lowered dense
success from 88%/48% to 12%/30% across two development seeds. Scale 1.0 remains
the default; transport reliability is not yet resolved. The report documents
one evaluation-only recovery and its missing final optimizer diagnostics.

`ppo_diagnostics.py` subclasses the installed PPO policy, runs its original loss,
then measures cached predictions/logits with gradients disabled. It adds no model
forward pass or random draws. Tests compare exact loss values, parameter gradients,
and Torch RNG state with stock RLlib across the supported routing conditions.
The diagnostics require the current feed-forward model and an enabled critic.

`metrics.jsonl` includes unclipped squared value error, absolute value error,
the fraction whose squared error exceeds `vf_clip_param`, target/prediction
magnitudes, and policy probability ratios outside the PPO clipping window.
The last quantity does not account for the advantage-dependent choice of the
surrogate-loss branch. These are measurements during optimizer updates.

RLlib averages statistics across loss minibatches. Count fields are mean counts
per minibatch, including repeated PPO epochs; they are not unique rollout counts.
Terminal saturation uses the ratio of averaged saturated-terminal and terminal
counts, and is null when there are no true terminal samples. Time-limit
truncations are excluded from those terminal counts. Keep `ppo_diagnostics.py`
with Ray checkpoints; its source hash is checked on resume. Portable actor
exports continue to use the ordinary model loader.

The [reward-scale protocol](pilot_runs/reward_scale_controls_20260921/PROTOCOL.md)
fixes eight development runs before training. `compare_reward_scales.py` requires
two explicit scales and matching architecture/routing, source, budget, and scenarios.
It permits only the declared environment reward-scale change and rejects scaled
learning return as a cross-scale endpoint. The ordinary comparison CLI still
requires identical environment settings.

## Evaluate checkpoints

```bash
venv314/bin/python evaluate.py runs/attention_k2_s0/policy_00300.pt --episodes 50 --seed-start 100000 --output evaluation_s0.json
venv314/bin/python analyze_attention_entropy.py runs/attention_k2_s0/policy_00300.pt --episodes 10 --output routing_s0.json
```

For transport failures, record full development trajectories and compare the original
episode horizon with extra time using the same deterministic policy:

```bash
venv314/bin/python diagnose_transport.py pilot_runs/dense_credit_s103_continued/policy_00081.pt --episodes 50 --seed-start 80000 --extended-steps 750 --output transport_diagnostic.json
```

This saves positions, headings, actions, contact counts, payload motion and tail
statistics. The extended rollout's prefix reproduces the original horizon because
`max_steps` affects truncation only. Extra-time success is a changed evaluation
condition, not evidence that training improved. Velocity agreement uses post-step
proximity weights and is descriptive, not an exact force decomposition.

Validation seeds start at 50000. Reserve evaluation seeds (default 100000) for held-out
comparisons after development choices are fixed. Actions use clipped Gaussian means;
routing remains stochastic but reproducible through seeded observation draws.

Evaluation records success, final payload distance, progress, duration, per-agent return,
action effort, logical directed deliveries per message-passing round and measured CPU
inference time. Logical payload bytes assume one float32 hidden vector per retained edge.
This excludes neighbor discovery, routing-score exchange, headers, retries and broadcast
sharing. Scoring and message construction still use dense tensors; fewer retained edges
do not establish reduced runtime or real network bandwidth.

Robustness options:
- `--num-robots` and `--comm-radius`: evaluation configuration changes without retraining.
- `--candidate-loss`: directed candidate links unavailable before selection, allowing rerouting;
  this is not post-transmission packet loss.
- `--delay`: stale node features with current geometry/local observations; startup uses the
  initial snapshot until the history fills. This is not a complete network delay model.
- `--noise`: independent Gaussian perturbations of observation features, in their raw units.
- `--diagnostic-interval`: zero individual first-layer selected messages while holding
  routing/normalization fixed, reporting changes in action means and log standard deviations.
  This measures action sensitivity, not causal benefit to task return or an oracle selector.

Entropy reports concentration, selection churn, score margins and overlap with random
selection on trained-policy trajectories. High entropy alone does not imply random or
useless rankings. Analysis rejects missing/incompatible weights rather than substituting
random parameters.

## Matched experiments

```bash
venv314/bin/python run_sweep.py --root runs/sweep_v2 --backend kinematic --budgets 1 2 --seeds 0 1 2 3 4
```

This prints commands. Add `--execute` to run them sequentially. Each condition uses the
same training budget and seed list. Old shell launchers are disabled because they reused
and truncated historical log paths. No long sweep is started automatically.

To reproduce the robot-relative transport pilot and its no-communication control:

```bash
venv314/bin/python run_sweep.py --root runs/transport_replication --backend kinematic --seeds 104 105 --conditions dense no_comm --iterations 40 --num-robots 8 --workers 4 --goal-observers -1 --observation-frame robot --max-steps 250 --train-batch-size 512 --epochs 2 --gnn-num-layers 1 --eval-episodes 20 --development-episodes 50
```

The runner saves the complete schedule and source hashes, forwards the same training
settings to every condition, rejects existing logs, and stops if task/model/training
sources change mid-sweep. `--development-episodes` evaluates final exports on seeds
starting at 80000 by default; it does not consume the reserved final-test set.

The completed [transport replication](pilot_runs/robot_frame_replication_20260917/RESULTS.md)
documents two matched dense/no-message seeds. It supports improved learnability but
does not establish that the current all-goal task benefits from communication.

The subsequent [private-goal controls](pilot_runs/private_goal_controls_20260917/RESULTS.md)
show a communication benefit in two development seeds, but dense success remains
variable (22% and 74%). The report includes message-removal checks, an information
boundary test and a value-loss scaling audit. Its reliability gate has not passed;
the attention-versus-random study is still pending.

After separately evaluating each trained policy on the same episode seeds:

```bash
venv314/bin/python compare_evaluations.py --left attention_s0.json attention_s1.json --right random_s0.json random_s1.json --metric success
```

The comparison rejects mismatched settings, iterations, code hashes, duplicate seeds
and unpaired episodes. It bootstraps differences across independent training seeds,
not episodes, and reports a paired standardized effect. Small seed counts produce
unstable intervals; there is no universal seed-count threshold or multiple-testing
correction. Report all prespecified comparisons and retain failed runs.

## Scope and provenance

Existing historical logs/checkpoints are preserved. Do not pool them with version 2.
`summarize_sweep.py` is a historical training-log utility, not held-out statistical evidence.
Track your own experiment decisions and interpretation. Code changes in this iteration
were AI-assisted; retain the conversation/prompt log and identify these portions when
documenting assistance. This README is software documentation, not a competition research
plan, abstract, poster, or student-authored scientific conclusion.
