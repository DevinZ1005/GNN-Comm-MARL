# Multi-Robot Communication Experiments

A research simulator using PyTorch graph attention and Ray RLlib PPO.
It is not a validated hardware controller or a demonstrated state-of-the-art method.

## Version 2 changes

- Explicit `pybullet` or `kinematic` backend; missing PyBullet never silently changes physics.
- `transport_v2` rewards distance progress, subtracts action effort and time costs, and
  grants a terminal completion bonus. Development shaping also rewards decreasing mean
  robot-payload distance and individual velocity contributions toward the goal while
  enough robots are near the payload. Recurring connectivity/contact rewards are removed.
  `legacy` retains those reward terms for controlled ablations, but does not recreate old runs.
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

Continue a Ray checkpoint into a fresh, empty run directory with
`--resume-from OLD_RUN/ray_00040`. In that mode, `--max-iterations` is the
number of additional iterations. The new manifest records the source checkpoint.
Repeat the original training arguments: resume rejects changes to training settings
or environment/model source files. Validation scheduling and output location may change.
Resuming restores RLlib training state but is not guaranteed to reproduce an uninterrupted
run's environment trajectories or random-number stream exactly.

`--goal-observers -1` gives every robot the goal. Positive counts mask the goal vector
for other robots. The current spawn ring rotates with the goal, so this option alone
does not establish a private-information task. Validate and remove geometric leakage
before using it to test whether communication is necessary. The sweep generator's
private-goal default is experimental and has not passed that validation.

`validate_task.py` compares a privileged scripted controller with random actions to
check feasibility. Scripted success does not establish PPO learnability. The development
pilots in `pilot_runs` use repeated validation seeds, not an untouched test set.

## Evaluate checkpoints

```bash
venv314/bin/python evaluate.py runs/attention_k2_s0/policy_00300.pt --episodes 50 --seed-start 100000 --output evaluation_s0.json
venv314/bin/python analyze_attention_entropy.py runs/attention_k2_s0/policy_00300.pt --episodes 10 --output routing_s0.json
```

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
