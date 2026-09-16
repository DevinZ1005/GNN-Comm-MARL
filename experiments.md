# Experiment Log

## Historical Runs (Pre-Bugfix) — VOID / INVALIDATED

> [!WARNING]
> **ALL PRE-FIX RESULTS BELOW ARE VOID AND MUST NOT BE CITED OR BUILT UPON.**
> An audit confirmed that these runs were affected by:
> 1. **Bug 1 (Inconsistent out-of-range masking in attention mode)**: In `gnn_comm_layer.py`, `torch.topk` was called on attention scores without masking non-neighbors (`adj_matrix == 0`), allowing out-of-range robots to be selected over valid in-range candidates.
> 2. **Bug 3 (10x Kinematic Time Dilation)**: In `env_core.py`, the headless kinematic fallback executed substeps that double-applied `dt`, diluting physics and reward dynamics by 10x.

| Date | Status | Config | Git Commit | Seeds | Mean Reward (±std) | Drop Frac | Notes |
|------|--------|--------|------------|-------|-------------------|-----------|-------|
| pre-Aug16 | **VOID** | dense, N=8, comm_radius=3.8 | baseline | 0-4 | 8108.96 | 0.0 | Invalidated by Bug 3 (kinematic time dilation) |
| pre-Aug16 | **VOID** | attn, N=8, k=2, comm_radius=3.8 | baseline | 0-4 | 8020.79 | — | Invalidated by Bug 1 (unmasked selection) & Bug 3 |
| pre-Aug16 | **VOID** | random, N=8, k=2, comm_radius=3.8 | baseline | 0-4 | 8501.70 | — | Invalidated by Bug 3; old "random > attention" claim is invalid |

---

## Post-Fix Diagnostics (Intermediate Checks — Not Final Evidence)

| Date | Config | Iterations | Seeds | Mean Reward | Drop Frac | Notes |
|------|--------|------------|-------|-------------|-----------|-------|
| Post-Fix | dense, N=8, comm_radius=3.8 | 150 | 0, 1 | s0: -439.1, s1: -681.4 | 0.0 | 2-seed diagnostic; high variance |
| Post-Fix | attn, N=8, k=2, comm_radius=3.8 | 150 | 0, 1 | s0: -882.9, s1: -720.9 | ~0.45 | Attention beat random on both seeds (+1102 on s0, +107 on s1); suggestive only |
| Post-Fix | rand, N=8, k=2, comm_radius=3.8 | 150 | 0, 1 | s0: -1985.3, s1: -827.8 | ~0.45 | High variance across seeds |

---

## Post-Fix Benchmark Sweep (300 Iterations, 5 Seeds, N=8, comm_radius=3.8, k=2)

*Current benchmark sweep executed with verified fixes (Bug 1 masking, Bug 2 detached Gumbel penalty, Bug 3 kinematic dt, Bug 4 checkpoint loading, and fairness fix `shared_topk_indices`).*

| Condition | Seed | Iterations | Reward Mean | Ep Len | Policy Loss | Drop Frac | Status |
|-----------|------|------------|-------------|--------|-------------|-----------|--------|
| dense | 0 | 300 | pending | pending | pending | pending | Queued |
| dense | 1 | 300 | pending | pending | pending | pending | Queued |
| dense | 2 | 300 | pending | pending | pending | pending | Queued |
| dense | 3 | 300 | pending | pending | pending | pending | Queued |
| dense | 4 | 300 | pending | pending | pending | pending | Queued |
| attn (k=2) | 0 | 300 | pending | pending | pending | pending | Queued |
| attn (k=2) | 1 | 300 | pending | pending | pending | pending | Queued |
| attn (k=2) | 2 | 300 | pending | pending | pending | pending | Queued |
| attn (k=2) | 3 | 300 | pending | pending | pending | pending | Queued |
| attn (k=2) | 4 | 300 | pending | pending | pending | pending | Queued |
| rand (k=2) | 0 | 300 | pending | pending | pending | pending | Queued |
| rand (k=2) | 1 | 300 | pending | pending | pending | pending | Queued |
| rand (k=2) | 2 | 300 | pending | pending | pending | pending | Queued |
| rand (k=2) | 3 | 300 | pending | pending | pending | pending | Queued |
| rand (k=2) | 4 | 300 | pending | pending | pending | pending | Queued |

### Condition Summary (Mean ± Std)
| Condition | Seeds Completed | Mean Reward (±std) | Mean Drop Frac |
|-----------|-----------------|-------------------|----------------|
| dense | 0/5 | pending | pending |
| attn (k=2) | 0/5 | pending | pending |
| rand (k=2) | 0/5 | pending | pending |
