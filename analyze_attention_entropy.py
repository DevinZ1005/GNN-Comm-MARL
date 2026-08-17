"""
Attention Entropy Analysis Script (Phase 1.1).

Diagnoses whether learned attention in the GNN communication layer has collapsed
to near-uniform distributions (≈ random), which would explain why random neighbor
selection matches or outperforms attention-based top-k.

Usage:
    python analyze_attention_entropy.py [--checkpoint-dir ./checkpoints/attn8_k2_s0]
                                        [--num-robots 8]
                                        [--num-batches 20]
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F

from env_core import MultiRobotPhysicsEnv
from marl_agent import GNNMARLModel


def compute_attention_entropy(
    model: GNNMARLModel,
    obs_batch: Dict[str, torch.Tensor],
    num_robots: int
) -> Dict[str, np.ndarray]:
    """
    Run a forward pass through the GNN layer with hooks to capture raw attention
    scores, then compute softmax entropy per node per head.

    Returns a dict with:
        - 'pre_mask_entropy': entropy of raw_scores before adjacency masking (H, N, N)
        - 'post_mask_entropy': entropy after adjacency mask but before top-k
        - 'raw_scores': the raw attention score tensor for inspection
        - 'masked_scores': scores after adjacency masking
    """
    captured = {}

    # Hook into the first GAT layer to capture scores at key points
    first_gat = model.gnn_layer.gat_layers[0]
    original_forward = first_gat.forward

    def hooked_forward(node_features, adj_matrix, edge_features,
                       random_comm_mask=None, shared_topk_indices=None):
        batch_size, num_nodes, _ = node_features.shape

        # Reproduce the attention computation to capture intermediate tensors
        q = first_gat.proj_q(node_features).view(batch_size, num_nodes, first_gat.num_heads, first_gat.head_dim)
        k = first_gat.proj_k(node_features).view(batch_size, num_nodes, first_gat.num_heads, first_gat.head_dim)
        e = first_gat.proj_e(edge_features).view(batch_size, num_nodes, num_nodes, first_gat.num_heads, first_gat.head_dim)

        q_expand = q.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
        k_expand = k.unsqueeze(1).expand(-1, num_nodes, -1, -1, -1)

        attn_input = torch.cat([q_expand, k_expand, e], dim=-1)
        raw_scores = (attn_input * first_gat.attn_vector).sum(dim=-1)
        raw_scores = F.leaky_relu(raw_scores, negative_slope=first_gat.leaky_relu_slope)

        # Capture raw scores before any masking
        captured['raw_scores'] = raw_scores.detach().cpu()

        # Compute entropy of raw scores (pre-mask) via softmax over sender dim
        raw_probs = F.softmax(raw_scores, dim=2)  # (B, N, N, H)
        raw_entropy = -(raw_probs * torch.log(raw_probs + 1e-12)).sum(dim=2)  # (B, N, H)
        captured['pre_mask_entropy'] = raw_entropy.detach().cpu().numpy()

        # Apply adjacency mask
        eye = torch.eye(num_nodes, device=adj_matrix.device, dtype=adj_matrix.dtype).unsqueeze(0)
        adj_with_loops = torch.clamp(adj_matrix + eye, 0.0, 1.0)
        adj_mask = adj_with_loops.unsqueeze(-1)
        masked_scores = raw_scores.masked_fill(adj_mask == 0, -1e9)

        captured['masked_scores'] = masked_scores.detach().cpu()

        # Compute entropy of masked scores (post-mask, pre-topk)
        masked_probs = F.softmax(masked_scores, dim=2)  # (B, N, N, H)
        masked_entropy = -(masked_probs * torch.log(masked_probs + 1e-12)).sum(dim=2)  # (B, N, H)
        captured['post_mask_entropy'] = masked_entropy.detach().cpu().numpy()

        # Compute max entropy for comparison (uniform over neighbors)
        neighbor_counts = adj_with_loops.sum(dim=2)  # (B, N)
        max_entropy = torch.log(neighbor_counts + 1e-12)  # (B, N)
        captured['max_entropy'] = max_entropy.detach().cpu().numpy()
        captured['neighbor_counts'] = neighbor_counts.detach().cpu().numpy()

        # Now run the actual forward pass normally
        return original_forward(node_features, adj_matrix, edge_features,
                                random_comm_mask=random_comm_mask,
                                shared_topk_indices=shared_topk_indices)

    first_gat.forward = hooked_forward

    # Run the model forward
    with torch.no_grad():
        model.forward({"obs": obs_batch}, [], torch.ones(obs_batch["local_obs"].shape[0]))

    # Restore original forward
    first_gat.forward = original_forward

    return captured


def generate_obs_batches(
    num_robots: int,
    comm_radius: float,
    num_batches: int,
    max_steps_per_episode: int = 50
) -> List[Dict[str, torch.Tensor]]:
    """
    Generate observation batches by running the environment with random actions.
    """
    env = MultiRobotPhysicsEnv({
        "num_robots": num_robots,
        "comm_radius": comm_radius,
        "max_steps": max_steps_per_episode
    })

    batches = []
    obs, _ = env.reset()
    steps_in_episode = 0

    for _ in range(num_batches):
        # Collect one observation per agent and stack into a batch
        agent_obs_list = []
        for agent_id in sorted(obs.keys()):
            agent_obs = {k: torch.tensor(v).unsqueeze(0) for k, v in obs[agent_id].items()}
            agent_obs_list.append(agent_obs)

        # Stack into batch: (num_robots, ...)
        batch = {}
        for key in agent_obs_list[0]:
            batch[key] = torch.cat([ao[key] for ao in agent_obs_list], dim=0)
        batches.append(batch)

        # Step with random actions
        actions = {
            f"robot_{i}": np.random.uniform(-1, 1, size=2).astype(np.float32)
            for i in range(num_robots)
        }
        obs, _, terminated, truncated, _ = env.step(actions)
        steps_in_episode += 1

        if terminated.get("__all__", False) or truncated.get("__all__", False) or steps_in_episode >= max_steps_per_episode:
            obs, _ = env.reset()
            steps_in_episode = 0

    return batches


def analyze_checkpoint(
    checkpoint_dir: str,
    num_robots: int = 8,
    comm_radius: float = 3.8,
    num_batches: int = 20
) -> None:
    """
    Load a checkpoint, generate observation batches, and compute attention entropy statistics.
    """
    print(f"\n{'='*70}")
    print(f"Analyzing checkpoint: {checkpoint_dir}")
    print(f"{'='*70}")

    # Build model with same config used in training
    obs_dim = 24
    edge_dim = 8
    model_config = {
        "custom_model_config": {
            "raw_obs_dim": obs_dim,
            "edge_dim": edge_dim,
            "comm_latent_dim": 64,
            "local_hidden_dim": 128,
            "gnn_num_layers": 2,
            "gnn_num_heads": 4,
            "top_k": 2,
            "topk_mode": "attention"
        }
    }

    model = GNNMARLModel(
        obs_space=None,
        action_space=None,
        num_outputs=4,  # 2 actions * 2 (mean + log_std)
        model_config=model_config,
        name="analysis_model"
    )

    # Try to load checkpoint weights
    policy_dir = os.path.join(checkpoint_dir, "policies", "shared_gnn_policy")
    model_weights_path = os.path.join(policy_dir, "model_weights.pth")

    if os.path.exists(model_weights_path):
        state_dict = torch.load(model_weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded model weights from {model_weights_path}")
    else:
        # Try finding weights in alternative locations
        for root, dirs, files in os.walk(checkpoint_dir):
            for f in files:
                if f.endswith(('.pth', '.pt')):
                    print(f"  Found: {os.path.join(root, f)}")
        print(f"WARNING: Could not find model_weights.pth at {model_weights_path}")
        print("Running analysis with randomly-initialized weights as a baseline comparison.")

    model.eval()

    # Generate observation batches
    print(f"\nGenerating {num_batches} observation batches with {num_robots} robots...")
    batches = generate_obs_batches(num_robots, comm_radius, num_batches)

    # Collect entropy statistics
    all_pre_mask_entropy = []
    all_post_mask_entropy = []
    all_max_entropy = []
    all_neighbor_counts = []
    all_entropy_ratios = []

    for batch_idx, batch in enumerate(batches):
        captured = compute_attention_entropy(model, batch, num_robots)

        pre_mask_h = captured['pre_mask_entropy']   # (B, N, H)
        post_mask_h = captured['post_mask_entropy']  # (B, N, H)
        max_h = captured['max_entropy']              # (B, N)
        n_counts = captured['neighbor_counts']       # (B, N)

        all_pre_mask_entropy.append(pre_mask_h)
        all_post_mask_entropy.append(post_mask_h)
        all_max_entropy.append(max_h)
        all_neighbor_counts.append(n_counts)

        # Compute entropy ratio: actual / maximum (1.0 = perfectly uniform = collapsed to random)
        # Average post_mask_entropy across heads for per-node ratio
        avg_post = post_mask_h.mean(axis=-1)  # (B, N)
        ratio = avg_post / (max_h + 1e-12)
        all_entropy_ratios.append(ratio)

    # Aggregate statistics
    pre_mask_all = np.concatenate(all_pre_mask_entropy, axis=0)   # (total_samples, N, H)
    post_mask_all = np.concatenate(all_post_mask_entropy, axis=0)
    max_h_all = np.concatenate(all_max_entropy, axis=0)
    ratios_all = np.concatenate(all_entropy_ratios, axis=0)
    n_counts_all = np.concatenate(all_neighbor_counts, axis=0)

    print(f"\n--- Attention Entropy Statistics ---")
    print(f"Samples analyzed: {pre_mask_all.shape[0]} (nodes × batches)")
    print(f"Number of attention heads: {pre_mask_all.shape[2]}")
    print(f"Average neighbor count: {n_counts_all.mean():.2f} (±{n_counts_all.std():.2f})")
    print()

    print(f"Pre-mask entropy (raw scores, all neighbors):")
    print(f"  Mean:   {pre_mask_all.mean():.4f}")
    print(f"  Std:    {pre_mask_all.std():.4f}")
    print(f"  Min:    {pre_mask_all.min():.4f}")
    print(f"  Max:    {pre_mask_all.max():.4f}")
    print()

    print(f"Post-mask entropy (after adjacency mask, before top-k):")
    print(f"  Mean:   {post_mask_all.mean():.4f}")
    print(f"  Std:    {post_mask_all.std():.4f}")
    print(f"  Min:    {post_mask_all.min():.4f}")
    print(f"  Max:    {post_mask_all.max():.4f}")
    print()

    print(f"Maximum possible entropy (uniform over neighbors):")
    print(f"  Mean:   {max_h_all.mean():.4f}")
    print(f"  Std:    {max_h_all.std():.4f}")
    print()

    print(f"Entropy ratio (actual / max, 1.0 = uniform ≈ random):")
    print(f"  Mean:   {ratios_all.mean():.4f}")
    print(f"  Std:    {ratios_all.std():.4f}")
    print(f"  Median: {np.median(ratios_all):.4f}")
    print(f"  >0.9:   {(ratios_all > 0.9).mean() * 100:.1f}% of nodes")
    print(f"  >0.95:  {(ratios_all > 0.95).mean() * 100:.1f}% of nodes")
    print(f"  <0.5:   {(ratios_all < 0.5).mean() * 100:.1f}% of nodes")
    print()

    # Per-head breakdown
    print(f"Per-head post-mask entropy (averaged across all nodes/batches):")
    for h in range(post_mask_all.shape[2]):
        head_entropy = post_mask_all[:, :, h]
        print(f"  Head {h}: mean={head_entropy.mean():.4f}, std={head_entropy.std():.4f}")

    # Diagnosis
    print(f"\n--- DIAGNOSIS ---")
    mean_ratio = ratios_all.mean()
    pct_near_uniform = (ratios_all > 0.9).mean() * 100

    if mean_ratio > 0.9:
        print(f"⚠️  ATTENTION HAS COLLAPSED TO NEAR-UNIFORM (ratio={mean_ratio:.3f})")
        print(f"   {pct_near_uniform:.0f}% of nodes have entropy ratio > 0.9")
        print(f"   → Attention is effectively random. The top-k selection is arbitrary.")
        print(f"   → Root cause is likely the non-differentiable torch.topk gate.")
        print(f"   → FIX: Implement Gumbel-Softmax relaxation (Phase 2.6)")
    elif mean_ratio > 0.7:
        print(f"⚠️  ATTENTION IS WEAKLY DIFFERENTIATED (ratio={mean_ratio:.3f})")
        print(f"   → Some specialization exists but not enough to outperform random.")
        print(f"   → Consider both reward fixes (Phase 1.3–1.4) AND Gumbel-Softmax.")
    else:
        print(f"✓  Attention IS differentiating (ratio={mean_ratio:.3f})")
        print(f"   → The problem is likely in what attention is being trained toward.")
        print(f"   → Focus on reward fixes (Phase 1.2–1.4) and CTDE critic (Phase 2.2).")


def main():
    parser = argparse.ArgumentParser(description="Analyze GNN attention entropy for collapse diagnosis.")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/attn8_k2_s0",
                        help="Path to a trained attention checkpoint directory.")
    parser.add_argument("--num-robots", type=int, default=8,
                        help="Number of robots (must match checkpoint training config).")
    parser.add_argument("--comm-radius", type=float, default=3.8,
                        help="Communication radius (must match checkpoint training config).")
    parser.add_argument("--num-batches", type=int, default=20,
                        help="Number of observation batches to analyze.")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Analyze all attn8_k2_s* checkpoints.")
    args = parser.parse_args()

    if args.all_checkpoints:
        checkpoint_base = os.path.dirname(args.checkpoint_dir) or "./checkpoints"
        for name in sorted(os.listdir(checkpoint_base)):
            if name.startswith("attn8_k2_s") or name.startswith("real_attn_s"):
                ckpt_path = os.path.join(checkpoint_base, name)
                if os.path.isdir(ckpt_path):
                    analyze_checkpoint(ckpt_path, args.num_robots, args.comm_radius, args.num_batches)
    else:
        analyze_checkpoint(args.checkpoint_dir, args.num_robots, args.comm_radius, args.num_batches)


if __name__ == "__main__":
    main()
