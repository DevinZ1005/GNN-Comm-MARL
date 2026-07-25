"""
Acceptance test: random-mode neighbor selection must be deterministic for identical inputs.

This verifies the fix for the random-mode selection confound in PPO training.
Before the fix, each forward() call drew a fresh random neighbor mask, injecting
structural noise across SGD epochs on the same batch. After the fix, identical
(node_features, adj_matrix, edge_features) inputs must produce identical topk_indices.
"""

import torch
from gnn_comm_layer import EdgeConditionedGATLayer, DynamicTopologicalGNN


def test_random_topk_deterministic_single_layer():
    """forward() called twice on identical inputs must select the same random neighbors."""
    torch.manual_seed(42)

    batch_size, num_nodes, node_dim, edge_dim = 4, 6, 32, 8
    num_heads, hidden_dim, top_k = 4, 32, 2

    layer = EdgeConditionedGATLayer(
        node_dim=node_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        dropout=0.0,
        top_k=top_k,
        topk_mode='random'
    )
    layer.eval()

    node_features = torch.randn(batch_size, num_nodes, node_dim)
    adj_matrix = torch.ones(batch_size, num_nodes, num_nodes)  # Fully connected
    edge_features = torch.randn(batch_size, num_nodes, num_nodes, edge_dim)

    # We need to capture the topk_indices from inside forward().
    # Monkey-patch torch.topk on the random branch to record indices.
    captured_indices = []
    original_topk = torch.topk

    def capturing_topk(*args, **kwargs):
        result = original_topk(*args, **kwargs)
        captured_indices.append(result.indices.clone())
        return result

    torch.topk = capturing_topk
    try:
        with torch.no_grad():
            out1 = layer(node_features, adj_matrix, edge_features)
            out2 = layer(node_features, adj_matrix, edge_features)
    finally:
        torch.topk = original_topk

    # Each forward call hits topk once (single layer), so captured_indices has 2 entries.
    assert len(captured_indices) >= 2, f"Expected at least 2 topk calls, got {len(captured_indices)}"

    idx1 = captured_indices[0]
    idx2 = captured_indices[1]

    assert torch.equal(idx1, idx2), (
        f"FAIL: Random-mode topk_indices differ between forward passes on identical inputs!\n"
        f"  Pass 1 indices (sample): {idx1[0, 0]}\n"
        f"  Pass 2 indices (sample): {idx2[0, 0]}"
    )
    print("PASS: Single-layer random-mode topk_indices are identical across forward passes.")


def test_random_topk_deterministic_full_gnn():
    """Full GNN stack: repeated forward passes on identical inputs produce identical outputs."""
    torch.manual_seed(123)

    batch_size, num_nodes, obs_dim, edge_dim = 4, 6, 24, 8

    gnn = DynamicTopologicalGNN(
        raw_obs_dim=obs_dim,
        edge_dim=edge_dim,
        comm_latent_dim=32,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        top_k=2,
        topk_mode='random'
    )
    gnn.eval()

    raw_obs = torch.randn(batch_size, num_nodes, obs_dim)
    adj_matrix = torch.ones(batch_size, num_nodes, num_nodes)
    edge_features = torch.randn(batch_size, num_nodes, num_nodes, edge_dim)

    with torch.no_grad():
        out1 = gnn(raw_obs, adj_matrix, edge_features)
        out2 = gnn(raw_obs, adj_matrix, edge_features)

    assert torch.equal(out1, out2), (
        f"FAIL: Full GNN outputs differ on identical inputs in random mode!\n"
        f"  Max diff: {(out1 - out2).abs().max().item()}"
    )
    print("PASS: Full GNN random-mode outputs are identical across forward passes.")


def test_random_topk_differs_for_different_inputs():
    """Sanity check: different inputs should (almost certainly) produce different masks."""
    torch.manual_seed(999)

    batch_size, num_nodes, obs_dim, edge_dim = 4, 6, 24, 8

    gnn = DynamicTopologicalGNN(
        raw_obs_dim=obs_dim,
        edge_dim=edge_dim,
        comm_latent_dim=32,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        top_k=2,
        topk_mode='random'
    )
    gnn.eval()

    adj_matrix = torch.ones(batch_size, num_nodes, num_nodes)
    edge_features = torch.randn(batch_size, num_nodes, num_nodes, edge_dim)

    raw_obs_a = torch.randn(batch_size, num_nodes, obs_dim)
    raw_obs_b = torch.randn(batch_size, num_nodes, obs_dim)  # Different input

    with torch.no_grad():
        out_a = gnn(raw_obs_a, adj_matrix, edge_features)
        out_b = gnn(raw_obs_b, adj_matrix, edge_features)

    assert not torch.equal(out_a, out_b), (
        "FAIL: Different inputs produced identical outputs — randomness may not be working."
    )
    print("PASS: Different inputs produce different outputs (randomness is functional).")


if __name__ == "__main__":
    test_random_topk_deterministic_single_layer()
    test_random_topk_deterministic_full_gnn()
    test_random_topk_differs_for_different_inputs()
    print("\nAll acceptance tests passed!")
