"""
Acceptance test: random-mode neighbor selection must be deterministic for identical inputs.

The random_comm_mask is generated once per env step in env_core.py and passed through
the observation dict. The GNN layer consumes it directly — no internal randomness.
Repeated forward passes with the same mask must produce identical topk selections.
"""

import torch
from gnn_comm_layer import EdgeConditionedGATLayer, DynamicTopologicalGNN


def test_random_topk_deterministic_single_layer():
    """forward() called twice with the same random_comm_mask must select identical neighbors."""
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
    # Pre-computed mask — same one used for both calls (mirrors PPO SGD replay)
    random_comm_mask = torch.rand(batch_size, num_nodes, num_nodes)

    # Capture topk_indices from inside forward()
    captured_indices = []
    original_topk = torch.topk

    def capturing_topk(*args, **kwargs):
        result = original_topk(*args, **kwargs)
        captured_indices.append(result.indices.clone())
        return result

    torch.topk = capturing_topk
    try:
        with torch.no_grad():
            out1 = layer(node_features, adj_matrix, edge_features, random_comm_mask=random_comm_mask)
            out2 = layer(node_features, adj_matrix, edge_features, random_comm_mask=random_comm_mask)
    finally:
        torch.topk = original_topk

    assert len(captured_indices) >= 2, f"Expected at least 2 topk calls, got {len(captured_indices)}"

    idx1 = captured_indices[0]
    idx2 = captured_indices[1]

    assert torch.equal(idx1, idx2), (
        f"FAIL: Random-mode topk_indices differ between forward passes with same mask!\n"
        f"  Pass 1 indices (sample): {idx1[0, 0]}\n"
        f"  Pass 2 indices (sample): {idx2[0, 0]}"
    )
    print("PASS: Single-layer random-mode topk_indices are identical across forward passes.")


def test_random_topk_deterministic_full_gnn():
    """Full GNN stack: repeated forward passes with the same mask produce identical outputs."""
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
    random_comm_mask = torch.rand(batch_size, num_nodes, num_nodes)

    with torch.no_grad():
        out1 = gnn(raw_obs, adj_matrix, edge_features, random_comm_mask=random_comm_mask)
        out2 = gnn(raw_obs, adj_matrix, edge_features, random_comm_mask=random_comm_mask)

    assert torch.equal(out1, out2), (
        f"FAIL: Full GNN outputs differ on identical inputs + same mask!\n"
        f"  Max diff: {(out1 - out2).abs().max().item()}"
    )
    print("PASS: Full GNN random-mode outputs are identical across forward passes.")


def test_random_topk_differs_for_different_masks():
    """Sanity check: different masks should (almost certainly) produce different outputs."""
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

    raw_obs = torch.randn(batch_size, num_nodes, obs_dim)
    adj_matrix = torch.ones(batch_size, num_nodes, num_nodes)
    edge_features = torch.randn(batch_size, num_nodes, num_nodes, edge_dim)

    mask_a = torch.rand(batch_size, num_nodes, num_nodes)
    mask_b = torch.rand(batch_size, num_nodes, num_nodes)

    with torch.no_grad():
        out_a = gnn(raw_obs, adj_matrix, edge_features, random_comm_mask=mask_a)
        out_b = gnn(raw_obs, adj_matrix, edge_features, random_comm_mask=mask_b)

    assert not torch.equal(out_a, out_b), (
        "FAIL: Different masks produced identical outputs — mask isn't being used."
    )
    print("PASS: Different masks produce different outputs (mask is functional).")


def test_missing_mask_raises_error():
    """Calling forward() with topk_mode='random' without a mask must raise ValueError."""
    torch.manual_seed(0)

    layer = EdgeConditionedGATLayer(
        node_dim=32, edge_dim=8, hidden_dim=32, num_heads=4,
        dropout=0.0, top_k=2, topk_mode='random'
    )
    layer.eval()

    node_features = torch.randn(2, 4, 32)
    adj_matrix = torch.ones(2, 4, 4)
    edge_features = torch.randn(2, 4, 4, 8)

    try:
        with torch.no_grad():
            layer(node_features, adj_matrix, edge_features)  # No mask provided
        assert False, "FAIL: Expected ValueError when random_comm_mask is None"
    except ValueError as e:
        assert "random_comm_mask" in str(e)
        print(f"PASS: Missing mask raises ValueError: {e}")


if __name__ == "__main__":
    test_random_topk_deterministic_single_layer()
    test_random_topk_deterministic_full_gnn()
    test_random_topk_differs_for_different_masks()
    test_missing_mask_raises_error()
    print("\nAll acceptance tests passed!")
