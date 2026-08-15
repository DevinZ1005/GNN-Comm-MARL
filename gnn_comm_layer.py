"""
Dynamic Topological Graph Neural Network (GNN) Communication Layer.

This module implements a permutation-invariant, edge-conditioned Graph Attention / Message-Passing
layer designed for decentralized Multi-Agent Reinforcement Learning (MARL). Robots dynamically route
latent communication vectors through physical communication links (determined by Euclidean proximity).

Mathematical & Strategic Reasoning:
1. Decentralized Execution: Global state is hidden. Node i only receives messages from neighbors N(i)
   where ||p_i - p_j|| <= R_comm.
2. Edge-Conditioned Message Passing: Relative spatial vectors (position, velocity differences) are explicitly
   injected into both attention weight computation and message content generation. This allows the network
   to prioritize urgent spatial events (e.g., imminent collision or payload equilibrium shift).
3. Multi-Hop Propagation: Stacking L layers allows 1-hop communication vectors to propagate across L physical
   links, enabling cooperative coordination across occluded or distant agents.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeConditionedGATLayer(nn.Module):
    """
    Single layer of Edge-Conditioned Graph Attention Network (EC-GAT).
    
    Computes attention weights and aggregates messages across dynamic neighborhood topologies:
        m_{ji} = MLP_msg([h_i || h_j || e_ij])
        alpha_{ij} = Softmax_j( LeakyReLU( a^T [W_q h_i || W_k h_j || W_e e_ij] ) )
        h_i' = LayerNorm( h_i + W_out * SUM_{j in N(i)} (alpha_{ij} * m_{ji}) )
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        leaky_relu_slope: float = 0.2,
        top_k: Optional[int] = None,
        topk_mode: str = 'attention'
    ) -> None:
        """
        Initialize the Edge-Conditioned GAT Layer.

        Args:
            node_dim: Dimension of input node feature vectors.
            edge_dim: Dimension of edge feature vectors (relative geometry/kinematics).
            hidden_dim: Total hidden dimension across all attention heads (split evenly:
                head_dim = hidden_dim // num_heads per head).
            num_heads: Number of parallel attention heads for multi-head attention.
            dropout: Dropout probability applied to attention weights and message transformations.
            leaky_relu_slope: Negative slope parameter for LeakyReLU activation in attention calculation.
            top_k: If set, each receiver node only aggregates from its top-K highest-scoring
                in-range neighbors (per forward pass). None disables sparsification (dense attention).
            topk_mode: Selection strategy — 'attention' uses learned attention scores, 'random'
                picks K in-range neighbors uniformly at random (baseline for ablation).
        """
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads."
        
        self.leaky_relu_slope = leaky_relu_slope
        self.dropout = nn.Dropout(dropout)
        self.top_k = top_k
        self.topk_mode = topk_mode
        # Fraction of in-range neighbors dropped by top-K sparsification (updated each forward pass).
        # Read this attribute externally to wire into RLlib callbacks for monitoring.
        self.last_drop_frac: float = 0.0

        # Linear projections for Query, Key, and Edge embeddings in attention formulation
        self.proj_q = nn.Linear(node_dim, hidden_dim, bias=False)
        self.proj_k = nn.Linear(node_dim, hidden_dim, bias=False)
        self.proj_e = nn.Linear(edge_dim, hidden_dim, bias=False)

        # Attention vector parameterizing the scoring function across all heads
        # Shape: (1, num_heads, 3 * head_dim)
        self.attn_vector = nn.Parameter(torch.Tensor(1, num_heads, 3 * self.head_dim))
        nn.init.xavier_uniform_(self.attn_vector)

        # Message generation network combining receiver node, sender node, and edge geometry
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Output projection and residual LayerNorm
        self.proj_out = nn.Linear(hidden_dim, node_dim)
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        node_features: torch.Tensor,
        adj_matrix: torch.Tensor,
        edge_features: torch.Tensor,
        random_comm_mask: Optional[torch.Tensor] = None,
        shared_topk_indices: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for dynamic topological message passing.

        Args:
            node_features: Tensor of shape (batch_size, num_nodes, node_dim).
            adj_matrix: Binary adjacency tensor of shape (batch_size, num_nodes, num_nodes).
                        adj_matrix[b, i, j] == 1 if node j can transmit to node i.
            edge_features: Tensor of shape (batch_size, num_nodes, num_nodes, edge_dim).
                           edge_features[b, i, j] contains the geometry of node j as seen
                           from node i (displacement from i to j).
            random_comm_mask: Optional tensor of shape (batch_size, num_nodes, num_nodes)
                              containing pre-drawn uniform random scores for topk_mode='random'.
                              Generated once per env step in env_core.py so PPO's multi-epoch
                              SGD replays the same neighbor selection. Required when
                              topk_mode='random'; ignored otherwise.
            shared_topk_indices: Optional tensor of shape (batch_size, num_nodes, effective_k)
                              from a previous layer's neighbor selection in this same forward
                              pass. When provided, this layer skips computing its own top-K
                              selection (attention or random) and reuses these indices instead,
                              so the *set* of neighbors is fixed for the whole multi-layer stack
                              while each layer still computes its own attention weights and
                              messages over that fixed set. Keeps 'attention' mode symmetric
                              with 'random' mode, whose mask is already identical across layers
                              since it's a deterministic function of the same random_comm_mask
                              and adjacency at every layer.

        Returns:
            Tuple of (updated node features of shape (batch_size, num_nodes, node_dim),
            the top-K neighbor indices used this layer, or None if top_k sparsification
            is disabled). The caller (DynamicTopologicalGNN) caches the first layer's
            indices and passes them back in as shared_topk_indices for later layers.

        Note on gradient flow and attention entropy (top_k mode):
            When top_k is enabled, ``torch.topk``'s index selection is a non-differentiable
            discrete operation. Gradients flow through the selected attention *values*
            (via softmax), but the binary decision of which K neighbors to retain has
            zero gradient. Early in PPO training — when attention weights are near-uniform
            across neighbors — this risks attention entropy collapse: a small initial score
            advantage causes a neighbor to be consistently selected, reinforcing its score
            while unselected neighbors receive no gradient signal. Since this policy is
            shared across all agents (``GNNMARLModel`` in marl_agent.py), entropy collapse
            in one agent's neighborhood can propagate through the shared parameters.
            Consider using entropy regularization or annealing top_k from ``None`` to the
            target K over the first ~100K training steps to mitigate this.
        """
        batch_size, num_nodes, _ = node_features.shape

        # 1. Compute multi-head Query, Key, and Edge representations
        # Reshape to (batch_size, num_nodes, num_heads, head_dim)
        q = self.proj_q(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.proj_k(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        e = self.proj_e(edge_features).view(batch_size, num_nodes, num_nodes, self.num_heads, self.head_dim)

        # Expand q and k across pairwise node interactions (batch_size, num_nodes, num_nodes, num_heads, head_dim)
        # q_expand: receiver node i; k_expand: sender node j
        q_expand = q.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
        k_expand = k.unsqueeze(1).expand(-1, num_nodes, -1, -1, -1)

        # Concatenate receiver (q), sender (k), and edge geometry (e)
        # Shape: (batch_size, num_nodes, num_nodes, num_heads, 3 * head_dim)
        attn_input = torch.cat([q_expand, k_expand, e], dim=-1)

        # Calculate raw attention scores via inner product with learnable attention vector
        # Shape: (batch_size, num_nodes, num_nodes, num_heads)
        raw_scores = (attn_input * self.attn_vector).sum(dim=-1)
        raw_scores = F.leaky_relu(raw_scores, negative_slope=self.leaky_relu_slope)

        # 2. Mask non-existent edges (where adj_matrix == 0) and self-loops if disconnected
        # Add self-loops to adjacency matrix to ensure every node retains its own features
        eye = torch.eye(num_nodes, device=adj_matrix.device, dtype=adj_matrix.dtype).unsqueeze(0)
        adj_with_loops = torch.clamp(adj_matrix + eye, 0.0, 1.0)
        
        # Expand adj mask across heads: (batch_size, num_nodes, num_nodes, 1)
        adj_mask = adj_with_loops.unsqueeze(-1)
        
        # Apply mask: set non-neighbor entries to -1e9 so Softmax drives attention to 0
        masked_scores = raw_scores.masked_fill(adj_mask == 0, -1e9)

        # 2b. Attention-driven top-K communication sparsification
        #     Applied on top of the proximity-based adjacency mask (combined, not replaced).
        if self.top_k is not None and num_nodes > 1:
            effective_k = min(self.top_k, num_nodes - 1)  # Exclude self from K budget

            if effective_k > 0:
                # Average attention scores across heads for consistent per-node neighbor selection
                avg_scores_for_topk = masked_scores.mean(dim=-1)  # (batch_size, N, N)

                # Exclude self-loop from top-K competition; self is always retained separately
                diag_mask = torch.eye(
                    num_nodes, device=adj_matrix.device, dtype=torch.bool
                ).unsqueeze(0)

                if shared_topk_indices is not None:
                    # Reuse a previous layer's neighbor selection instead of recomputing:
                    # keeps the chosen neighbor *set* fixed across the whole stack while this
                    # layer still uses its own attention weights/messages over that set.
                    topk_indices = shared_topk_indices
                elif self.topk_mode == 'attention':
                    selection_scores = avg_scores_for_topk.clone()
                    selection_scores.masked_fill_(diag_mask, -float('inf'))
                    # GRADIENT STOP: torch.topk's index selection is a non-differentiable
                    # discrete operation (argmax-like). Gradients flow through the selected
                    # attention *values* post-softmax, but the discrete choice of *which* K
                    # neighbors to keep has zero gradient — effectively a hard gate.
                    _, topk_indices = torch.topk(selection_scores, k=effective_k, dim=2)
                elif self.topk_mode == 'random':
                    # Random baseline: use pre-computed random scores from env_core.py.
                    # These are drawn once per env step and replayed across all PPO SGD
                    # epochs, matching the attention branch's per-state consistency.
                    if random_comm_mask is None:
                        raise ValueError(
                            "topk_mode='random' requires random_comm_mask to be provided. "
                            "This tensor should be generated once per env step (in "
                            "env_core.py) and passed through the observation dict."
                        )
                    random_scores = random_comm_mask.clone()
                    random_scores.masked_fill_(adj_matrix == 0, -float('inf'))
                    random_scores.masked_fill_(diag_mask, -float('inf'))
                    _, topk_indices = torch.topk(random_scores, k=effective_k, dim=2)
                else:
                    raise ValueError(
                        f"Unknown topk_mode: {self.topk_mode!r}. Must be 'attention' or 'random'."
                    )

                # Build top-K mask: 1 at selected indices, 0 elsewhere
                topk_mask = torch.zeros(
                    batch_size, num_nodes, num_nodes, device=adj_matrix.device
                )
                topk_mask.scatter_(2, topk_indices, 1.0)
                # Always retain self-loops for attention stability (residual also preserves,
                # but keeping self in softmax avoids degenerate zero-neighbor distributions)
                topk_mask.masked_fill_(diag_mask, 1.0)

                # Track fraction of in-range neighbors dropped by sparsification
                in_range_counts = adj_matrix.sum(dim=2)  # (B, N) — excludes self
                kept_neighbor_mask = topk_mask * adj_matrix  # Only count actual in-range retained
                kept_counts = kept_neighbor_mask.sum(dim=2)  # (B, N)
                total_in_range = in_range_counts.sum()
                if total_in_range > 0:
                    self.last_drop_frac = float(1.0 - kept_counts.sum() / total_in_range)
                else:
                    self.last_drop_frac = 0.0

                # Apply top-K mask on top of existing adjacency mask (combined, not replaced)
                topk_mask_expanded = topk_mask.unsqueeze(-1)  # (B, N, N, 1)
                masked_scores = masked_scores.masked_fill(topk_mask_expanded == 0, -1e9)
            else:
                self.last_drop_frac = 0.0
                topk_indices = None
        else:
            self.last_drop_frac = 0.0
            topk_indices = None

        # Compute normalized attention coefficients across neighbors (dim=2 is sender node j)
        alpha = F.softmax(masked_scores, dim=2)
        alpha = self.dropout(alpha)  # Shape: (batch_size, num_nodes, num_nodes, num_heads)

        # 3. Generate pairwise messages conditioned on receiver, sender, and edge geometry
        # Expand node features for pairwise concatenation
        h_i = node_features.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        h_j = node_features.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        
        # Concatenate: [h_i || h_j || e_ij] -> shape (batch_size, num_nodes, num_nodes, 2*node_dim + edge_dim)
        msg_input = torch.cat([h_i, h_j, edge_features], dim=-1)
        messages = self.msg_mlp(msg_input)  # Shape: (batch_size, num_nodes, num_nodes, hidden_dim)
        
        # Reshape messages to multi-head format: (batch_size, num_nodes, num_nodes, num_heads, head_dim)
        messages_mh = messages.view(batch_size, num_nodes, num_nodes, self.num_heads, self.head_dim)

        # 4. Aggregate neighborhood messages using attention weights
        # alpha shape expanded: (batch_size, num_nodes, num_nodes, num_heads, 1)
        weighted_messages = messages_mh * alpha.unsqueeze(-1)
        
        # Sum across sender nodes j (dim=2): (batch_size, num_nodes, num_heads, head_dim)
        aggregated_mh = weighted_messages.sum(dim=2)
        
        # Flatten head dimension: (batch_size, num_nodes, hidden_dim)
        aggregated = aggregated_mh.view(batch_size, num_nodes, self.hidden_dim)

        # 5. Output projection + Residual connection + LayerNorm
        out_projected = self.dropout(self.proj_out(aggregated))
        updated_features = self.layer_norm(node_features + out_projected)

        return updated_features, topk_indices


class DynamicTopologicalGNN(nn.Module):
    """
    Multi-Layer Dynamic Topological GNN for Multi-Robot Communication.

    Processes raw node state observations and relative edge geometries through L layers of
    edge-conditioned message passing, returning a compact latent communication vector per robot.
    """

    def __init__(
        self,
        raw_obs_dim: int,
        edge_dim: int,
        comm_latent_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        top_k: Optional[int] = None,
        topk_mode: str = 'attention'
    ) -> None:
        """
        Initialize the multi-layer topological communication network.

        Args:
            raw_obs_dim: Dimension of raw local sensor observation vector per robot.
            edge_dim: Dimension of edge features (relative position, velocity, distance).
            comm_latent_dim: Dimension of the output latent communication vector z_comm per robot.
            hidden_dim: Internal feature dimension across GNN layers.
            num_layers: Number of message-passing hops (layers).
            num_heads: Number of attention heads per GAT layer.
            dropout: Dropout probability.
            top_k: If set, passed through to each EdgeConditionedGATLayer to enable top-K
                communication sparsification. None disables (dense attention).
            topk_mode: Selection strategy passed through to GAT layers ('attention' or 'random').
        """
        super().__init__()
        self.raw_obs_dim = raw_obs_dim
        self.edge_dim = edge_dim
        self.comm_latent_dim = comm_latent_dim
        self.num_layers = num_layers
        self.top_k = top_k
        self.topk_mode = topk_mode

        # Input feature encoder projecting raw sensor observations into GNN latent space
        self.node_encoder = nn.Sequential(
            nn.Linear(raw_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Stack L layers of Edge-Conditioned Graph Attention
        self.gat_layers = nn.ModuleList([
            EdgeConditionedGATLayer(
                node_dim=hidden_dim,
                edge_dim=edge_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                top_k=top_k,
                topk_mode=topk_mode
            )
            for _ in range(num_layers)
        ])

        # Final projection head compressing multi-hop embeddings into compact communication vectors
        self.comm_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, comm_latent_dim),
            nn.LayerNorm(comm_latent_dim)
        )

    def forward(
        self,
        raw_obs: torch.Tensor,
        adj_matrix: torch.Tensor,
        edge_features: torch.Tensor,
        random_comm_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Execute multi-hop topological message passing.

        Args:
            raw_obs: Local observations tensor of shape (batch_size, num_nodes, raw_obs_dim).
            adj_matrix: Adjacency matrix tensor of shape (batch_size, num_nodes, num_nodes).
            edge_features: Edge geometry tensor of shape (batch_size, num_nodes, num_nodes, edge_dim).
            random_comm_mask: Optional pre-drawn random scores for topk_mode='random'.
                              Shape (batch_size, num_nodes, num_nodes). See
                              EdgeConditionedGATLayer.forward() for details.

        Returns:
            Latent communication vectors of shape (batch_size, num_nodes, comm_latent_dim).
        """
        # Encode initial node representations
        h = self.node_encoder(raw_obs)

        # Propagate messages across dynamic topology across num_layers hops.
        # Neighbor selection (topk_indices) is computed once, at the first layer, and
        # reused unchanged at every subsequent layer — each layer still computes its own
        # attention weights and messages over that fixed neighbor set. This keeps the
        # comparison between topk_mode='attention' and topk_mode='random' fair: without
        # this, 'random' mode's neighbor set is already identical across layers (same
        # random_comm_mask + adjacency every layer), while 'attention' mode would otherwise
        # re-select independently at each layer since each has its own learned q/k.
        shared_topk_indices = None
        for layer in self.gat_layers:
            h, layer_topk_indices = layer(
                h, adj_matrix, edge_features,
                random_comm_mask=random_comm_mask,
                shared_topk_indices=shared_topk_indices
            )
            if shared_topk_indices is None:
                shared_topk_indices = layer_topk_indices

        # Project to compact communication latent space
        comm_vectors = self.comm_head(h)
        return comm_vectors


if __name__ == "__main__":
    # Smoke test validating tensor dimensions and backpropagation flow
    print("Running verification smoke test for DynamicTopologicalGNN...")
    batch_size, num_robots, obs_dim, edge_dim, latent_dim = 4, 6, 24, 8, 32

    dummy_obs = torch.randn(batch_size, num_robots, obs_dim)
    dummy_adj = torch.randint(0, 2, (batch_size, num_robots, num_robots)).float()
    dummy_edges = torch.randn(batch_size, num_robots, num_robots, edge_dim)

    # --- Test 1: Dense attention (top_k=None, original behavior) ---
    print("\n[Test 1] Dense attention (top_k=None)...")
    gnn_dense = DynamicTopologicalGNN(
        raw_obs_dim=obs_dim,
        edge_dim=edge_dim,
        comm_latent_dim=latent_dim,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        top_k=None
    )

    out_dense = gnn_dense(dummy_obs, dummy_adj, dummy_edges)
    assert out_dense.shape == (batch_size, num_robots, latent_dim), \
        f"Expected shape {(batch_size, num_robots, latent_dim)}, got {out_dense.shape}"

    loss_dense = out_dense.pow(2).sum()
    loss_dense.backward()
    assert gnn_dense.node_encoder[0].weight.grad is not None, \
        "Gradient propagation failed through GNN layers (dense mode)."
    for layer in gnn_dense.gat_layers:
        assert layer.last_drop_frac == 0.0, \
            f"Expected 0.0 drop frac for dense mode, got {layer.last_drop_frac}"
    print("  Output shape:", out_dense.shape, "OK")
    print("  Gradient flow: OK")
    print("  Drop fraction: 0.0 OK")

    # --- Test 2: Top-K sparsified attention (top_k=2) ---
    print("\n[Test 2] Top-K sparsified attention (top_k=2)...")
    gnn_sparse = DynamicTopologicalGNN(
        raw_obs_dim=obs_dim,
        edge_dim=edge_dim,
        comm_latent_dim=latent_dim,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        top_k=2,
        topk_mode='attention'
    )

    out_sparse = gnn_sparse(dummy_obs, dummy_adj, dummy_edges)
    assert out_sparse.shape == (batch_size, num_robots, latent_dim), \
        f"Expected shape {(batch_size, num_robots, latent_dim)}, got {out_sparse.shape}"

    loss_sparse = out_sparse.pow(2).sum()
    loss_sparse.backward()
    assert gnn_sparse.node_encoder[0].weight.grad is not None, \
        "Gradient propagation failed through GNN layers (top-K mode)."
    for layer in gnn_sparse.gat_layers:
        print(f"  Layer drop fraction: {layer.last_drop_frac:.4f}")
    print("  Output shape:", out_sparse.shape, "OK")
    print("  Gradient flow: OK")

    # --- Test 3: Random baseline (top_k=2, topk_mode='random') ---
    print("\n[Test 3] Random baseline (top_k=2, topk_mode='random')...")
    gnn_random = DynamicTopologicalGNN(
        raw_obs_dim=obs_dim,
        edge_dim=edge_dim,
        comm_latent_dim=latent_dim,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        top_k=2,
        topk_mode='random'
    )

    dummy_random_mask = torch.rand(batch_size, num_robots, num_robots)
    out_random = gnn_random(dummy_obs, dummy_adj, dummy_edges, random_comm_mask=dummy_random_mask)
    assert out_random.shape == (batch_size, num_robots, latent_dim), \
        f"Expected shape {(batch_size, num_robots, latent_dim)}, got {out_random.shape}"

    loss_random = out_random.pow(2).sum()
    loss_random.backward()
    assert gnn_random.node_encoder[0].weight.grad is not None, \
        "Gradient propagation failed through GNN layers (random mode)."
    for layer in gnn_random.gat_layers:
        print(f"  Layer drop fraction: {layer.last_drop_frac:.4f}")
    print("  Output shape:", out_random.shape, "OK")
    print("  Gradient flow: OK")

    print("\nAll verification tests passed! DynamicTopologicalGNN functions correctly.")
