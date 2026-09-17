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
        topk_mode: str = 'attention',
        gumbel_temperature: float = 1.0
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
            topk_mode: Selection strategy — 'attention' uses learned attention scores with hard
                top-k (non-differentiable index selection), 'gumbel' uses Gumbel-Softmax
                differentiable relaxation so gradient flows through neighbor selection,
                'random' picks K in-range neighbors uniformly at random (ablation baseline).
            gumbel_temperature: Temperature for Gumbel-Softmax relaxation (topk_mode='gumbel').
                Lower temperature → harder/more discrete selection. Higher → softer/more
                exploratory. Typically annealed from ~1.0 down to ~0.1 during training.
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
        self.gumbel_temperature = gumbel_temperature
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

        # FP16-safe mask value: -1e9 overflows to -inf under AMP (FP16 max ≈ 65504).
        # -1e4 is sufficient to drive softmax attention to ~0 while staying in-range.
        self._MASK_VALUE = -1e4

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
            random_comm_mask: Uniform draws stored in the observation, required for
                random and Gumbel routing so PPO can replay its stochastic input.
            shared_topk_indices: First-layer integer indices, or the differentiable
                (B, N, N) gate for Gumbel mode. Reused across the layer stack.

        Returns:
            Updated features and the shared routing tensor, or None for dense mode.

        Hard attention indices have no derivative, but selected attention values
        and shared score parameters still learn. Gumbel mode supplies a biased
        straight-through surrogate gradient for routing. Neither method guarantees
        useful rankings; entropy alone cannot diagnose routing quality.
        """
        batch_size, num_nodes, _ = node_features.shape

        # 1. Compute multi-head Query, Key, and Edge representations
        # Reshape to (batch_size, num_nodes, num_heads, head_dim)
        q = self.proj_q(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        k = self.proj_k(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim)
        e = self.proj_e(edge_features).view(batch_size, num_nodes, num_nodes, self.num_heads, self.head_dim)

        # Efficient additive attention scoring without materializing 5D concatenated tensors:
        # attn_vector has shape (1, num_heads, 3 * head_dim) = [a_q || a_k || a_e]
        a_q = self.attn_vector[..., :self.head_dim]
        a_k = self.attn_vector[..., self.head_dim:2 * self.head_dim]
        a_e = self.attn_vector[..., 2 * self.head_dim:]

        score_q = (q * a_q).sum(dim=-1).unsqueeze(2)  # (batch_size, num_nodes, 1, num_heads)
        score_k = (k * a_k).sum(dim=-1).unsqueeze(1)  # (batch_size, 1, num_nodes, num_heads)
        score_e = (e * a_e).sum(dim=-1)               # (batch_size, num_nodes, num_nodes, num_heads)
        raw_scores = F.leaky_relu(score_q + score_k + score_e, negative_slope=self.leaky_relu_slope)

        # 2. Mask non-existent edges (where adj_matrix == 0) and self-loops if disconnected
        # Add self-loops to adjacency matrix to ensure every node retains its own features
        eye = torch.eye(num_nodes, device=adj_matrix.device, dtype=adj_matrix.dtype).unsqueeze(0)
        adj_with_loops = torch.clamp(adj_matrix + eye, 0.0, 1.0)
        
        # Expand adj mask across heads: (batch_size, num_nodes, num_nodes, 1)
        adj_mask = adj_with_loops.unsqueeze(-1)
        
        # Apply mask: set non-neighbor entries to _MASK_VALUE so Softmax drives attention to ~0
        masked_scores = raw_scores.masked_fill(adj_mask == 0, -float("inf"))

        # The observation carries routing randomness so PPO replays the same
        # stochastic policy input during sampling and all optimization epochs.
        valid = adj_matrix.bool() & ~eye.bool()
        scores = raw_scores.mean(dim=-1)
        gate = adj_with_loops
        topk_indices = None
        if self.top_k is not None:
            if self.top_k < 0:
                raise ValueError("top_k must be nonnegative or None")
            effective_k = min(self.top_k, num_nodes - 1)
            selection = scores
            if self.topk_mode in ("random", "gumbel"):
                if random_comm_mask is None:
                    raise ValueError("random and gumbel routing require observation random_comm_mask")
                uniform = random_comm_mask.clamp(1e-6, 1 - 1e-6)
                selection = (uniform if self.topk_mode == "random"
                             else scores - torch.log(-torch.log(uniform)))
            elif self.topk_mode == "distance":
                selection = -edge_features[..., 6]
            elif self.topk_mode != "attention":
                raise ValueError(f"Unknown routing mode: {self.topk_mode}")
            if shared_topk_indices is not None and self.topk_mode == "gumbel":
                gate = shared_topk_indices
                topk_indices = gate
            else:
                selection = selection.masked_fill(~valid, -float("inf"))
                if shared_topk_indices is not None:
                    topk_indices = shared_topk_indices
                else:
                    topk_indices = selection.topk(effective_k, dim=2).indices
                hard = torch.zeros_like(scores).scatter(2, topk_indices, 1.0) * valid
                if self.topk_mode == "gumbel" and effective_k > 0:
                    if self.gumbel_temperature <= 0:
                        raise ValueError("gumbel_temperature must be positive")
                    # Sequential relaxed samples supply a surrogate gradient;
                    # forward execution retains exactly min(K, degree) edges.
                    remaining = valid.clone()
                    soft = torch.zeros_like(scores)
                    for rank in range(effective_k):
                        candidate = remaining.any(dim=2, keepdim=True)
                        logits = selection.masked_fill(~remaining, -float("inf"))
                        logits = torch.where(candidate, logits, torch.zeros_like(logits))
                        sample = F.softmax(logits / self.gumbel_temperature, dim=2)
                        soft = soft + sample * candidate
                        chosen = topk_indices[:, :, rank:rank + 1]
                        remaining = remaining.scatter(2, chosen, False)
                    gate = hard + soft - soft.detach() + eye
                    topk_indices = gate
                else:
                    gate = hard + eye
        selected = (gate.detach() > 0.5) & valid
        self.last_selected_mask = selected.detach()
        self.last_attention_scores = scores.detach()
        self.last_available_edges = int(valid.sum().item())
        self.last_selected_edges = int(selected.sum().item())
        self.last_drop_frac = (1 - self.last_selected_edges / self.last_available_edges
                               if self.last_available_edges else 0.0)
        if self.topk_mode == "gumbel" and self.top_k is not None:
            # Do not mask away candidate weights before the straight-through
            # gate: that would suppress gradients for unselected messages.
            weights = torch.exp(masked_scores - masked_scores.amax(dim=2, keepdim=True))
            weights = weights * gate.unsqueeze(-1)
            alpha = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-12)
        else:
            masked_scores = masked_scores.masked_fill(gate.unsqueeze(-1) < 0.5, -float("inf"))
            alpha = F.softmax(masked_scores, dim=2)
        self.last_attention_weights = alpha.detach()
        # Evaluation-only message intervention: hold routing and normalization fixed.
        ablation = getattr(self, "message_ablation", None)
        if ablation is not None:
            receiver, sender = ablation
            keep = torch.ones_like(alpha)
            keep[:, receiver, sender, :] = 0
            alpha = alpha * keep
        alpha = self.dropout(alpha)

        # 3. Generate pairwise messages conditioned on receiver, sender, and edge geometry
        # Expand node features for pairwise concatenation
        h_i = node_features.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        h_j = node_features.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        
        # Concatenate: [h_i || h_j || e_ij] -> shape (batch_size, num_nodes, num_nodes, 2*node_dim + edge_dim)
        msg_input = torch.cat([h_i, h_j, edge_features], dim=-1)
        messages = self.msg_mlp(msg_input)  # Shape: (batch_size, num_nodes, num_nodes, hidden_dim)
        
        # Reshape messages to multi-head format: (batch_size, num_nodes, num_nodes, self.num_heads, self.head_dim)
        messages_mh = messages.view(batch_size, num_nodes, num_nodes, self.num_heads, self.head_dim)

        # 4. Aggregate neighborhood messages using attention weights via efficient einsum
        # Sum across sender nodes j (dim=2): (batch_size, num_nodes, num_heads, head_dim)
        aggregated_mh = torch.einsum('bijh,bijhd->bihd', alpha, messages_mh)
        
        # Flatten head dimension: (batch_size, num_nodes, hidden_dim)
        aggregated = aggregated_mh.reshape(batch_size, num_nodes, self.hidden_dim)

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
        topk_mode: str = 'attention',
        gumbel_temperature: float = 1.0
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
            topk_mode: Selection strategy passed through to GAT layers ('attention', 'gumbel', or 'random').
            gumbel_temperature: Temperature for Gumbel-Softmax (topk_mode='gumbel').
        """
        super().__init__()
        self.raw_obs_dim = raw_obs_dim
        self.edge_dim = edge_dim
        self.comm_latent_dim = comm_latent_dim
        self.num_layers = num_layers
        self.top_k = top_k
        self.topk_mode = topk_mode
        self.gumbel_temperature = gumbel_temperature

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
                topk_mode=topk_mode,
                gumbel_temperature=gumbel_temperature
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

    # --- Test 4: Gumbel-Softmax with isolated nodes (NaN guard) ---
    print("\n[Test 4] Gumbel-Softmax with isolated nodes (NaN guard)...")
    gnn_gumbel = DynamicTopologicalGNN(
        raw_obs_dim=obs_dim,
        edge_dim=edge_dim,
        comm_latent_dim=latent_dim,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        top_k=2,
        topk_mode='gumbel',
        gumbel_temperature=1.0
    )

    # Create adjacency where node 0 in batch 0 is completely isolated
    disconnected_adj = torch.randint(0, 2, (batch_size, num_robots, num_robots)).float()
    disconnected_adj[0, 0, :] = 0.0  # Node 0 has no outgoing edges
    disconnected_adj[0, :, 0] = 0.0  # Node 0 has no incoming edges

    out_gumbel = gnn_gumbel(dummy_obs, disconnected_adj, dummy_edges, random_comm_mask=dummy_random_mask)
    assert not torch.isnan(out_gumbel).any(), \
        "NaN detected in Gumbel-Softmax output with isolated nodes!"
    assert out_gumbel.shape == (batch_size, num_robots, latent_dim), \
        f"Expected shape {(batch_size, num_robots, latent_dim)}, got {out_gumbel.shape}"

    loss_gumbel = out_gumbel.pow(2).sum()
    loss_gumbel.backward()
    assert not any(
        torch.isnan(p.grad).any() for p in gnn_gumbel.parameters() if p.grad is not None
    ), "NaN detected in gradients with isolated nodes!"
    assert gnn_gumbel.node_encoder[0].weight.grad is not None, \
        "Gradient propagation failed through GNN layers (Gumbel mode)."
    for layer in gnn_gumbel.gat_layers:
        print(f"  Layer drop fraction: {layer.last_drop_frac:.4f}")
    print("  Output shape:", out_gumbel.shape, "OK")
    print("  No NaN in output: OK")
    print("  No NaN in gradients: OK")
    print("  Gradient flow: OK")

    print("\nAll verification tests passed! DynamicTopologicalGNN functions correctly.")
