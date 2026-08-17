"""
Multi-Agent Reinforcement Learning (MARL) Local Policy Network with GNN Integration.

This module implements the local policy architecture (`GNNMARLModel`) using Ray RLlib's `TorchModelV2`
interface. The policy fuses local exteroceptive/kinematic sensor data with multi-hop topological
communication vectors produced by the `DynamicTopologicalGNN` (`gnn_comm_layer.py`).

Strategic & Mathematical Reasoning:
1. Decentralized Execution via Local + Communication Fusion:
   While the graph layer processes neighborhood feature vectors, agent i makes its local decision based on:
       z_joint = [ h_local(o_i) || z_comm(i) ]
   where o_i is local sensor readings and z_comm(i) is the aggregated communication embedding for node i.
2. Actor-Critic Parameterization (PPO Compatible):
   - Actor Head: Outputs action distribution logits (continuous Gaussian means/log_stds or discrete logits).
   - Critic Head: Estimates state-value V(o_i, z_comm). During CTDE (Centralized Training with Decentralized Execution),
     this value head can be expanded to evaluate the entire graph latent state while the actor remains local.
"""

from typing import Dict, List, Any, Tuple, Optional
import torch
import torch.nn as nn
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.typing import TensorType, ModelConfigDict

from gnn_comm_layer import DynamicTopologicalGNN


class GNNMARLModel(TorchModelV2, nn.Module):
    """
    RLlib TorchModelV2 integrating local sensor processing with dynamic topological GNN communication.

    Observation Dictionary Structure (`input_dict["obs"]`):
        - `local_obs`: Tensor of shape (batch_size, raw_obs_dim) - Agent i's direct sensor readings.
        - `node_features`: Tensor of shape (batch_size, num_nodes, raw_obs_dim) - All node states in graph.
        - `adj_matrix`: Binary/weighted adjacency matrix of shape (batch_size, num_nodes, num_nodes).
        - `edge_features`: Spatial edge geometry of shape (batch_size, num_nodes, num_nodes, edge_dim).
        - `node_index`: Integer index tensor of shape (batch_size, 1) or (batch_size,) identifying which
                        robot in the N-node graph corresponds to this policy evaluation instance.
    """

    def __init__(
        self,
        obs_space: Any,
        action_space: Any,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
        **kwargs: Any
    ) -> None:
        """
        Initialize the GNN-integrated MARL Model.

        Args:
            obs_space: Gym/RLlib observation space (`spaces.Dict`).
            action_space: Gym/RLlib action space (continuous Box or discrete Discrete).
            num_outputs: Number of action distribution parameters (e.g., 2 * action_dim for DiagGaussian).
            model_config: Dictionary containing custom hyperparameters under `custom_model_config`.
            name: Model identifier string.
            kwargs: Additional keyword arguments passed by Ray RLlib catalog.
        """
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Extract hyperparameters from custom_model_config or kwargs with robust fallbacks
        custom_cfg = model_config.get("custom_model_config", {})
        self.raw_obs_dim = kwargs.get("raw_obs_dim", custom_cfg.get("raw_obs_dim", 24))
        self.edge_dim = kwargs.get("edge_dim", custom_cfg.get("edge_dim", 8))
        self.comm_latent_dim = kwargs.get("comm_latent_dim", custom_cfg.get("comm_latent_dim", 64))
        self.local_hidden_dim = kwargs.get("local_hidden_dim", custom_cfg.get("local_hidden_dim", 128))
        self.gnn_num_layers = kwargs.get("gnn_num_layers", custom_cfg.get("gnn_num_layers", 2))
        self.gnn_num_heads = kwargs.get("gnn_num_heads", custom_cfg.get("gnn_num_heads", 4))
        self.top_k = kwargs.get("top_k", custom_cfg.get("top_k", None))
        self.topk_mode = kwargs.get("topk_mode", custom_cfg.get("topk_mode", "attention"))
        self.gumbel_temperature = kwargs.get("gumbel_temperature", custom_cfg.get("gumbel_temperature", 1.0))
        self.no_comm = kwargs.get("no_comm", custom_cfg.get("no_comm", False))

        # If annealing is configured, start dense — the callback in train.py drives the ramp-down
        top_k_anneal_steps = kwargs.get("top_k_anneal_steps", custom_cfg.get("top_k_anneal_steps", None))
        initial_top_k = None if top_k_anneal_steps is not None else self.top_k

        # 1. Instantiate the Dynamic Topological GNN communication engine
        self.gnn_layer = DynamicTopologicalGNN(
            raw_obs_dim=self.raw_obs_dim,
            edge_dim=self.edge_dim,
            comm_latent_dim=self.comm_latent_dim,
            hidden_dim=self.local_hidden_dim,
            num_layers=self.gnn_num_layers,
            num_heads=self.gnn_num_heads,
            top_k=initial_top_k,
            topk_mode=self.topk_mode,
            gumbel_temperature=self.gumbel_temperature
        )

        # 2. Local sensor feature encoder (processing direct exteroceptive/proprioceptive inputs)
        self.local_encoder = nn.Sequential(
            nn.Linear(self.raw_obs_dim, self.local_hidden_dim),
            nn.LayerNorm(self.local_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.local_hidden_dim, self.local_hidden_dim),
            nn.ReLU()
        )

        # Joint feature dimension = encoded local observation + GNN communication vector
        joint_dim = self.local_hidden_dim + self.comm_latent_dim

        # 3. Actor Head (Policy action distribution logits generator)
        self.actor_head = nn.Sequential(
            nn.Linear(joint_dim, self.local_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.local_hidden_dim, num_outputs)
        )

        # 4. Critic Head (CTDE Value function: sees global state during centralized training)
        # Input: [h_local || z_comm_i || global_state_pooled]
        # The actor uses only joint_dim = local + comm; the critic additionally sees
        # a mean-pooled global state across all node embeddings for CTDE.
        critic_dim = joint_dim + self.comm_latent_dim
        self.critic_head = nn.Sequential(
            nn.Linear(critic_dim, self.local_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.local_hidden_dim, 1)
        )

        # Internal value cache required by RLlib's value_function() API
        self._cur_value: Optional[torch.Tensor] = None
        self._last_drop_frac: float = 0.0

    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType
    ) -> Tuple[TensorType, List[TensorType]]:
        """
        Forward pass computing action distribution logits and caching value estimates.

        Args:
            input_dict: Dictionary containing observation tensors under `input_dict["obs"]`.
            state: Hidden state tensors for recurrent models (unused in feedforward GNN).
            seq_lens: Sequence lengths for batch evaluation.

        Returns:
            Tuple of (action_logits, state).
        """
        obs_dict = input_dict["obs"]

        # Extract structured graph components and local state
        # Convert all to float/long tensors ensuring device consistency
        local_obs = obs_dict["local_obs"].float()
        node_features = obs_dict["node_features"].float()
        adj_matrix = obs_dict["adj_matrix"].float()
        edge_features = obs_dict["edge_features"].float()
        random_comm_mask = obs_dict["random_comm_mask"].float()
        
        # node_index indicates which node in the graph corresponds to the evaluating agent
        node_index = obs_dict["node_index"].long()
        if node_index.dim() > 1:
            node_index = node_index.squeeze(-1)

        batch_size = local_obs.shape[0]

        if self.no_comm:
            # IPPO baseline: no GNN communication. Zero out communication vectors.
            # GNN layer exists but is not called (keeps weights for checkpoint compat).
            num_nodes = node_features.shape[1]
            gnn_latents = torch.zeros(
                batch_size, num_nodes, self.comm_latent_dim,
                device=local_obs.device, dtype=local_obs.dtype
            )
            self._last_drop_frac = 0.0
        else:
            # 1. Execute topological message passing across the entire dynamic neighborhood graph
            # Shape of gnn_latents: (batch_size, num_nodes, comm_latent_dim)
            gnn_latents = self.gnn_layer(
                node_features, adj_matrix, edge_features,
                random_comm_mask=random_comm_mask
            )

            # Average last_drop_frac across all GAT layers
            if hasattr(self.gnn_layer, "gat_layers") and len(self.gnn_layer.gat_layers) > 0:
                self._last_drop_frac = float(
                    sum(getattr(layer, "last_drop_frac", 0.0) for layer in self.gnn_layer.gat_layers)
                    / len(self.gnn_layer.gat_layers)
                )
            else:
                self._last_drop_frac = 0.0

        # 2. Extract the specific communication latent vector z_comm corresponding to agent i
        # Using batch indexing: for each batch element b, select row node_index[b]
        batch_indices = torch.arange(batch_size, device=gnn_latents.device)
        z_comm_i = gnn_latents[batch_indices, node_index, :]  # Shape: (batch_size, comm_latent_dim)

        # 3. Encode local sensor readings
        h_local = self.local_encoder(local_obs)  # Shape: (batch_size, local_hidden_dim)

        # 4. Concatenate local representation with multi-hop communication embedding
        joint_features = torch.cat([h_local, z_comm_i], dim=-1)  # Shape: (batch_size, joint_dim)

        # 5. Compute action distribution logits (actor: decentralized, uses local + comm only)
        action_logits = self.actor_head(joint_features)

        # 6. CTDE Critic: additionally sees global state via mean-pooling across all nodes.
        # This gives the value function access to the full team's latent state during
        # centralized training, while the actor remains strictly decentralized.
        global_state = gnn_latents.mean(dim=1)  # (batch_size, comm_latent_dim)
        critic_input = torch.cat([joint_features, global_state], dim=-1)  # (batch_size, critic_dim)
        self._cur_value = self.critic_head(critic_input).squeeze(-1)

        return action_logits, state

    def value_function(self) -> TensorType:
        """
        Return the cached value prediction computed during the forward pass.

        Returns:
            Tensor of shape (batch_size,) containing V(s) estimates.
        """
        assert self._cur_value is not None, "value_function() called before forward() pass."
        return self._cur_value

    def get_drop_frac(self) -> float:
        """
        Return the average fraction of in-range neighbors dropped by top-K sparsification across GAT layers.
        """
        return getattr(self, "_last_drop_frac", 0.0)

    def set_top_k(self, top_k: Optional[int]) -> None:
        """
        Set the top-K sparsification budget on the GNN layer and all its GAT sub-layers.

        Called externally (e.g. from train.py's iteration loop) to drive annealing
        from the training iteration count rather than a per-model-instance forward counter,
        which would desync across rollout workers.

        Args:
            top_k: Number of top neighbors to retain, or None for dense (no sparsification).
        """
        self.top_k = top_k
        self.gnn_layer.top_k = top_k
        if hasattr(self.gnn_layer, "gat_layers"):
            for layer in self.gnn_layer.gat_layers:
                layer.top_k = top_k


if __name__ == "__main__":
    # Verification smoke test
    print("Running verification smoke test for GNNMARLModel...")
    batch_size, num_robots, obs_dim, edge_dim = 4, 5, 24, 8
    
    # Mock observation dict as structured by spaces.Dict
    mock_obs = {
        "local_obs": torch.randn(batch_size, obs_dim),
        "node_features": torch.randn(batch_size, num_robots, obs_dim),
        "adj_matrix": torch.randint(0, 2, (batch_size, num_robots, num_robots)).float(),
        "edge_features": torch.randn(batch_size, num_robots, num_robots, edge_dim),
        "node_index": torch.randint(0, num_robots, (batch_size,)),
        "random_comm_mask": torch.rand(batch_size, num_robots, num_robots)
    }
    mock_input_dict = {"obs": mock_obs}
    
    # Mock ModelConfig with top_k passed through custom_model_config
    mock_cfg = {
        "custom_model_config": {
            "raw_obs_dim": obs_dim,
            "edge_dim": edge_dim,
            "comm_latent_dim": 32,
            "local_hidden_dim": 64,
            "gnn_num_layers": 2,
            "top_k": 2,
            "topk_mode": "attention"
        }
    }
    
    model = GNNMARLModel(
        obs_space=None,
        action_space=None,
        num_outputs=4,  # e.g., 2 continuous actions (mean + log_std for each)
        model_config=mock_cfg,
        name="test_gnn_marl_model"
    )
    
    logits, _ = model.forward(mock_input_dict, [], torch.ones(batch_size))
    values = model.value_function()
    drop_frac = model.get_drop_frac()
    
    assert logits.shape == (batch_size, 4), f"Expected logits shape {(batch_size, 4)}, got {logits.shape}"
    assert values.shape == (batch_size,), f"Expected values shape {(batch_size,)}, got {values.shape}"
    assert model.gnn_layer.top_k == 2, f"Expected gnn_layer.top_k to be 2, got {model.gnn_layer.top_k}"
    print(f"Logits shape: {logits.shape} | Values shape: {values.shape} | Average Drop Frac: {drop_frac:.4f}")
    print("Verification passed! GNNMARLModel integrates correctly with RLlib TorchModelV2 API and top-K sparsification.")

