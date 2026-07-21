# Dynamic Topological GNN & Decentralized MARL Simulator

A state-of-the-art multi-robot coordination simulator using **Dynamic Topological Graph Neural Networks (GNNs)** combined with **Decentralized Multi-Agent Proximal Policy Optimization (MAPPO)**.

## Project Structure

```
marl_gnn_simulator/
├── env_core.py             # Custom MultiAgentEnv wrapper with PyBullet/MuJoCo engine
├── gnn_comm_layer.py       # PyTorch dynamic topological GNN / Graph Attention layers
├── marl_agent.py           # RLlib TorchModelV2 integrating GNN and local PPO policy head
├── train.py                # Distributed Ray RLlib PPO training entrypoint
├── requirements.txt        # Pinned dependencies
└── README.md               # Quickstart guide and architecture overview
```

## Quickstart

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run Training
To start multi-agent training with Ray RLlib:
```powershell
python train.py --num-robots 4 --num-workers 2 --max-iterations 100
```

### 3. Run with PyBullet GUI (Debug Mode)
```powershell
python train.py --num-robots 4 --num-workers 1 --render --max-iterations 10
```

## Key Architectural Concepts

- **Dynamic Adjacency Topology**: At each physics step ($1/240\text{s}$), the environment computes pairwise Euclidean distances ($d_{ij}$). Edges $(j, i)$ are formed if $d_{ij} \le R_{\text{comm}}$. Edge features encode relative coordinates, velocities, and normalized direction vectors.
- **Topological Message Passing**: Agents perform $L$-layer edge-conditioned graph attention over neighborhood feature vectors. Information hops across physical links without centralized state broadcasts.
- **Potential-Based Reward Shaping**: Guarantees policy convergence without artificial local minima loops by shaping rewards against target payload distance reductions and formation connectivity preservation.
