# GNN-Comm-MARL

Multi-robot payload transport simulator with a graph-attention communication layer, trained with Ray RLlib PPO.

## Setup

```
pip install -r requirements.txt
```
## Run training

```
python train.py --num-robots 8 --num-workers 8 --max-iterations 300
```

## Debug with the PyBullet GUI

```
python train.py --num-robots 4 --num-workers 1 --render --max-iterations 10
```
