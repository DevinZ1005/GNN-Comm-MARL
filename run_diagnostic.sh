#!/bin/bash
mkdir -p logs

python train.py --num-robots 8 --comm-radius 3.8 --num-workers 8 --max-iterations 150 --gnn-num-layers 1 --seed 0 --checkpoint-dir ./checkpoints/diag_dense_l1_s0 > logs/diag_dense_l1_s0.log 2>&1 &
python train.py --num-robots 8 --comm-radius 3.8 --num-workers 8 --max-iterations 150 --gnn-num-layers 1 --seed 1 --checkpoint-dir ./checkpoints/diag_dense_l1_s1 > logs/diag_dense_l1_s1.log 2>&1 &
wait

python train.py --num-robots 8 --comm-radius 3.8 --num-workers 8 --max-iterations 150 --gnn-num-layers 1 --top-k 2 --topk-mode attention --seed 0 --checkpoint-dir ./checkpoints/diag_attn_l1_s0 > logs/diag_attn_l1_s0.log 2>&1 &
python train.py --num-robots 8 --comm-radius 3.8 --num-workers 8 --max-iterations 150 --gnn-num-layers 1 --top-k 2 --topk-mode attention --seed 1 --checkpoint-dir ./checkpoints/diag_attn_l1_s1 > logs/diag_attn_l1_s1.log 2>&1 &
wait

python train.py --num-robots 8 --comm-radius 3.8 --num-workers 8 --max-iterations 150 --gnn-num-layers 1 --top-k 2 --topk-mode random --seed 0 --checkpoint-dir ./checkpoints/diag_rand_l1_s0 > logs/diag_rand_l1_s0.log 2>&1 &
python train.py --num-robots 8 --comm-radius 3.8 --num-workers 8 --max-iterations 150 --gnn-num-layers 1 --top-k 2 --topk-mode random --seed 1 --checkpoint-dir ./checkpoints/diag_rand_l1_s1 > logs/diag_rand_l1_s1.log 2>&1 &
wait

echo "ALL DONE"