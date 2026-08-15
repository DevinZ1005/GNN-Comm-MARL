#!/bin/bash
mkdir -p logs

run_pair() {
  local mode=$1; local topk=$2; local extra=$3; local s1=$4; local s2=$5
  local name1="${mode}_s${s1}"
  local name2="${mode}_s${s2}"
  python train.py --num-robots 8 --comm-radius 3.8 --num-workers 8 --max-iterations 300 \
    --gnn-num-layers 2 $topk $extra --seed $s1 --checkpoint-dir ./checkpoints/real_${name1} \
    > logs/real_${name1}.log 2>&1 &
  python train.py --num-robots 8 --comm-radius 3.8 --num-workers 8 --max-iterations 300 \
    --gnn-num-layers 2 $topk $extra --seed $s2 --checkpoint-dir ./checkpoints/real_${name2} \
    > logs/real_${name2}.log 2>&1 &
  wait
}

# Dense seed 4 (last one — s0-3 already done)
python train.py --num-robots 8 --comm-radius 3.8 --num-workers 8 --max-iterations 300 \
  --gnn-num-layers 2 --seed 4 --checkpoint-dir ./checkpoints/real_dense_s4 \
  > logs/real_dense_s4.log 2>&1 &
wait

# Attention K=2, 5 seeds
run_pair "attn" "--top-k 2" "--topk-mode attention" 0 1
run_pair "attn" "--top-k 2" "--topk-mode attention" 2 3
python train.py --num-robots 8 --comm-radius 3.8 --num-workers 8 --max-iterations 300 \
  --gnn-num-layers 2 --top-k 2 --topk-mode attention --seed 4 --checkpoint-dir ./checkpoints/real_attn_s4 \
  > logs/real_attn_s4.log 2>&1 &
wait

# Random K=2, 5 seeds
run_pair "rand" "--top-k 2" "--topk-mode random" 0 1
run_pair "rand" "--top-k 2" "--topk-mode random" 2 3
python train.py --num-robots 8 --comm-radius 3.8 --num-workers 8 --max-iterations 300 \
  --gnn-num-layers 2 --top-k 2 --topk-mode random --seed 4 --checkpoint-dir ./checkpoints/real_rand_s4 \
  > logs/real_rand_s4.log 2>&1 &
wait

echo "ALL DONE"
