#!/bin/bash
# usage: ./run_pacman.sh <seed>   (the report used seeds 0, 1, 2, each on its own GPU)
S=${1:-0}
python dqn.py --env pacman --seed $S --episodes 4000 --eps_decay_steps 100000 --learn_start 5000 --target_update 5000 \
  --lr 2.5e-4 --lr_decay_updates 100000 --eval_every 250 --eval_episodes 5 --out runs/pacman_long_s$S
