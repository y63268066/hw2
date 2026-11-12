#!/bin/bash
# MsPacman, time-reduced recommended setting (single GPU). Usage: nohup ./run_pacman.sh > pacman_log.txt 2>&1 &
python dqn.py --env pacman --seed 0 --episodes 2000 \
  --eps_decay_steps 250000 --learn_start 5000 --target_update 5000 \
  --lr 2.5e-4 --lr_decay_updates 350000 --eval_every 200 --eval_episodes 5
