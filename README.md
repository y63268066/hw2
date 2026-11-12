# Assignment 2: Deep Q-Network (CartPole-v1, MsPacman-v5)

Keyu Yan, GitHub: y63268066, repo: https://github.com/y63268066/hw2

## Files
- `dqn.py`: DQN (network, replay buffer, target network, training, rollouts)
- `plots.py`: figures
- `tests.py`: validation tests
- `experiments.py`: Q-vs-return check and ablations
- `finish.py`: 500 rollouts and figures from a saved checkpoint
- `run_pacman.sh`: MsPacman command used for the main runs

## How to run
~~~
pip install -r requirements.txt
python tests.py
python dqn.py --env cartpole --seed 1 --out runs/cp_s1
python dqn.py --env pacman --seed 0 --episodes 4000 --eps_decay_steps 100000 --learn_start 5000 --target_update 5000 \
    --lr 2.5e-4 --lr_decay_updates 100000 --eval_every 250 --eval_episodes 5 --out runs/pacman_long_s0   # also seeds 1 and 2
python finish.py --run runs/pacman_s0 --rollouts 500 --random_baseline 20   # random baseline (first 500-episode run)
python experiments.py q_vs_return --run runs/cp_s1
CKPT=last python experiments.py q_vs_return --run runs/cp_s1
python experiments.py ablations --seeds 2 --episodes 250
~~~
Each run writes metrics.csv, results.json and the figures into its folder under `runs/`.

## Results (runs/)
- `cp_s1`: CartPole main run. `cartpole_s0`, `cp_s2`: other seeds. `cp_s1_double`: Double DQN.
- `pacman_long_s0`, `pacman_long_s1`, `pacman_long_s2`: MsPacman, 4000 episodes each (used in the report).
- `pacman_s0`: first 500-episode test run of MsPacman (also gives the random baseline).
- `ablations`: CartPole ablation runs and summary.json.
- Logs: `test_output.txt`, `pacman_log.txt`, `pacman_long_s*.log`, `cp1.log`, `cp2.log`, `cp3.log`, `abl_log.txt`.

## MsPacman checks
- Q vs return: `python pacman_q_check.py --run runs/pacman_long_s0 --episodes 20` (seeds 0, 1, 2)
- Parameter changes: the 500-episode command with one change, `--target_update 1`, `--buffer_size 32 --learn_start 32`, or `--gamma 0.9`
- Results: `runs/pacman_abl_no_target`, `runs/pacman_abl_no_replay`, `runs/pacman_abl_gamma09`
