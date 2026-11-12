# Assignment 2: Deep Q-Network (CartPole-v1, MsPacman-v5)

Keyu Yan, GitHub: y63268066, repo: https://github.com/y63268066/hw2

## Files
- `dqn.py`: DQN (network, replay buffer, target network, training, rollouts)
- `plots.py`: figures
- `tests.py`: validation tests
- `experiments.py`: Q-vs-return check and ablations
- `finish.py`: 500 rollouts and figures from a saved checkpoint
- `run_pacman.sh`: MsPacman command (the run in the report used 500 episodes, see below)

## How to run
~~~
pip install -r requirements.txt
python tests.py
python dqn.py --env cartpole --seed 1 --out runs/cp_s1
python dqn.py --env pacman --episodes 500 --eps_decay_steps 100000 --learn_start 5000 --target_update 5000 \
    --lr 2.5e-4 --lr_decay_updates 100000 --eval_every 100 --eval_episodes 3 --final_rollouts 0
python finish.py --run runs/pacman_s0 --rollouts 500 --random_baseline 20
python experiments.py q_vs_return --run runs/cp_s1
CKPT=last python experiments.py q_vs_return --run runs/cp_s1
python experiments.py ablations --seeds 2 --episodes 250
~~~
Each run writes metrics.csv, results.json and the figures into its folder under `runs/`.

## Results (runs/)
- `cp_s1`: CartPole main run. `cartpole_s0`, `cp_s2`: other seeds. `cp_s1_double`: Double DQN.
- `pacman_s0`: MsPacman run (500 episodes) with 500 rollouts.
- `ablations`: ablation runs and summary.json.
- Logs: `test_output.txt`, `pacman_log.txt`, `cp1.log`, `cp2.log`, `cp3.log`, `abl_log.txt`.
