"""Finish a run from its checkpoint (use if training was stopped early / time ran out).
  python finish.py --run runs/pacman_s0 [--ckpt best|last] [--rollouts 500] [--random_baseline 50]
Writes rollout_rewards.npy, results.json and regenerates all figures from metrics.csv (saved every 100 episodes)."""
import argparse, json, os
import numpy as np
import dqn, plots

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--ckpt", default="best")
ap.add_argument("--rollouts", type=int, default=500); ap.add_argument("--random_baseline", type=int, default=0)
a = ap.parse_args()
path = os.path.join(a.run, f"{a.ckpt}.pt")
if not os.path.exists(path):
    path = os.path.join(a.run, "last.pt")
agent, cfg = dqn.load_agent(path, "cuda" if __import__("torch").cuda.is_available() else "cpu")
roll = dqn.evaluate(cfg, agent, a.rollouts, cfg.eval_eps, seed=100_000)
np.save(os.path.join(a.run, "rollout_rewards.npy"), roll)
res = dict(ckpt=path, rollout_mean=float(roll.mean()), rollout_std=float(roll.std()), rollout_n=len(roll))
if a.random_baseline:
    rb = dqn.evaluate(cfg, agent, a.random_baseline, 1.0, seed=200_000)  # eps=1 -> uniformly random policy
    res.update(random_mean=float(rb.mean()), random_std=float(rb.std()))
json.dump(res, open(os.path.join(a.run, "results.json"), "w"), indent=2)
plots.make_all(a.run); print(json.dumps(res, indent=2))
