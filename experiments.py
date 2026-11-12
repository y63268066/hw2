"""Validation experiments.
  python experiments.py q_vs_return --run runs/cartpole_s0      # Q estimates vs. Monte-Carlo discounted returns
  python experiments.py ablations --seeds 3 --episodes 300      # remove/alter one component, compare to prediction
"""
import argparse
import json
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dqn
from plots import moving_avg, load_metrics


def q_vs_return(run, n_episodes=100, horizon_used=250):
    """CartPole with gamma=0.95 and a 500-step cap: the maximum possible Q is (1-g^500)/(1-g) = 20.
    For a (near-)optimal policy, Q(s_t,a_t) must match the Monte-Carlo discounted return G_t.  We use t <= 250 so
    the truncated tail (g^250 ~ 3e-6) is negligible.  Mean(Q - G) > 0 quantifies over-estimation bias."""
    agent, cfg = dqn.load_agent(os.path.join(run, os.environ.get("CKPT", "best") + ".pt"))
    env = dqn.make_env("cartpole"); pre = dqn.StatePreproc("cartpole")
    Q, G, lens = [], [], []
    for ep in range(n_episodes):
        o, _ = env.reset(seed=7000 + ep); s = pre.reset(o); qs, rs = [], []
        while True:
            a, q = agent.act(s, 0.0); qs.append(q)
            o, r, te, tr, _ = env.step(a); s = pre.step(o); rs.append(r)
            if te or tr:
                break
        lens.append(len(rs)); g = 0.0; ret = np.zeros(len(rs))
        for t in reversed(range(len(rs))):
            g = rs[t] + cfg.gamma * g; ret[t] = g
        m = min(len(rs), horizon_used) if (tr and not te) else len(rs)
        if True:  # only full-length episodes have an (almost) untruncated return for t<=250
            Q += qs[:m]; G += ret[:m].tolist()
    Q, G = np.array(Q), np.array(G)
    out = dict(n_states=int(len(Q)), episodes_at_500=int(sum(l >= 500 for l in lens)), mean_Q=float(Q.mean()),
               mean_G=float(G.mean()), bias=float((Q - G).mean()), rmse=float(np.sqrt(((Q - G) ** 2).mean())),
               theoretical_max_Q=float((1 - cfg.gamma ** 500) / (1 - cfg.gamma)))
    json.dump(out, open(os.path.join(run, "q_vs_return.json"), "w"), indent=2)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(G, Q, s=4, alpha=.3); lo, hi = min(G.min(), Q.min()), max(G.max(), Q.max())
    ax.plot([lo, hi], [lo, hi], "r--", label="Q = G"); ax.set_xlabel("MC discounted return G_t"); ax.set_ylabel("Q(s_t,a_t)")
    ax.set_title(f"CartPole: Q vs. return (bias {out['bias']:+.2f}, RMSE {out['rmse']:.2f})"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(run, "fig_q_vs_return.png"), dpi=150)
    print(json.dumps(out, indent=2))


ABLATIONS = {  # name: (overrides, prediction)
    "baseline": ({}, "solves CartPole; max Q -> ~20"),
    "no_target_net": (dict(tau=1.0, target_update=0), "less stable / slower; larger Q oscillations"),
    "no_replay": (dict(buffer_size=64, learn_start=64), "highly correlated data -> unstable, worse"),
    "gamma_0.5": (dict(gamma=0.5), "max Q -> 1/(1-0.5) = 2 (return limit changes accordingly)"),
    "lr_1e-2": (dict(lr=1e-2), "too large step size -> unstable"),
}


def ablations(seeds, episodes, root="runs/ablations"):
    res = {}
    for name, (ov, _) in ABLATIONS.items():
        res[name] = []
        for sd in range(seeds):
            cfg = dqn.make_config("cartpole", seed=sd, episodes=episodes, final_rollouts=0, eval_episodes=0,
                                  out=f"{root}/{name}_s{sd}", eps_decay_steps=5000, **ov)
            dqn.train(cfg, verbose=False)
            res[name].append(load_metrics(cfg.out))
            print("done", name, sd, flush=True)
    summary = {}
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, runs in res.items():
        R = np.array([moving_avg(m["reward"], 100) for m in runs]); Qm = np.array([moving_avg(m["max_q"], 20) for m in runs])
        x = np.arange(R.shape[1])
        for ax, A in zip(axs, (R, Qm)):
            ax.plot(x, A.mean(0), label=name); ax.fill_between(x, A.min(0), A.max(0), alpha=.15)
        summary[name] = dict(final_MA100_reward=float(R[:, -1].mean()), final_max_Q=float(Qm[:, -1].mean()),
                             prediction=ABLATIONS[name][1])
    axs[0].set_title("Moving-average(100) reward (mean, min-max over seeds)"); axs[1].set_title("Max Q (moving avg 20)")
    for ax in axs:
        ax.set_xlabel("Episode"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); os.makedirs(root, exist_ok=True); fig.savefig(f"{root}/fig_ablations.png", dpi=150)
    json.dump(summary, open(f"{root}/summary.json", "w"), indent=2); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["q_vs_return", "ablations"])
    ap.add_argument("--run", default="runs/cartpole_s0"); ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=300)
    a = ap.parse_args()
    q_vs_return(a.run) if a.cmd == "q_vs_return" else ablations(a.seeds, a.episodes)
