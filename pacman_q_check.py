import argparse, json, os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dqn

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--episodes", type=int, default=20)
ap.add_argument("--ckpt", default="best")
ap.add_argument("--eps", type=float, default=0.05)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
path = os.path.join(a.run, a.ckpt + ".pt")
agent, cfg = dqn.load_agent(path, dev)
env = dqn.make_env("pacman")
pre = dqn.StatePreproc("pacman")
Q, G, scores = [], [], []
for i in range(a.episodes):
    obs, _ = env.reset(seed=300000 + i)
    s = pre.reset(obs)
    qs, rs, tot, done = [], [], 0.0, False
    while not done:
        act, q = agent.act(s, a.eps)
        qs.append(q)
        obs, r, te, tr, _ = env.step(act)
        s = pre.step(obs)
        tot += r
        done = te or tr
        rs.append(float(np.sign(r)) if cfg.reward_clip else float(r))
    g, ret = 0.0, np.zeros(len(rs))
    for t in reversed(range(len(rs))):
        g = rs[t] + cfg.gamma * g
        ret[t] = g
    Q += qs
    G += ret.tolist()
    scores.append(tot)
Q, G = np.array(Q), np.array(G)
out = dict(checkpoint=a.ckpt, episodes=a.episodes, n_states=int(len(Q)))
out["mean_score"] = float(np.mean(scores))
out["mean_Q"] = float(Q.mean())
out["mean_G"] = float(G.mean())
out["bias"] = float((Q - G).mean())
out["rmse"] = float(np.sqrt(((Q - G) ** 2).mean()))
out["correlation"] = float(np.corrcoef(Q, G)[0, 1])
json.dump(out, open(os.path.join(a.run, "q_vs_return.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
rng = np.random.default_rng(0)
idx = rng.choice(len(Q), size=min(8000, len(Q)), replace=False)
fig, ax = plt.subplots(figsize=(5.2, 4.8))
ax.scatter(G[idx], Q[idx], s=3, alpha=0.25)
lo, hi = min(G.min(), Q.min()), max(G.max(), Q.max())
ax.plot([lo, hi], [lo, hi], "r--", label="Q = G")
ax.set_xlabel("discounted return G_t (clipped rewards)")
ax.set_ylabel("max Q(s_t,a)")
ax.set_title("MsPacman: Q vs. return (bias %.2f, corr %.2f)" % (out["bias"], out["correlation"]), fontsize=9)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(a.run, "fig_q_vs_return.png"), dpi=150)
