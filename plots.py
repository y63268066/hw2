"""All figures. `python plots.py --run runs/cartpole_s0` regenerates them from saved logs."""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def moving_avg(x, w=100):
    c = np.cumsum(np.insert(np.asarray(x, float), 0, 0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i + 1 - w)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def load_metrics(run):
    return np.genfromtxt(os.path.join(run, "metrics.csv"), delimiter=",", names=True)


def plot_maxq(m, run, name):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(m["episode"], m["max_q"], alpha=0.35, label="max Q in episode")
    ax.plot(m["episode"], moving_avg(m["max_q"], 100), lw=2, label="moving avg (100)")
    ax.set_xlabel("Training episode"); ax.set_ylabel("max_s,a Q(s,a)")
    ax.set_title(f"{name}: maximum Q-value vs. training episode"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(run, "fig_i_max_q.png"), dpi=150); plt.close(fig)


def plot_rewards(m, run, name):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(m["episode"], m["reward"], alpha=0.35, label="episode reward")
    ax.plot(m["episode"], moving_avg(m["reward"], 100), color="C3", lw=2, label="moving avg (last 100)")
    ax.set_xlabel("Training episode"); ax.set_ylabel("Episode reward")
    ax.set_title(f"{name}: episode reward vs. training episode"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(run, "fig_ii_rewards.png"), dpi=150); plt.close(fig)


def plot_hist(run, name):
    r = np.load(os.path.join(run, "rollout_rewards.npy"))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(r, bins=30, edgecolor="k", alpha=.8)
    ax.axvline(r.mean(), color="r", ls="--", label=f"mean = {r.mean():.1f}")
    ax.set_xlabel("Episode reward"); ax.set_ylabel("Count")
    ax.set_title(f"{name}: {len(r)} rollouts of trained agent (mean {r.mean():.1f}, std {r.std():.1f})")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(run, "fig_iii_rollout_hist.png"), dpi=150); plt.close(fig)


def plot_age_heatmap(run, name):
    """NON-STANDARD PLOT 1: how old are the tuples the learner samples, as training progresses?"""
    p = os.path.join(run, "age_hist.npz")
    if not os.path.exists(p):
        return
    d = np.load(p)
    cap = float(d["capacity"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(d["hist"].T, origin="lower", aspect="auto", cmap="magma",
                   extent=[d["steps"][0], d["steps"][-1], 0, cap])
    ax.plot(d["steps"], d["mean"], "c-", lw=1.5, label="mean sampled age")
    ax.set_xlabel("Environment step"); ax.set_ylabel("Age of sampled tuple (env steps)")
    ax.set_title(f"{name}: age distribution of replay samples during training")
    fig.colorbar(im, label="fraction of minibatch"); ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(run, "fig_nonstd_replay_age.png"), dpi=150); plt.close(fig)


def plot_policy_maps(run):
    """NON-STANDARD PLOT 2 (CartPole): evolution of Q(right)-Q(left) over the (angle, angular-velocity) plane.
    Expected after learning: positive (red = push right) when the pole leans/falls right, negative otherwise."""
    p = os.path.join(run, "policy_snapshots.npz")
    if not os.path.exists(p):
        return
    d = np.load(p)
    n = len(d["episode"]); cols = 4; rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.3 * rows), squeeze=False)
    vmax = np.abs(d["gap"]).max() + 1e-9
    for ax in axs.ravel():
        ax.axis("off")
    for i in range(n):
        ax = axs[i // cols][i % cols]; ax.axis("on")
        im = ax.imshow(d["gap"][i], origin="lower", extent=[-0.21, 0.21, -2, 2], aspect="auto",
                       cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.contour(np.linspace(-.21, .21, d["gap"][i].shape[1]), np.linspace(-2, 2, d["gap"][i].shape[0]),
                   d["gap"][i], levels=[0], colors="k", linewidths=1)
        ax.set_title(f"episode {int(d['episode'][i])}", fontsize=9)
        ax.set_xlabel("pole angle (rad)", fontsize=8); ax.set_ylabel("angular vel.", fontsize=8)
    fig.colorbar(im, ax=axs.ravel().tolist(), label="Q(push right) - Q(push left)")
    fig.suptitle("CartPole: learned decision boundary (black line) over training")
    fig.savefig(os.path.join(run, "fig_nonstd_policy_map.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def plot_eval_curve(run, name):
    p = os.path.join(run, "evals.json")
    if not os.path.exists(p):
        return
    e = json.load(open(p))
    if not e:
        return
    x = [d["episode"] for d in e]; y = np.array([d["mean"] for d in e]); s = np.array([d["std"] for d in e])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y); ax.fill_between(x, y - s, y + s, alpha=.25)
    ax.set_xlabel("Training episode"); ax.set_ylabel("Greedy-policy eval reward")
    ax.set_title(f"{name}: periodic evaluation (checkpoint selection)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(run, "fig_eval_curve.png"), dpi=150); plt.close(fig)


def make_all(run):
    cfg = json.load(open(os.path.join(run, "config.json")))
    name = "CartPole-v1" if cfg["env"] == "cartpole" else "MsPacman-v5"
    m = load_metrics(run)
    plot_maxq(m, run, name); plot_rewards(m, run, name); plot_eval_curve(run, name)
    if os.path.exists(os.path.join(run, "rollout_rewards.npy")):
        plot_hist(run, name)
    plot_age_heatmap(run, name); plot_policy_maps(run)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    make_all(ap.parse_args().run)
