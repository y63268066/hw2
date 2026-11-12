"""Deep Q-Network for CartPole-v1 and ALE/MsPacman-v5 (single-file implementation).

Components required by the assignment: neural network, replay buffer, target network.
Usage:
    python dqn.py --env cartpole
    python dqn.py --env pacman
Any field of `Config` can be overridden from the command line, e.g. --episodes 100 --seed 1
"""
import argparse
import csv
import json
import os
import random
import time
from collections import deque
from dataclasses import dataclass, asdict, fields

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MSPACMAN_COLOR = 210 + 164 + 74


# --------------------------------------------------------------------------- preprocessing
def preprocess_observation(obs):
    """Pre-processing function provided in the assignment (unchanged)."""
    img = obs[1:176:2, ::2]  # crop and downsize
    img = img.sum(axis=2)  # to greyscale
    img[img == MSPACMAN_COLOR] = 0  # improve contrast
    img = (img // 3 - 128).astype(np.int8)  # normalize from -128 to 127
    return img.reshape(88, 80, 1)


class StatePreproc:
    """CartPole: raw float vector. MsPacman: last k preprocessed frames stacked -> (k, 88, 80) int8."""

    def __init__(self, env_name, k=4):
        self.pacman = env_name == "pacman"
        self.k = k
        self.frames = deque(maxlen=k)

    def reset(self, obs):
        if not self.pacman:
            return np.asarray(obs, dtype=np.float32)
        f = preprocess_observation(obs)[..., 0]
        for _ in range(self.k):
            self.frames.append(f)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        if not self.pacman:
            return np.asarray(obs, dtype=np.float32)
        self.frames.append(preprocess_observation(obs)[..., 0])
        return np.stack(self.frames, axis=0)


# --------------------------------------------------------------------------- config
@dataclass
class Config:
    env: str = "cartpole"
    seed: int = 0
    episodes: int = 600
    gamma: float = 0.95
    lr: float = 5e-4
    lr_final_frac: float = 1.0  # final lr = lr * lr_final_frac (linear decay over lr_decay_updates)
    lr_decay_updates: int = 0
    batch_size: int = 64
    buffer_size: int = 50_000
    learn_start: int = 1_000
    train_freq: int = 1
    target_update: int = 0  # >0: hard copy every N env steps. 0: soft update with tau every update
    tau: float = 0.005
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay_steps: int = 5_000
    double: bool = False
    huber: bool = True
    grad_clip: float = 10.0
    reward_clip: bool = False  # clip reward to sign(r) for training only (plots use raw rewards)
    eval_every: int = 10
    eval_episodes: int = 10
    eval_eps: float = 0.0
    final_rollouts: int = 500
    final_ckpt: str = "best"  # best | last
    age_every: int = 200  # log replay-age histogram every N updates
    snapshot_count: int = 8  # CartPole policy-map snapshots
    out: str = ""
    device: str = "auto"


PRESETS = {
    "cartpole": dict(),
    "pacman": dict(
        episodes=5000, gamma=0.99, lr=1e-4, lr_final_frac=0.2, lr_decay_updates=700_000,
        batch_size=32, buffer_size=60_000, learn_start=10_000, train_freq=4,
        target_update=10_000, eps_start=1.0, eps_end=0.05, eps_decay_steps=500_000,
        reward_clip=True, eval_every=250, eval_episodes=10, eval_eps=0.05,
    ),
}


def make_config(env, **overrides):
    cfg = Config(env=env, **PRESETS[env])
    for k, v in overrides.items():
        if v is not None:
            setattr(cfg, k, v)
    if not cfg.out:
        cfg.out = f"runs/{env}_s{cfg.seed}"
    return cfg


# --------------------------------------------------------------------------- env
def make_env(name):
    if name == "cartpole":
        return gym.make("CartPole-v1")
    try:
        import ale_py
        gym.register_envs(ale_py)
    except Exception:
        pass
    return gym.make("ALE/MsPacman-v5")  # 9 actions (full_action_space=False)


def done_flag(terminated, truncated):
    """Only true termination zeroes the bootstrap. CartPole's 500-step time limit is a *truncation*:
    the state is not terminal, so we must still bootstrap from it."""
    return float(terminated)


# --------------------------------------------------------------------------- networks
class MLPQ(nn.Module):
    def __init__(self, in_dim=4, n_actions=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                                 nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x):
        return self.net(x.float())


class CNNQ(nn.Module):
    def __init__(self, in_ch=4, n_actions=9):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                                  nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())
        self.fc = nn.Sequential(nn.Linear(64 * 7 * 6, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x):
        return self.fc(self.conv(x.float() / 128.0))


def build_net(env_name, n_actions):
    return MLPQ(4, n_actions) if env_name == "cartpole" else CNNQ(4, n_actions)


# --------------------------------------------------------------------------- replay buffer
class ReplayBuffer:
    def __init__(self, capacity, state_shape, state_dtype):
        self.capacity = capacity
        self.s = np.zeros((capacity, *state_shape), dtype=state_dtype)
        self.s2 = np.zeros((capacity, *state_shape), dtype=state_dtype)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)
        self.t = np.zeros(capacity, dtype=np.int64)  # env step at which the tuple was inserted (for age)
        self.idx = 0
        self.size = 0

    def add(self, s, a, r, s2, d, step):
        i = self.idx
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i], self.t[i] = s, a, r, s2, d, step
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        return rng.integers(0, self.size, batch_size)

    def batch(self, idx):
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


# --------------------------------------------------------------------------- agent
def compute_targets(target_net, online_net, r, s2, d, gamma, double=False):
    """y = r + gamma * (1-d) * max_a' Q_target(s',a')   (or Double-DQN variant)."""
    with torch.no_grad():
        if double:
            a2 = online_net(s2).argmax(1, keepdim=True)
            q2 = target_net(s2).gather(1, a2).squeeze(1)
        else:
            q2 = target_net(s2).max(1)[0]
        return r + gamma * (1.0 - d) * q2


class DQNAgent:
    def __init__(self, net_fn, n_actions, gamma, lr, device="cpu", tau=0.0, double=False, huber=True,
                 grad_clip=10.0, lr_final_frac=1.0, lr_decay_updates=0, seed=0):
        self.device = torch.device(device)
        self.q = net_fn().to(self.device)
        self.target = net_fn().to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.target.eval()
        self.n_actions, self.gamma, self.tau = n_actions, gamma, tau
        self.double, self.huber, self.grad_clip = double, huber, grad_clip
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        if lr_decay_updates > 0:
            f = lambda u: 1.0 - (1.0 - lr_final_frac) * min(1.0, u / lr_decay_updates)
            self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, f)
        else:
            self.sched = None
        self.rng = np.random.default_rng(seed)

    def act(self, state, eps):
        with torch.no_grad():
            q = self.q(torch.as_tensor(state, device=self.device).unsqueeze(0))[0]
        if self.rng.random() < eps:
            a = int(self.rng.integers(self.n_actions))
        else:
            a = int(q.argmax().item())
        return a, float(q.max().item())

    def update(self, batch):
        s, a, r, s2, d = [torch.as_tensor(x, device=self.device) for x in batch]
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        y = compute_targets(self.target, self.q, r, s2, d, self.gamma, self.double)
        loss = F.smooth_l1_loss(q, y) if self.huber else F.mse_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        if self.grad_clip:
            nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.opt.step()
        if self.sched:
            self.sched.step()
        if self.tau > 0:  # soft target update (tau=1 -> target == online, i.e. "no target network")
            with torch.no_grad():
                for tp, p in zip(self.target.parameters(), self.q.parameters()):
                    tp.mul_(1 - self.tau).add_(self.tau * p)
        return float(loss.item())

    def sync_target(self):
        self.target.load_state_dict(self.q.state_dict())


def epsilon(cfg, step):
    frac = min(1.0, step / max(1, cfg.eps_decay_steps))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def build_agent(cfg, n_actions, device):
    return DQNAgent(lambda: build_net(cfg.env, n_actions), n_actions, cfg.gamma, cfg.lr, device,
                    tau=cfg.tau if cfg.target_update == 0 else 0.0, double=cfg.double, huber=cfg.huber,
                    grad_clip=cfg.grad_clip, lr_final_frac=cfg.lr_final_frac,
                    lr_decay_updates=cfg.lr_decay_updates, seed=cfg.seed)


def load_agent(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ck["cfg"])
    n_actions = 2 if cfg.env == "cartpole" else 9
    agent = build_agent(cfg, n_actions, device)
    agent.q.load_state_dict(ck["q"])
    agent.sync_target()
    return agent, cfg


def save_ckpt(agent, cfg, path, episode):
    torch.save({"q": agent.q.state_dict(), "cfg": asdict(cfg), "episode": episode}, path)


# --------------------------------------------------------------------------- evaluation
def evaluate(cfg, agent, n, eps, seed=None):
    env = make_env(cfg.env)
    pre = StatePreproc(cfg.env)
    rewards = []
    for i in range(n):
        obs, _ = env.reset(seed=None if seed is None else seed + i)
        s = pre.reset(obs)
        tot, done = 0.0, False
        while not done:
            a, _ = agent.act(s, eps)
            obs, r, term, trunc, _ = env.step(a)
            s = pre.step(obs)
            tot += r
            done = term or trunc
        rewards.append(tot)
    env.close()
    return np.array(rewards)


def cartpole_policy_gap(agent, n=41):
    """Q(push right) - Q(push left) on a (theta, theta_dot) grid with x = x_dot = 0."""
    th = np.linspace(-0.21, 0.21, n)
    thd = np.linspace(-2.0, 2.0, n)
    T, TD = np.meshgrid(th, thd)
    S = np.zeros((n * n, 4), dtype=np.float32)
    S[:, 2], S[:, 3] = T.ravel(), TD.ravel()
    with torch.no_grad():
        q = agent.q(torch.as_tensor(S, device=agent.device)).cpu().numpy()
    return (q[:, 1] - q[:, 0]).reshape(n, n)


# --------------------------------------------------------------------------- training
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(cfg, verbose=True):
    os.makedirs(cfg.out, exist_ok=True)
    seed_all(cfg.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if cfg.device == "auto" else cfg.device
    env = make_env(cfg.env)
    env.action_space.seed(cfg.seed)
    pre = StatePreproc(cfg.env)
    obs, _ = env.reset(seed=cfg.seed)
    s0 = pre.reset(obs)
    n_actions = env.action_space.n
    agent = build_agent(cfg, n_actions, device)
    buffer = ReplayBuffer(cfg.buffer_size, s0.shape, s0.dtype)
    json.dump(asdict(cfg), open(os.path.join(cfg.out, "config.json"), "w"), indent=2)

    rows, evals, age_steps, age_hist, age_mean = [], [], [], [], []
    snaps = {"episode": [], "gap": []}
    snap_eps = set(np.linspace(0, cfg.episodes - 1, cfg.snapshot_count).astype(int).tolist())
    age_bins = np.linspace(0, cfg.buffer_size, 41)
    gstep, n_upd, best_eval, t0 = 0, 0, -np.inf, time.time()

    for ep in range(cfg.episodes):
        obs, _ = env.reset(seed=cfg.seed if ep == 0 else None)
        s = pre.reset(obs)
        ep_r, ep_len, qmax, qsum, losses = 0.0, 0, -np.inf, 0.0, []
        while True:
            eps = epsilon(cfg, gstep)
            a, qv = agent.act(s, eps)
            qmax, qsum = max(qmax, qv), qsum + qv
            obs2, r, term, trunc, _ = env.step(a)
            s2 = pre.step(obs2)
            buffer.add(s, a, float(np.sign(r)) if cfg.reward_clip else float(r), s2, done_flag(term, trunc), gstep)
            s, ep_r, ep_len, gstep = s2, ep_r + r, ep_len + 1, gstep + 1
            if buffer.size >= cfg.learn_start and gstep % cfg.train_freq == 0:
                idx = buffer.sample(cfg.batch_size, agent.rng)
                losses.append(agent.update(buffer.batch(idx)))
                n_upd += 1
                if n_upd % cfg.age_every == 0:
                    ages = gstep - buffer.t[idx]
                    h, _ = np.histogram(ages, bins=age_bins)
                    age_steps.append(gstep)
                    age_hist.append(h / len(ages))
                    age_mean.append(float(ages.mean()))
            if cfg.target_update > 0 and gstep % cfg.target_update == 0:
                agent.sync_target()
            if term or trunc:
                break
        rows.append(dict(episode=ep, steps=ep_len, reward=ep_r, max_q=qmax, mean_q=qsum / ep_len, eps=eps,
                         loss=float(np.mean(losses)) if losses else np.nan, global_step=gstep))
        if cfg.env == "cartpole" and ep in snap_eps:
            snaps["episode"].append(ep)
            snaps["gap"].append(cartpole_policy_gap(agent))
        if (ep + 1) % cfg.eval_every == 0 and cfg.eval_episodes > 0:
            er = evaluate(cfg, agent, cfg.eval_episodes, cfg.eval_eps, seed=50_000 + ep)
            evals.append(dict(episode=ep, mean=float(er.mean()), std=float(er.std())))
            if er.mean() > best_eval:
                best_eval = er.mean()
                save_ckpt(agent, cfg, os.path.join(cfg.out, "best.pt"), ep)
        if verbose and (ep + 1) % 10 == 0:
            ma = np.mean([x["reward"] for x in rows[-100:]])
            print(f"[{cfg.env} s{cfg.seed}] ep {ep+1}/{cfg.episodes} step {gstep} R {ep_r:.0f} MA100 {ma:.1f} "
                  f"maxQ {qmax:.2f} eps {eps:.3f} best_eval {best_eval:.1f} {time.time()-t0:.0f}s", flush=True)
        if (ep + 1) % 100 == 0:
            _dump(cfg, rows, evals, age_steps, age_hist, age_mean, snaps)
    save_ckpt(agent, cfg, os.path.join(cfg.out, "last.pt"), cfg.episodes - 1)
    _dump(cfg, rows, evals, age_steps, age_hist, age_mean, snaps)
    env.close()

    result = {"best_eval": float(best_eval), "train_minutes": (time.time() - t0) / 60}
    if cfg.final_rollouts > 0:
        ck = os.path.join(cfg.out, "best.pt")
        if cfg.final_ckpt == "best" and os.path.exists(ck):
            agent.q.load_state_dict(torch.load(ck, map_location=device, weights_only=False)["q"])
        roll = evaluate(cfg, agent, cfg.final_rollouts, cfg.eval_eps, seed=100_000)
        np.save(os.path.join(cfg.out, "rollout_rewards.npy"), roll)
        result.update(rollout_mean=float(roll.mean()), rollout_std=float(roll.std()), rollout_min=float(roll.min()),
                      rollout_max=float(roll.max()), rollout_n=int(len(roll)),
                      frac_500=float((roll >= 500).mean()) if cfg.env == "cartpole" else None)
        json.dump(result, open(os.path.join(cfg.out, "results.json"), "w"), indent=2)
        try:
            import plots
            plots.make_all(cfg.out)
        except Exception as e:  # plotting must never lose a finished training run
            print("plotting failed:", e)
        if verbose:
            print(json.dumps(result, indent=2))
    return result


def _dump(cfg, rows, evals, age_steps, age_hist, age_mean, snaps):
    with open(os.path.join(cfg.out, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(evals, open(os.path.join(cfg.out, "evals.json"), "w"))
    if age_hist:
        np.savez(os.path.join(cfg.out, "age_hist.npz"), steps=np.array(age_steps), hist=np.array(age_hist),
                 mean=np.array(age_mean), capacity=cfg.buffer_size)
    if snaps["gap"]:
        np.savez(os.path.join(cfg.out, "policy_snapshots.npz"), episode=np.array(snaps["episode"]),
                 gap=np.array(snaps["gap"]))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", choices=["cartpole", "pacman"], required=True)
    for f in fields(Config):
        if f.name == "env":
            continue
        t = (lambda x: str(x).lower() in ("1", "true", "yes")) if f.type is bool else f.type
        p.add_argument(f"--{f.name}", type=t, default=None)
    a = vars(p.parse_args())
    env = a.pop("env")
    return make_config(env, **a)


if __name__ == "__main__":
    train(parse_args())
