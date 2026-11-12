"""Validation tests (run: python tests.py).  Each test documents WHAT is checked and WHY it validates DQN.
These go beyond format checks: they compare the learner with analytic ground truth and known edge cases."""
import sys
import traceback

import gymnasium as gym
import numpy as np
import torch

from dqn import (Config, DQNAgent, MLPQ, ReplayBuffer, compute_targets, done_flag, epsilon,
                 preprocess_observation)


# 1 ---------------------------------------------------------------- preprocessing edge cases
def test_preprocess_range_and_contrast():
    """Output shape/dtype/range match the network input; the Ms. Pac-Man colour pixel is mapped to background."""
    rng = np.random.default_rng(0)
    obs = rng.integers(0, 256, (210, 160, 3), dtype=np.uint8)
    out = preprocess_observation(obs.copy())
    assert out.shape == (88, 80, 1) and out.dtype == np.int8
    assert out.min() >= -128 and out.max() <= 127
    obs = np.zeros((210, 160, 3), dtype=np.uint8)
    obs[1, 0] = (210, 164, 74)  # Pac-Man colour -> sum 448 -> set to 0 -> -128
    assert preprocess_observation(obs)[0, 0, 0] == -128
    obs[1, 0] = (255, 255, 255)  # white -> 765//3-128 = 127
    assert preprocess_observation(obs)[0, 0, 0] == 127


# 2 ---------------------------------------------------------------- replay buffer
def test_replay_buffer_fifo_and_age():
    """Circular overwrite keeps the newest `capacity` tuples; ages = current_step - insert_step are exact."""
    b = ReplayBuffer(5, (2,), np.float32)
    for i in range(8):
        b.add(np.full(2, i, np.float32), i % 2, float(i), np.full(2, i + 1, np.float32), 0.0, step=i)
    assert b.size == 5
    assert sorted(b.t.tolist()) == [3, 4, 5, 6, 7]
    assert sorted(b.r.tolist()) == [3, 4, 5, 6, 7]
    ages = 10 - b.t
    assert ages.min() == 3 and ages.max() == 7
    idx = b.sample(1000, np.random.default_rng(0))
    assert idx.min() >= 0 and idx.max() < 5  # never samples unfilled slots
    b2 = ReplayBuffer(100, (2,), np.float32); b2.add(np.zeros(2), 0, 0., np.zeros(2), 0., 0)
    assert b2.sample(50, np.random.default_rng(0)).max() == 0  # partially filled buffer


# 3 ---------------------------------------------------------------- Bellman target
class _Const(torch.nn.Module):
    def __init__(self, table):
        super().__init__(); self.t = torch.tensor(table, dtype=torch.float32)

    def forward(self, x):
        return self.t.expand(x.shape[0], -1)


def test_bellman_target_terminal_and_double():
    """Hand-computed targets: terminal -> y=r (no bootstrap); non-terminal -> r+gamma*max Q_target; Double DQN
    evaluates the online argmax under the target net."""
    tgt = _Const([[1.0, 5.0]]); onl = _Const([[9.0, 0.0]])
    r = torch.tensor([1.0, 1.0]); d = torch.tensor([1.0, 0.0]); s2 = torch.zeros(2, 3)
    y = compute_targets(tgt, onl, r, s2, d, 0.95)
    assert torch.allclose(y, torch.tensor([1.0, 1.0 + 0.95 * 5.0]))
    yd = compute_targets(tgt, onl, r, s2, d, 0.95, double=True)  # online argmax = action 0 -> Q_target=1
    assert torch.allclose(yd, torch.tensor([1.0, 1.0 + 0.95 * 1.0]))


def test_truncation_is_not_terminal():
    """CartPole's 500-step cap is a TimeLimit truncation; we must keep bootstrapping there (done=0)."""
    env = gym.make("CartPole-v1", max_episode_steps=3)
    env.reset(seed=0)
    for t in range(3):
        _, _, term, trunc, _ = env.step(1 if t % 2 else 0)
    assert trunc and not term
    assert done_flag(term, trunc) == 0.0
    assert done_flag(True, False) == 1.0


# 4 ---------------------------------------------------------------- learning-rule correctness
def test_overfit_fixed_batch():
    """With a frozen target net the regression target is fixed, so training loss must go to ~0."""
    torch.manual_seed(0); rng = np.random.default_rng(0)
    ag = DQNAgent(lambda: MLPQ(4, 2, 64), 2, 0.95, 3e-3, "cpu", seed=0)
    batch = (rng.normal(size=(32, 4)).astype(np.float32), rng.integers(0, 2, 32), rng.normal(size=32).astype(np.float32),
             rng.normal(size=(32, 4)).astype(np.float32), (rng.random(32) < .2).astype(np.float32))
    losses = [ag.update(batch) for _ in range(1500)]
    assert losses[-1] < 1e-3 < losses[0], (losses[0], losses[-1])


def test_chain_mdp_matches_analytic_q():
    """Deterministic 6-state chain (reward 1 on reaching the goal, gamma=0.95): the exact optimal Q* is known in
    closed form: Q*(s,right)=g^(N-2-s), Q*(s,left)=g^(N-s) (s>0), g^(N-1) (s=0). DQN trained on the full set of
    transitions with a periodically-synced target must reproduce it (tests update rule + target network)."""
    N, g = 6, 0.95
    S = np.eye(N, dtype=np.float32)
    tr = []
    for s in range(N - 1):
        for a in (0, 1):
            s2 = min(s + 1, N - 1) if a == 1 else max(s - 1, 0)
            term = s2 == N - 1
            tr.append((s, a, 1.0 if term else 0.0, s2, float(term)))
    s, a, r, s2, d = map(np.array, zip(*tr))
    batch = (S[s], a.astype(np.int64), r.astype(np.float32), S[s2], d.astype(np.float32))
    torch.manual_seed(0)
    ag = DQNAgent(lambda: MLPQ(N, 2, 64), 2, g, 1e-3, "cpu", seed=0)
    for u in range(6000):
        ag.update(batch)
        if (u + 1) % 50 == 0:
            ag.sync_target()
    Q = ag.q(torch.tensor(S)).detach().numpy()
    err = 0.0
    for st in range(N - 1):
        qr = g ** (N - 2 - st); ql = g ** (N - st) if st > 0 else g ** (N - 1)
        err = max(err, abs(Q[st, 1] - qr), abs(Q[st, 0] - ql))
        assert Q[st, 1] > Q[st, 0], "optimal action must be 'right'"
    print(f"    max |Q - Q*| = {err:.4f}")
    assert err < 0.05


# 5 ---------------------------------------------------------------- schedules / determinism
def test_epsilon_schedule():
    c = Config(eps_start=1.0, eps_end=0.1, eps_decay_steps=100)
    assert abs(epsilon(c, 0) - 1.0) < 1e-9 and abs(epsilon(c, 50) - 0.55) < 1e-9
    assert abs(epsilon(c, 100) - 0.1) < 1e-9 and abs(epsilon(c, 10**6) - 0.1) < 1e-9


def test_first_10_steps_trace_deterministic():
    """Isolate the first 10 steps (eps=0, fixed seed): greedy action == argmax Q at every step, rewards are 1.0,
    and two runs give identical trajectories (all randomness removed)."""
    def run():
        torch.manual_seed(0)
        ag = DQNAgent(lambda: MLPQ(4, 2), 2, 0.95, 1e-3, "cpu", seed=0)
        env = gym.make("CartPole-v1"); o, _ = env.reset(seed=0); rows = []
        for t in range(10):
            q = ag.q(torch.tensor(o, dtype=torch.float32).unsqueeze(0))[0].detach().numpy()
            a, _ = ag.act(o.astype(np.float32), 0.0)
            assert a == int(q.argmax())
            o2, r, te, tr, _ = env.step(a)
            rows.append((t, [round(float(v), 3) for v in o], [round(float(v), 3) for v in q], a, r, te))
            o = o2
        return rows
    r1, r2 = run(), run()
    print("    t | state | Q(left,right) | action | reward | terminated")
    for row in r1:
        print("   ", *row)
    assert r1 == r2 and all(x[4] == 1.0 for x in r1)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print(f"PASS {t.__name__}")
        except Exception:
            failed += 1; print(f"FAIL {t.__name__}"); traceback.print_exc()
    print(f"\n{len(tests)-failed}/{len(tests)} tests passed"); sys.exit(1 if failed else 0)
