"""Layer 2 pilot: a ring world that agents shape by use. Mechanics follow BRIEF.md sections 2-4.

Angles are in degrees. Four actions with centres 0, 90, 180, 270; action a is accepted when the
item's angle is within 60 degrees of its centre. Overlap zones (two actions accepted) are
[30,60], [120,150], [210,240], [300,330]. Boundaries sit every 30 degrees from 30.
"""
import numpy as np
from numba import njit

CENTRES = np.array([0.0, 90.0, 180.0, 270.0])
HALF_ARC = 60.0
BOUNDARIES = np.array([30.0, 60.0, 120.0, 150.0, 210.0, 240.0, 300.0, 330.0])
OBS_DIM = 16
BODY_HID = 32
WEAR, EYE = 0, 1


# ---------------------------------------------------------------- geometry (numpy, for measures)

def wrapdiff(a, b):
    """Signed difference a - b on the circle, in (-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


def accepted_matrix(theta):
    """(n, 4) boolean: which actions the world accepts for each angle."""
    return np.abs(wrapdiff(theta[:, None], CENTRES[None, :])) <= HALF_ARC


def in_overlap(theta):
    return accepted_matrix(theta).sum(axis=1) >= 2


def is_edge(theta, window=10.0):
    d = np.abs(wrapdiff(theta[:, None], BOUNDARIES[None, :])).min(axis=1)
    return d <= window


# ---------------------------------------------------------------- bodies and agents

def make_body(rng, gamma=1.5):
    """A fixed random map from (cos, sin, n1, n2) to a 16-d observation."""
    return dict(Win=rng.normal(0, 1, (BODY_HID, 4)), bin=rng.normal(0, 0.5, BODY_HID),
                Wout=rng.normal(0, 1, (OBS_DIM, BODY_HID)) / np.sqrt(BODY_HID), gamma=gamma)


def observe(body, theta, nuis, rng=None, noise=0.05):
    r = np.deg2rad(theta)
    x = np.stack([np.cos(r), np.sin(r), nuis[:, 0], nuis[:, 1]], axis=1)
    h = np.tanh(body["gamma"] * x @ body["Win"].T + body["bin"])
    o = h @ body["Wout"].T
    if rng is not None and noise > 0:
        o = o + rng.normal(0, noise, o.shape)
    return o


def make_agents(rng, L, H):
    return dict(W1=rng.normal(0, 1, (L, H, OBS_DIM)) / np.sqrt(OBS_DIM), b1=np.zeros((L, H)),
                W2=rng.normal(0, 1, (L, 4, H)) / np.sqrt(H), b2=np.zeros((L, 4)))


def copy_agents(ag, idx=None):
    if idx is None:
        return {k: v.copy() for k, v in ag.items()}
    return {k: v[idx:idx + 1].copy() for k, v in ag.items()}


def predict(ag, i, obs):
    """(n, 4) predicted acceptance probabilities of agent i."""
    a1 = np.tanh(obs @ ag["W1"][i].T + ag["b1"][i])
    z = a1 @ ag["W2"][i].T + ag["b2"][i]
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------- the simulation core (numba)

@njit(cache=True)
def _obs1(theta_deg, n1, n2, Win, bin_, Wout, gamma, noise, out):
    r = theta_deg * np.pi / 180.0
    x0, x1 = np.cos(r), np.sin(r)
    H = Win.shape[0]
    for j in range(out.shape[0]):
        out[j] = 0.0
    for k in range(H):
        h = np.tanh(gamma * (Win[k, 0] * x0 + Win[k, 1] * x1 + Win[k, 2] * n1 + Win[k, 3] * n2) + bin_[k])
        for j in range(out.shape[0]):
            out[j] += Wout[j, k] * h
    if noise > 0:
        for j in range(out.shape[0]):
            out[j] += noise * np.random.normal()


@njit(cache=True)
def _forward(W1, b1, W2, b2, i, o, a1, p):
    H = W1.shape[1]
    for k in range(H):
        s = b1[i, k]
        for j in range(o.shape[0]):
            s += W1[i, k, j] * o[j]
        a1[k] = np.tanh(s)
    for a in range(4):
        s = b2[i, a]
        for k in range(H):
            s += W2[i, a, k] * a1[k]
        p[a] = 1.0 / (1.0 + np.exp(-s))


@njit(cache=True)
def _accepts(theta, a):
    c = a * 90.0
    d = (theta - c + 180.0) % 360.0 - 180.0
    return abs(d) <= 60.0


@njit(cache=True)
def run_steps(theta, home, nuis, Win, bin_, Wout, gamma, noise,
              LW1, Lb1, LW2, Lb2, learn, FW1, Fb1, FW2, Fb2,
              eta, rule, eye_step, delta, drift_noise, eps, lr, steps, acc_out):
    """Advance the world `steps` steps. Every agent acts once per step on a random item.

    L* arrays: learners (updated online from the refusal bit when learn is True).
    F* arrays: frozen agents (act and shape, never learn). Either set may be empty.
    acc_out[t] receives learner 0's accepted bit (with exploration), or -1 if there is no learner.
    """
    N = theta.shape[0]
    nL = LW1.shape[0]
    nF = FW1.shape[0]
    o = np.empty(Wout.shape[0])
    oh = np.empty(Wout.shape[0])
    HL = LW1.shape[1] if nL > 0 else 1
    HF = FW1.shape[1] if nF > 0 else 1
    a1L = np.empty(HL)
    a1F = np.empty(HF)
    p = np.empty(4)
    ph = np.empty(4)
    for t in range(steps):
        for ag in range(nL + nF):
            frozen = ag >= nL
            i = ag - nL if frozen else ag
            it = np.random.randint(0, N)
            _obs1(theta[it], nuis[it, 0], nuis[it, 1], Win, bin_, Wout, gamma, noise, o)
            if frozen:
                _forward(FW1, Fb1, FW2, Fb2, i, o, a1F, p)
            else:
                _forward(LW1, Lb1, LW2, Lb2, i, o, a1L, p)
            # two offers per turn (D1): the item affords two distinct random actions
            o1 = np.random.randint(0, 4)
            o2 = np.random.randint(0, 3)
            if o2 >= o1:
                o2 += 1
            if np.random.random() < eps:
                a = o1 if np.random.random() < 0.5 else o2
            else:
                a = o1 if p[o1] >= p[o2] else o2
            y = 1.0 if _accepts(theta[it], a) else 0.0
            if (not frozen) and i == 0:
                acc_out[t] = y
            if (not frozen) and learn:
                g = p[a] - y
                for k in range(HL):
                    w2 = LW2[i, a, k]
                    LW2[i, a, k] -= lr * g * a1L[k]
                    dz = g * w2 * (1.0 - a1L[k] * a1L[k])
                    Lb1[i, k] -= lr * dz
                    for j in range(o.shape[0]):
                        LW1[i, k, j] -= lr * dz * o[j]
                Lb2[i, a] -= lr * g
            if y > 0 and eta > 0:
                if rule == 0:  # wear: move a fraction eta toward the action's centre
                    d = (a * 90.0 - theta[it] + 180.0) % 360.0 - 180.0
                    theta[it] = (theta[it] + eta * d) % 360.0
                else:  # shaping by eye: step toward higher own confidence in action a
                    hstep = 1.0
                    _obs1(theta[it] + hstep, nuis[it, 0], nuis[it, 1], Win, bin_, Wout, gamma, 0.0, oh)
                    if frozen:
                        _forward(FW1, Fb1, FW2, Fb2, i, oh, a1F, ph)
                    else:
                        _forward(LW1, Lb1, LW2, Lb2, i, oh, a1L, ph)
                    up = ph[a]
                    _obs1(theta[it] - hstep, nuis[it, 0], nuis[it, 1], Win, bin_, Wout, gamma, 0.0, oh)
                    if frozen:
                        _forward(FW1, Fb1, FW2, Fb2, i, oh, a1F, ph)
                    else:
                        _forward(LW1, Lb1, LW2, Lb2, i, oh, a1L, ph)
                    dn = ph[a]
                    if up > dn:
                        theta[it] = (theta[it] + eye_step) % 360.0
                    elif dn > up:
                        theta[it] = (theta[it] - eye_step) % 360.0
        if delta > 0 or drift_noise > 0:
            for it in range(N):
                d = (home[it] - theta[it] + 180.0) % 360.0 - 180.0
                theta[it] = (theta[it] + delta * d + drift_noise * np.random.normal()) % 360.0


# ---------------------------------------------------------------- Python-side wrappers

EMPTY = None


def empty_agents():
    return dict(W1=np.zeros((0, 1, OBS_DIM)), b1=np.zeros((0, 1)), W2=np.zeros((0, 4, 1)), b2=np.zeros((0, 4)))


class World:
    def __init__(self, rng, n_items=600, gamma=1.5, noise=0.05, delta=0.0005, drift_noise=0.05):
        self.home = rng.uniform(0, 360, n_items)
        self.theta = self.home.copy()
        self.nuis = rng.normal(0, 1, (n_items, 2))
        self.body = make_body(rng, gamma)
        self.noise, self.delta, self.drift_noise = noise, delta, drift_noise

    def copy(self):
        w = object.__new__(World)
        w.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        return w

    def snap_home(self):
        self.theta = self.home.copy()

    def built_share(self):
        ov = in_overlap(self.home)
        return float((~in_overlap(self.theta[ov])).mean())

    def edge_share(self):
        return float(is_edge(self.theta).mean())


def step(world, learners, frozen, steps, eta, rule=WEAR, learn=True, eps=0.05, lr=0.05, eye_step=3.0,
         seed=0, drift=True):
    np.random.seed(seed)  # seeds numba's generator too
    _seed_numba(seed)
    acc = -np.ones(steps)
    L = learners if learners is not None else empty_agents()
    F = frozen if frozen is not None else empty_agents()
    run_steps(world.theta, world.home, world.nuis, world.body["Win"], world.body["bin"], world.body["Wout"],
              world.body["gamma"], world.noise, L["W1"], L["b1"], L["W2"], L["b2"], learn,
              F["W1"], F["b1"], F["W2"], F["b2"], eta, rule, eye_step,
              world.delta if drift else 0.0, world.drift_noise if drift else 0.0, eps, lr, steps, acc)
    return acc


@njit(cache=True)
def _seed_numba(s):
    np.random.seed(s)


# ---------------------------------------------------------------- measures

def random_offers(rng, n):
    o1 = rng.integers(0, 4, n)
    o2 = rng.integers(0, 3, n)
    o2 = o2 + (o2 >= o1)
    return o1, o2


def greedy_acceptance(world, ag, i, rng, n=400):
    """Competence (D1): of held-out offers with at least one accepted action, share where the pick is accepted."""
    idx = rng.integers(0, len(world.theta), n)
    o = observe(world.body, world.theta[idx], world.nuis[idx], rng, world.noise)
    p = predict(ag, i, o)
    o1, o2 = random_offers(rng, n)
    r = np.arange(n)
    pick = np.where(p[r, o1] >= p[r, o2], o1, o2)
    acc = accepted_matrix(world.theta[idx])
    possible = acc[r, o1] | acc[r, o2]
    return float(acc[r, pick][possible].mean())


def natural_test_set(rng, n=1000):
    return dict(theta=rng.uniform(0, 360, n), nuis=rng.normal(0, 1, (n, 2)))


def boundary_action(theta):
    """For each angle, the action whose arc edge is nearest (D1)."""
    edges = np.array([[(c - HALF_ARC) % 360, (c + HALF_ARC) % 360] for c in CENTRES])  # (4, 2)
    d = np.abs(wrapdiff(theta[:, None, None], edges[None, :, :])).min(axis=2)  # (n, 4)
    return d.argmin(axis=1)


def edge_scores(body, noise, ag, i, test, rng, window=10.0):
    """Edge accuracy and calibration gap (D1): the agent's prediction for the boundary action on edge items."""
    th, nu = test["theta"], test["nuis"]
    e = is_edge(th, window)
    o = observe(body, th[e], nu[e], rng, noise)
    p = predict(ag, i, o)
    ba = boundary_action(th[e])
    r = np.arange(e.sum())
    pb = p[r, ba]
    truth = accepted_matrix(th[e])[r, ba]
    correct = (pb > 0.5) == truth
    conf = np.maximum(pb, 1 - pb)
    return dict(edge_acc=float(correct.mean()), calib_gap=float(conf.mean() - correct.mean()),
                n_edge=int(e.sum()))


def home_log_loss(world, ag, i, rng):
    o = observe(world.body, world.theta, world.nuis, rng, world.noise)
    p = np.clip(predict(ag, i, o), 1e-6, 1 - 1e-6)
    y = accepted_matrix(world.theta)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
