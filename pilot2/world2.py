"""Layer 2 pilot 2: the version 3 world (BRIEF.md sections 2-4).

Items have an angle alpha (degrees) and two appearance coordinates (u1, u2), plus a kind flag
(0 = A, the main kind; 1 = B, E8's second task, never shaped). Four actions with arcs of +-60
degrees around 0, 90, 180, 270. Each turn: one item, one offered action; the agent acts if its
predicted acceptance exceeds 2/3 (or with probability eps), else skips. Payoffs +1 / 0 / -2.
"""
import numpy as np
from numba import njit

CENTRES = np.array([0.0, 90.0, 180.0, 270.0])
HALF_ARC = 60.0
BOUNDARIES = np.array([30.0, 60.0, 120.0, 150.0, 210.0, 240.0, 300.0, 330.0])
OBS_DIM, BODY_HID, IN_DIM = 16, 32, 5
ACT_THRESHOLD = 2.0 / 3.0
NONE, WEAR, EYE = 0, 1, 2
DEG_PER_UNIT = 20.0  # angle step per unit of appearance step at kappa = 1


# ------------------------------------------------------------------ geometry

def wrapdiff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def accepted_matrix(alpha):
    return np.abs(wrapdiff(alpha[:, None], CENTRES[None, :])) <= HALF_ARC


def in_overlap(alpha):
    return accepted_matrix(alpha).sum(axis=1) >= 2


def is_edge(alpha, window=10.0):
    return np.abs(wrapdiff(alpha[:, None], BOUNDARIES[None, :])).min(axis=1) <= window


def boundary_action(alpha):
    edges = np.array([[(c - HALF_ARC) % 360, (c + HALF_ARC) % 360] for c in CENTRES])
    d = np.abs(wrapdiff(alpha[:, None, None], edges[None, :, :])).min(axis=2)
    return d.argmin(axis=1)


def nearest_action(alpha):
    return np.abs(wrapdiff(alpha[:, None], CENTRES[None, :])).argmin(axis=1)


# ------------------------------------------------------------------ bodies, agents

def make_bodies(rng, n, gamma=1.5):
    return dict(Win=rng.normal(0, 1, (n, BODY_HID, IN_DIM)), bin=rng.normal(0, 0.5, (n, BODY_HID)),
                Wout=rng.normal(0, 1, (n, OBS_DIM, BODY_HID)) / np.sqrt(BODY_HID), gamma=gamma)


def observe(bodies, b, alpha, u, kind, rng=None, noise=0.05):
    r = np.deg2rad(alpha)
    x = np.stack([np.cos(r), np.sin(r), u[:, 0], u[:, 1], kind.astype(float)], axis=1)
    h = np.tanh(bodies["gamma"] * x @ bodies["Win"][b].T + bodies["bin"][b])
    o = h @ bodies["Wout"][b].T
    if rng is not None and noise > 0:
        o = o + rng.normal(0, noise, o.shape)
    return o


def make_agents(rng, L, H, body=0):
    return dict(W1=rng.normal(0, 1, (L, H, OBS_DIM)) / np.sqrt(OBS_DIM), b1=np.zeros((L, H)),
                W2=rng.normal(0, 1, (L, 4, H)) / np.sqrt(H), b2=np.zeros((L, 4)),
                body=np.full(L, body, dtype=np.int64))


def empty_agents():
    return dict(W1=np.zeros((0, 1, OBS_DIM)), b1=np.zeros((0, 1)), W2=np.zeros((0, 4, 1)), b2=np.zeros((0, 4)),
                body=np.zeros(0, dtype=np.int64))


def copy_agents(ag):
    return {k: v.copy() for k, v in ag.items()}


def predict(ag, i, obs):
    a1 = np.tanh(obs @ ag["W1"][i].T + ag["b1"][i])
    return 1.0 / (1.0 + np.exp(-(a1 @ ag["W2"][i].T + ag["b2"][i])))


# ------------------------------------------------------------------ numba core

@njit(cache=True)
def _obs1(alpha, u1, u2, kind, Win, bin_, Wout, b, gamma, noise, out):
    r = alpha * np.pi / 180.0
    x0, x1 = np.cos(r), np.sin(r)
    for j in range(out.shape[0]):
        out[j] = 0.0
    for k in range(Win.shape[1]):
        h = np.tanh(gamma * (Win[b, k, 0] * x0 + Win[b, k, 1] * x1 + Win[b, k, 2] * u1 + Win[b, k, 3] * u2
                             + Win[b, k, 4] * kind) + bin_[b, k])
        for j in range(out.shape[0]):
            out[j] += Wout[b, j, k] * h
    if noise > 0:
        for j in range(out.shape[0]):
            out[j] += noise * np.random.normal()


@njit(cache=True)
def _forward(W1, b1, W2, b2, i, o, a1, p):
    for k in range(W1.shape[1]):
        s = b1[i, k]
        for j in range(o.shape[0]):
            s += W1[i, k, j] * o[j]
        a1[k] = np.tanh(s)
    for a in range(4):
        s = b2[i, a]
        for k in range(W1.shape[1]):
            s += W2[i, a, k] * a1[k]
        p[a] = 1.0 / (1.0 + np.exp(-s))


@njit(cache=True)
def _accepts(alpha, a):
    d = (alpha - a * 90.0 + 180.0) % 360.0 - 180.0
    return abs(d) <= 60.0


@njit(cache=True)
def _conf(alpha, u1, u2, kind, Win, bin_, Wout, b, gamma, W1, b1, W2, b2, i, a, o, a1, p):
    _obs1(alpha, u1, u2, kind, Win, bin_, Wout, b, gamma, 0.0, o)
    _forward(W1, b1, W2, b2, i, o, a1, p)
    return p[a]


@njit(cache=True)
def run_steps(alpha, u, home_a, home_u, kind,
              Win, bin_, Wout, gamma, noise,
              LW1, Lb1, LW2, Lb2, Lbody, learn, learner_shapes,
              FW1, Fb1, FW2, Fb2, Fbody,
              idxL_A, idxL_B, pB, idxF,
              rule, eta, s_u, kappa, eps_r,
              delta, drift_a, drift_u, eps, lr, steps,
              pay_out, item_out, act_out, offer_out):
    """Advance `steps` steps. Learners pick items from idxL_A (or idxL_B with probability pB);
    frozen agents pick from idxF. Learner 0's per-step payoff, item, act flag and offer are recorded."""
    nL = LW1.shape[0]
    nF = FW1.shape[0]
    o = np.empty(Wout.shape[1])
    oh = np.empty(Wout.shape[1])
    HL = LW1.shape[1] if nL > 0 else 1
    HF = FW1.shape[1] if nF > 0 else 1
    a1L = np.empty(HL)
    a1F = np.empty(HF)
    p = np.empty(4)
    ph = np.empty(4)
    N = alpha.shape[0]
    for t in range(steps):
        for ag in range(nL + nF):
            frozen = ag >= nL
            i = ag - nL if frozen else ag
            if frozen:
                if idxF.shape[0] == 0:
                    continue
                it = idxF[np.random.randint(0, idxF.shape[0])]
                b = Fbody[i]
            else:
                if idxL_B.shape[0] > 0 and np.random.random() < pB:
                    it = idxL_B[np.random.randint(0, idxL_B.shape[0])]
                else:
                    it = idxL_A[np.random.randint(0, idxL_A.shape[0])]
                b = Lbody[i]
            a = np.random.randint(0, 4)
            _obs1(alpha[it], u[it, 0], u[it, 1], kind[it], Win, bin_, Wout, b, gamma, noise, o)
            if frozen:
                _forward(FW1, Fb1, FW2, Fb2, i, o, a1F, p)
            else:
                _forward(LW1, Lb1, LW2, Lb2, i, o, a1L, p)
            act = p[a] > 2.0 / 3.0 or np.random.random() < eps
            pay = 0.0
            y = 0.0
            if act:
                if _accepts(alpha[it], a) and not (eps_r > 0 and np.random.random() < eps_r):
                    y = 1.0
                pay = 1.0 if y > 0 else -2.0
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
                shapes = frozen or learner_shapes
                if y > 0 and shapes and kind[it] == 0 and rule != 0:
                    if rule == 1:  # wear
                        d = (a * 90.0 - alpha[it] + 180.0) % 360.0 - 180.0
                        alpha[it] = (alpha[it] + eta * d) % 360.0
                    else:  # shaping by eye: sign step in every coordinate toward higher own p[a]
                        if frozen:
                            W1, b1, W2, b2 = FW1, Fb1, FW2, Fb2
                            a1 = a1F
                        else:
                            W1, b1, W2, b2 = LW1, Lb1, LW2, Lb2
                            a1 = a1L
                        ha, hu = 1.0, 0.05
                        cp = _conf(alpha[it] + ha, u[it, 0], u[it, 1], kind[it], Win, bin_, Wout, b, gamma, W1, b1, W2, b2, i, a, oh, a1, ph)
                        cm = _conf(alpha[it] - ha, u[it, 0], u[it, 1], kind[it], Win, bin_, Wout, b, gamma, W1, b1, W2, b2, i, a, oh, a1, ph)
                        da = 1.0 if cp > cm else (-1.0 if cm > cp else 0.0)
                        cp = _conf(alpha[it], u[it, 0] + hu, u[it, 1], kind[it], Win, bin_, Wout, b, gamma, W1, b1, W2, b2, i, a, oh, a1, ph)
                        cm = _conf(alpha[it], u[it, 0] - hu, u[it, 1], kind[it], Win, bin_, Wout, b, gamma, W1, b1, W2, b2, i, a, oh, a1, ph)
                        d1 = 1.0 if cp > cm else (-1.0 if cm > cp else 0.0)
                        cp = _conf(alpha[it], u[it, 0], u[it, 1] + hu, kind[it], Win, bin_, Wout, b, gamma, W1, b1, W2, b2, i, a, oh, a1, ph)
                        cm = _conf(alpha[it], u[it, 0], u[it, 1] - hu, kind[it], Win, bin_, Wout, b, gamma, W1, b1, W2, b2, i, a, oh, a1, ph)
                        d2 = 1.0 if cp > cm else (-1.0 if cm > cp else 0.0)
                        alpha[it] = (alpha[it] + da * kappa * s_u * 20.0) % 360.0
                        u[it, 0] += d1 * s_u
                        u[it, 1] += d2 * s_u
            if (not frozen) and i == 0:
                pay_out[t] = pay
                item_out[t] = it
                act_out[t] = 1 if act else 0
                offer_out[t] = a
        if delta > 0 or drift_a > 0:
            for it in range(N):
                d = (home_a[it] - alpha[it] + 180.0) % 360.0 - 180.0
                alpha[it] = (alpha[it] + delta * d + drift_a * np.random.normal()) % 360.0
                for c in range(2):
                    u[it, c] = u[it, c] + delta * (home_u[it, c] - u[it, c]) + drift_u * np.random.normal()


@njit(cache=True)
def _seed(s):
    np.random.seed(s)


# ------------------------------------------------------------------ Python side

class World:
    def __init__(self, rng, n_items=600, delta=0.0005, drift_a=0.05, drift_u=0.002, noise=0.05):
        self.home_a = rng.uniform(0, 360, n_items)
        self.home_u = rng.normal(0, 1, (n_items, 2))
        self.alpha = self.home_a.copy()
        self.u = self.home_u.copy()
        self.kind = np.zeros(n_items, dtype=np.int64)
        self.delta, self.drift_a, self.drift_u, self.noise = delta, drift_a, drift_u, noise

    def copy(self):
        w = object.__new__(World)
        w.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}
        return w

    def add_kind_b(self, rng, n):
        self.home_a = np.concatenate([self.home_a, rng.uniform(0, 360, n)])
        hu = rng.normal(0, 1, (n, 2))
        self.home_u = np.concatenate([self.home_u, hu])
        self.alpha = np.concatenate([self.alpha, self.home_a[-n:]])
        self.u = np.concatenate([self.u, hu.copy()])
        self.kind = np.concatenate([self.kind, np.ones(n, dtype=np.int64)])

    def snap_home(self):
        self.alpha = self.home_a.copy()
        self.u = self.home_u.copy()

    def idx(self, kind=0, exclude_edges=False):
        m = self.kind == kind
        if exclude_edges:
            m &= ~is_edge(self.alpha)
        return np.nonzero(m)[0].astype(np.int64)

    def built_share(self):
        a = self.kind == 0
        ov = in_overlap(self.home_a) & a
        return float((~in_overlap(self.alpha[ov])).mean())

    def edge_share(self):
        a = self.kind == 0
        return float(is_edge(self.alpha[a]).mean())


def step(world, bodies, learners, frozen, steps, seed, rule=NONE, eta=0.0, s_u=0.0, kappa=1.0, eps_r=0.0,
         learn=True, learner_shapes=False, eps=0.05, lr=0.05, pB=0.0, filtered=False, chunk=500):
    """Run `steps` steps in chunks (the filtered sampler is refreshed each chunk). Returns learner 0's records."""
    L = learners if learners is not None else empty_agents()
    F = frozen if frozen is not None else empty_agents()
    recs = dict(pay=[], item=[], act=[], offer=[])
    done = 0
    k = 0
    while done < steps:
        n = min(chunk, steps - done)
        _seed(seed * 1000 + k)
        idxA = world.idx(0, exclude_edges=filtered)
        idxB = world.idx(1)
        idxF = world.idx(0)
        pay = np.zeros(n); item = -np.ones(n, dtype=np.int64); act = np.zeros(n, dtype=np.int64); offer = np.zeros(n, dtype=np.int64)
        run_steps(world.alpha, world.u, world.home_a, world.home_u, world.kind,
                  bodies["Win"], bodies["bin"], bodies["Wout"], bodies["gamma"], world.noise,
                  L["W1"], L["b1"], L["W2"], L["b2"], L["body"], learn, learner_shapes,
                  F["W1"], F["b1"], F["W2"], F["b2"], F["body"],
                  idxA, idxB, pB, idxF, rule, eta, s_u, kappa, eps_r,
                  world.delta, world.drift_a, world.drift_u, eps, lr, n, pay, item, act, offer)
        for key, v in (("pay", pay), ("item", item), ("act", act), ("offer", offer)):
            recs[key].append(v)
        done += n
        k += 1
    return {key: np.concatenate(v) for key, v in recs.items()}


# ------------------------------------------------------------------ measures

def competence(world, bodies, ag, i, rng, kind=0, n=600, eps_r=0.0):
    idx = world.idx(kind)
    it = idx[rng.integers(0, len(idx), n)]
    o = observe(bodies, ag["body"][i], world.alpha[it], world.u[it], world.kind[it], rng, world.noise)
    p = predict(ag, i, o)
    a = rng.integers(0, 4, n)
    q = (1 - eps_r) * accepted_matrix(world.alpha[it])[np.arange(n), a]
    act = p[np.arange(n), a] > ACT_THRESHOLD
    opt = np.maximum(0, 3 * q - 2).sum()
    return float((act * (3 * q - 2)).sum() / opt) if opt > 0 else float("nan")


def natural_test_set(rng, n=1500):
    return dict(alpha=rng.uniform(0, 360, n), u=rng.normal(0, 1, (n, 2)), kind=np.zeros(n, dtype=np.int64))


def edge_scores(bodies, ag, i, test, rng, noise=0.05):
    th = test["alpha"]
    e = is_edge(th)
    o = observe(bodies, ag["body"][i], th[e], test["u"][e], test["kind"][e], rng, noise)
    return _edge_from_p(predict(ag, i, o), th[e])


def edge_scores_ensemble(bodies, ags, test, rng, noise=0.05):
    th = test["alpha"]
    e = is_edge(th)
    ps = []
    for ag in ags:
        o = observe(bodies, ag["body"][0], th[e], test["u"][e], test["kind"][e], rng, noise)
        ps.append(predict(ag, 0, o))
    return _edge_from_p(np.mean(ps, axis=0), th[e])


def _edge_from_p(p, th_e):
    ba = boundary_action(th_e)
    r = np.arange(len(th_e))
    pb = p[r, ba]
    truth = accepted_matrix(th_e)[r, ba]
    correct = (pb > 0.5) == truth
    act = pb > ACT_THRESHOLD
    pay = np.where(act, np.where(truth, 1.0, -2.0), 0.0)
    opt = truth.astype(float)
    nearest = nearest_action(th_e) == ba
    heur = np.where(nearest, np.where(truth, 1.0, -2.0), 0.0)
    return dict(edge_acc=float(correct.mean()), calib_gap=float(np.maximum(pb, 1 - pb).mean() - correct.mean()),
                decisive_pay=float(pay.mean()), decisive_ratio=float(pay.sum() / opt.sum()),
                heuristic_ratio=float(heur.sum() / opt.sum()), n_edge=int(len(th_e)))


def log_loss(world, bodies, ag, i, rng, eps_r=0.0, kind=0):
    m = world.kind == kind
    o = observe(bodies, ag["body"][i], world.alpha[m], world.u[m], world.kind[m], rng, world.noise)
    p = np.clip(predict(ag, i, o), 1e-6, 1 - 1e-6)
    q = (1 - eps_r) * accepted_matrix(world.alpha[m])
    return float(-(q * np.log(p) + (1 - q) * np.log(1 - p)).mean())


def forestry_entropy(bodies, b, ref, state, bins=10):
    """Entropy (bits) of `state`'s kind-A items under body b, on a grid fitted to `ref` (the natural pool)."""
    def obs(w):
        m = w.kind == 0
        return observe(bodies, b, w.alpha[m], w.u[m], w.kind[m], None, 0.0)
    R = obs(ref)
    mu = R.mean(axis=0)
    _, _, Vt = np.linalg.svd(R - mu, full_matrices=False)
    P = Vt[:2].T
    zr = (R - mu) @ P
    edges = [np.linspace(np.percentile(zr[:, c], 1), np.percentile(zr[:, c], 99), bins + 1) for c in range(2)]

    def H(z):
        zc = np.stack([np.clip(z[:, c], edges[c][0], edges[c][-1] - 1e-9) for c in range(2)], axis=1)
        h, _, _ = np.histogram2d(zc[:, 0], zc[:, 1], bins=edges)
        pr = h.ravel() / h.sum()
        pr = pr[pr > 0]
        return float(-(pr * np.log2(pr)).sum())
    return H(zr), H((obs(state) - mu) @ P)
