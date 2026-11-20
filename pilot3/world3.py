"""Layer 2 pilot 3: the version 4 world (BRIEF.md sections 2-4).

Changes from pilot 2's world2.py:
- structured bodies: type 1 reads (cos a, sin a, lam*u1, kind), type 2 reads (cos a, sin a, lam*u2, kind);
- exploration: act if p > 2/3; within +-m of 2/3 act with prob eps_m; global floor eps_min; neutral init;
- random refusal applies to learners only;
- ensembles trained by Thompson sampling with a bootstrap mask, optional diverse widths and randomized priors;
- per-step log loss of the offered action's prediction (for code length).
"""
import numpy as np
from numba import njit

CENTRES = np.array([0.0, 90.0, 180.0, 270.0])
HALF_ARC = 60.0
BOUNDARIES = np.array([30.0, 60.0, 120.0, 150.0, 210.0, 240.0, 300.0, 330.0])
OBS_DIM, BODY_HID, IN_DIM = 16, 32, 5
ACT = 2.0 / 3.0
NONE, WEAR, EYE = 0, 1, 2


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


# ------------------------------------------------------------------ bodies

def make_bodies(rng, types, lam, gamma=1.5):
    """One random map per entry of `types` (1 reads u1, 2 reads u2); the read appearance input is scaled by lam."""
    n = len(types)
    Win = rng.normal(0, 1, (n, BODY_HID, IN_DIM))
    for b, t in enumerate(types):
        Win[b, :, 3 if t == 1 else 2] = 0.0   # the unread appearance coordinate
        Win[b, :, 2 if t == 1 else 3] *= lam  # the read one, scaled
    return dict(Win=Win, bin=rng.normal(0, 0.5, (n, BODY_HID)),
                Wout=rng.normal(0, 1, (n, OBS_DIM, BODY_HID)) / np.sqrt(BODY_HID), gamma=gamma,
                types=np.array(types))


def observe(bodies, b, alpha, u, kind, rng=None, noise=0.05):
    r = np.deg2rad(alpha)
    x = np.stack([np.cos(r), np.sin(r), u[:, 0], u[:, 1], kind.astype(float)], axis=1)
    h = np.tanh(bodies["gamma"] * x @ bodies["Win"][b].T + bodies["bin"][b])
    o = h @ bodies["Wout"][b].T
    if rng is not None and noise > 0:
        o = o + rng.normal(0, noise, o.shape)
    return o


# ------------------------------------------------------------------ agents (single networks)

def make_agents(rng, L, H, body=0):
    return dict(W1=rng.normal(0, 1, (L, H, OBS_DIM)) / np.sqrt(OBS_DIM), b1=np.zeros((L, H)),
                W2=rng.normal(0, 1, (L, 4, H)) / np.sqrt(H), b2=np.zeros((L, 4)),
                body=np.full(L, body, dtype=np.int64))


def empty_agents():
    return dict(W1=np.zeros((0, 1, OBS_DIM)), b1=np.zeros((0, 1)), W2=np.zeros((0, 4, 1)), b2=np.zeros((0, 4)),
                body=np.zeros(0, dtype=np.int64))


def copy_agents(ag):
    return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ag.items()}


def hidden(ag, i, obs):
    return np.tanh(obs @ ag["W1"][i].T + ag["b1"][i])


def predict(ag, i, obs):
    return 1.0 / (1.0 + np.exp(-(hidden(ag, i, obs) @ ag["W2"][i].T + ag["b2"][i])))


# ------------------------------------------------------------------ ensembles

def make_ensemble(rng, K, widths, body=0, beta=0.0, prior_width=32):
    """K members, zero-padded to the largest width (padded units have zero weights and stay dead)."""
    Hm = max(widths)
    W1 = np.zeros((K, Hm, OBS_DIM)); W2 = np.zeros((K, 4, Hm))
    for k, h in enumerate(widths):
        W1[k, :h] = rng.normal(0, 1, (h, OBS_DIM)) / np.sqrt(OBS_DIM)
        W2[k, :, :h] = rng.normal(0, 1, (4, h)) / np.sqrt(h)
    return dict(W1=W1, b1=np.zeros((K, Hm)), W2=W2, b2=np.zeros((K, 4)),
                P1=rng.normal(0, 1, (K, prior_width, OBS_DIM)) / np.sqrt(OBS_DIM), Pb1=rng.normal(0, 0.5, (K, prior_width)),
                P2=rng.normal(0, 1, (K, 4, prior_width)) / np.sqrt(prior_width), Pb2=np.zeros((K, 4)),
                beta=float(beta), body=int(body), widths=np.array(widths))


def ensemble_member_p(ens, obs):
    """(K, n, 4) member predictions, including the fixed random prior."""
    out = []
    for k in range(ens["W1"].shape[0]):
        z = np.tanh(obs @ ens["W1"][k].T + ens["b1"][k]) @ ens["W2"][k].T + ens["b2"][k]
        if ens["beta"] > 0:
            z = z + ens["beta"] * (np.tanh(obs @ ens["P1"][k].T + ens["Pb1"][k]) @ ens["P2"][k].T + ens["Pb2"][k])
        out.append(1.0 / (1.0 + np.exp(-z)))
    return np.stack(out)


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
def _logits(W1, b1, W2, b2, i, o, a1, z):
    for k in range(W1.shape[1]):
        s = b1[i, k]
        for j in range(o.shape[0]):
            s += W1[i, k, j] * o[j]
        a1[k] = np.tanh(s)
    for a in range(4):
        s = b2[i, a]
        for k in range(W1.shape[1]):
            s += W2[i, a, k] * a1[k]
        z[a] = s


@njit(cache=True)
def _accepts(alpha, a):
    d = (alpha - a * 90.0 + 180.0) % 360.0 - 180.0
    return abs(d) <= 60.0


@njit(cache=True)
def _sgd(W1, b1, W2, b2, i, o, a1, a, g, lr):
    for k in range(W1.shape[1]):
        w2 = W2[i, a, k]
        if w2 == 0.0 and a1[k] == 0.0:
            continue  # a dead (padded) unit stays dead
        W2[i, a, k] -= lr * g * a1[k]
        dz = g * w2 * (1.0 - a1[k] * a1[k])
        b1[i, k] -= lr * dz
        for j in range(o.shape[0]):
            W1[i, k, j] -= lr * dz * o[j]
    b2[i, a] -= lr * g


@njit(cache=True)
def _decide(p, eps_min, eps_m, m):
    if p > 2.0 / 3.0:
        return True
    if abs(p - 2.0 / 3.0) < m and np.random.random() < eps_m:
        return True
    return np.random.random() < eps_min


@njit(cache=True)
def _shape(it, alpha, u, kind, a, rule, eta, s_u, kappa, Win, bin_, Wout, b, gamma, W1, b1, W2, b2, i, oh, a1, zh):
    if kind[it] != 0 or rule == 0:
        return
    if rule == 1:
        d = (a * 90.0 - alpha[it] + 180.0) % 360.0 - 180.0
        alpha[it] = (alpha[it] + eta * d) % 360.0
        return
    # shaping by eye: sign of the finite-difference gradient of own logit for action a, per coordinate
    steps = np.zeros(3)
    for c in range(3):
        h = 1.0 if c == 0 else 0.05
        vals = np.zeros(2)
        for sgn in range(2):
            dlt = h if sgn == 0 else -h
            aa, u1, u2 = alpha[it], u[it, 0], u[it, 1]
            if c == 0:
                aa += dlt
            elif c == 1:
                u1 += dlt
            else:
                u2 += dlt
            _obs1(aa, u1, u2, kind[it], Win, bin_, Wout, b, gamma, 0.0, oh)
            _logits(W1, b1, W2, b2, i, oh, a1, zh)
            vals[sgn] = zh[a]
        steps[c] = 1.0 if vals[0] > vals[1] else (-1.0 if vals[1] > vals[0] else 0.0)
    alpha[it] = (alpha[it] + steps[0] * kappa * s_u * 20.0) % 360.0
    u[it, 0] += steps[1] * s_u
    u[it, 1] += steps[2] * s_u


@njit(cache=True)
def _drift(alpha, u, home_a, home_u, delta, drift_a, drift_u):
    for it in range(alpha.shape[0]):
        d = (home_a[it] - alpha[it] + 180.0) % 360.0 - 180.0
        alpha[it] = (alpha[it] + delta * d + drift_a * np.random.normal()) % 360.0
        for c in range(2):
            u[it, c] = u[it, c] + delta * (home_u[it, c] - u[it, c]) + drift_u * np.random.normal()


@njit(cache=True)
def run_steps(alpha, u, home_a, home_u, kind, Win, bin_, Wout, gamma, noise,
              LW1, Lb1, LW2, Lb2, Lbody, learn, learner_shapes,
              FW1, Fb1, FW2, Fb2, Fbody, idxL_A, idxL_B, pB, idxF,
              rule, eta, s_u, kappa, eps_r, delta, drift_a, drift_u,
              eps_min, eps_m, m, lr, steps, pay_out, item_out, act_out, offer_out, ll_out):
    nL = LW1.shape[0]
    nF = FW1.shape[0]
    o = np.empty(Wout.shape[1]); oh = np.empty(Wout.shape[1])
    a1L = np.empty(LW1.shape[1] if nL > 0 else 1); a1F = np.empty(FW1.shape[1] if nF > 0 else 1)
    z = np.empty(4); zh = np.empty(4)
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
                _logits(FW1, Fb1, FW2, Fb2, i, o, a1F, z)
            else:
                _logits(LW1, Lb1, LW2, Lb2, i, o, a1L, z)
            p = 1.0 / (1.0 + np.exp(-z[a]))
            truth = _accepts(alpha[it], a)
            if (not frozen) and i == 0:
                pc = min(max(p, 1e-6), 1 - 1e-6)
                ll_out[t] = -np.log(pc) if truth else -np.log(1 - pc)
            act = _decide(p, eps_min, eps_m, m)
            pay = 0.0
            if act:
                y = 1.0 if truth else 0.0
                if y > 0 and (not frozen) and eps_r > 0 and np.random.random() < eps_r:
                    y = 0.0
                pay = 1.0 if y > 0 else -2.0
                if (not frozen) and learn:
                    _sgd(LW1, Lb1, LW2, Lb2, i, o, a1L, a, p - y, lr)
                if y > 0 and (frozen or learner_shapes):
                    if frozen:
                        _shape(it, alpha, u, kind, a, rule, eta, s_u, kappa, Win, bin_, Wout, b, gamma, FW1, Fb1, FW2, Fb2, i, oh, a1F, zh)
                    else:
                        _shape(it, alpha, u, kind, a, rule, eta, s_u, kappa, Win, bin_, Wout, b, gamma, LW1, Lb1, LW2, Lb2, i, oh, a1L, zh)
            if (not frozen) and i == 0:
                pay_out[t] = pay; item_out[t] = it; act_out[t] = 1 if act else 0; offer_out[t] = a
        if delta > 0 or drift_a > 0:
            _drift(alpha, u, home_a, home_u, delta, drift_a, drift_u)


@njit(cache=True)
def run_steps_ens(alpha, u, home_a, home_u, kind, Win, bin_, Wout, gamma, noise,
                  EW1, Eb1, EW2, Eb2, P1, Pb1, P2, Pb2, beta, ebody,
                  FW1, Fb1, FW2, Fb2, Fbody, idxL_A, idxF,
                  rule, eta, s_u, kappa, delta, drift_a, drift_u, eps_min, lr, steps, pay_out, act_out):
    """One ensemble learner acting by Thompson sampling, with frozen builders alongside."""
    K = EW1.shape[0]
    nF = FW1.shape[0]
    o = np.empty(Wout.shape[1]); oh = np.empty(Wout.shape[1])
    a1 = np.empty(EW1.shape[1]); ap = np.empty(P1.shape[1]); a1F = np.empty(FW1.shape[1] if nF > 0 else 1)
    z = np.empty(4); zp = np.empty(4); zh = np.empty(4)
    for t in range(steps):
        # builders
        for i in range(nF):
            it = idxF[np.random.randint(0, idxF.shape[0])]
            a = np.random.randint(0, 4)
            _obs1(alpha[it], u[it, 0], u[it, 1], kind[it], Win, bin_, Wout, Fbody[i], gamma, noise, o)
            _logits(FW1, Fb1, FW2, Fb2, i, o, a1F, z)
            p = 1.0 / (1.0 + np.exp(-z[a]))
            if _decide(p, eps_min, 0.0, 0.0) and _accepts(alpha[it], a):
                _shape(it, alpha, u, kind, a, rule, eta, s_u, kappa, Win, bin_, Wout, Fbody[i], gamma, FW1, Fb1, FW2, Fb2, i, oh, a1F, zh)
        # the ensemble learner
        it = idxL_A[np.random.randint(0, idxL_A.shape[0])]
        a = np.random.randint(0, 4)
        _obs1(alpha[it], u[it, 0], u[it, 1], kind[it], Win, bin_, Wout, ebody, gamma, noise, o)
        k = np.random.randint(0, K)
        _logits(EW1, Eb1, EW2, Eb2, k, o, a1, z)
        zz = z[a]
        if beta > 0:
            _logits(P1, Pb1, P2, Pb2, k, o, ap, zp)
            zz += beta * zp[a]
        p = 1.0 / (1.0 + np.exp(-zz))
        act = _decide(p, eps_min, 0.0, 0.0)
        pay = 0.0
        if act:
            y = 1.0 if _accepts(alpha[it], a) else 0.0
            pay = 1.0 if y > 0 else -2.0
            for mm in range(K):
                if np.random.random() < 0.5:
                    _logits(EW1, Eb1, EW2, Eb2, mm, o, a1, z)
                    zz = z[a]
                    if beta > 0:
                        _logits(P1, Pb1, P2, Pb2, mm, o, ap, zp)
                        zz += beta * zp[a]
                    _sgd(EW1, Eb1, EW2, Eb2, mm, o, a1, a, 1.0 / (1.0 + np.exp(-zz)) - y, lr)
        pay_out[t] = pay; act_out[t] = 1 if act else 0
        if delta > 0 or drift_a > 0:
            _drift(alpha, u, home_a, home_u, delta, drift_a, drift_u)


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

    def frozen(self):
        w = self.copy()
        w.delta = w.drift_a = w.drift_u = 0.0
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
        msk = self.kind == kind
        if exclude_edges:
            msk &= ~is_edge(self.alpha)
        return np.nonzero(msk)[0].astype(np.int64)

    def built_share(self):
        ov = in_overlap(self.home_a) & (self.kind == 0)
        return float((~in_overlap(self.alpha[ov])).mean())

    def edge_share(self):
        return float(is_edge(self.alpha[self.kind == 0]).mean())

    def u_move(self, c):
        a = self.kind == 0
        return float(np.abs(self.u[a, c] - self.home_u[a, c]).mean())


EXPLORE = dict(eps_min=0.02, eps_m=0.6, m=0.2)  # tuned on competence (PILOT3_LOG.md, step 2)


def step(world, bodies, learners, frozen, steps, seed, rule=NONE, eta=0.0, s_u=0.0, kappa=1.0, eps_r=0.0,
         learn=True, learner_shapes=False, lr=0.05, pB=0.0, filtered=False, chunk=500, explore=None):
    ex = explore or EXPLORE
    L = learners if learners is not None else empty_agents()
    F = frozen if frozen is not None else empty_agents()
    recs = {k: [] for k in ("pay", "item", "act", "offer", "ll")}
    done = k = 0
    while done < steps:
        n = min(chunk, steps - done)
        _seed(seed * 1000 + k)
        idxA, idxB, idxF = world.idx(0, exclude_edges=filtered), world.idx(1), world.idx(0)
        pay = np.zeros(n); item = -np.ones(n, dtype=np.int64); act = np.zeros(n, dtype=np.int64)
        offer = np.zeros(n, dtype=np.int64); ll = np.zeros(n)
        run_steps(world.alpha, world.u, world.home_a, world.home_u, world.kind,
                  bodies["Win"], bodies["bin"], bodies["Wout"], bodies["gamma"], world.noise,
                  L["W1"], L["b1"], L["W2"], L["b2"], L["body"], learn, learner_shapes,
                  F["W1"], F["b1"], F["W2"], F["b2"], F["body"], idxA, idxB, pB, idxF,
                  rule, eta, s_u, kappa, eps_r, world.delta, world.drift_a, world.drift_u,
                  ex["eps_min"], ex["eps_m"], ex["m"], lr, n, pay, item, act, offer, ll)
        for key, v in (("pay", pay), ("item", item), ("act", act), ("offer", offer), ("ll", ll)):
            recs[key].append(v)
        done += n; k += 1
    return {key: np.concatenate(v) for key, v in recs.items()}


def step_ens(world, bodies, ens, frozen, steps, seed, rule=NONE, eta=0.0, s_u=0.0, kappa=1.0, lr=0.05,
             eps_min=0.02, chunk=500):
    F = frozen if frozen is not None else empty_agents()
    acts = []
    done = k = 0
    while done < steps:
        n = min(chunk, steps - done)
        _seed(seed * 1000 + k)
        pay = np.zeros(n); act = np.zeros(n, dtype=np.int64)
        run_steps_ens(world.alpha, world.u, world.home_a, world.home_u, world.kind,
                      bodies["Win"], bodies["bin"], bodies["Wout"], bodies["gamma"], world.noise,
                      ens["W1"], ens["b1"], ens["W2"], ens["b2"], ens["P1"], ens["Pb1"], ens["P2"], ens["Pb2"],
                      ens["beta"], ens["body"], F["W1"], F["b1"], F["W2"], F["b2"], F["body"],
                      world.idx(0), world.idx(0), rule, eta, s_u, kappa,
                      world.delta, world.drift_a, world.drift_u, eps_min, lr, n, pay, act)
        acts.append(act)
        done += n; k += 1
    return np.concatenate(acts)


# ------------------------------------------------------------------ measures

def competence(world, bodies, ag, i, rng, kind=0, n=600, eps_r=0.0):
    idx = world.idx(kind)
    it = idx[rng.integers(0, len(idx), n)]
    o = observe(bodies, ag["body"][i], world.alpha[it], world.u[it], world.kind[it], rng, world.noise)
    p = predict(ag, i, o)
    a = rng.integers(0, 4, n)
    q = (1 - eps_r) * accepted_matrix(world.alpha[it])[np.arange(n), a]
    act = p[np.arange(n), a] > ACT
    opt = np.maximum(0, 3 * q - 2).sum()
    return float((act * (3 * q - 2)).sum() / opt) if opt > 0 else float("nan")


def natural_test_set(rng, n=3000):
    return dict(alpha=rng.uniform(0, 360, n), u=rng.normal(0, 1, (n, 2)), kind=np.zeros(n, dtype=np.int64))


def _edge_from_p(p, th_e, spread=None):
    ba = boundary_action(th_e)
    r = np.arange(len(th_e))
    pb = p[r, ba]
    truth = accepted_matrix(th_e)[r, ba]
    inside, outside = truth, ~truth
    act = pb > ACT
    pay_in = np.where(act[inside], 1.0, 0.0).mean()            # optimal +1
    pay_out = np.where(act[outside], -2.0, 0.0).mean()         # optimal 0
    dec_pay = 0.5 * pay_in + 0.5 * pay_out                      # balanced 50/50
    act_rate = 0.5 * act[inside].mean() + 0.5 * act[outside].mean()
    acts_w = 0.5 * act[inside].mean() + 0.5 * act[outside].mean()
    ref_per_act = (0.5 * act[outside].mean()) / acts_w if acts_w > 0 else float("nan")
    correct05 = (pb > 0.5) == truth
    correct23 = (pb > ACT) == truth
    out = dict(edge_acc=float(correct05.mean()), edge_acc_23=float(correct23.mean()),
               calib_gap=float(np.maximum(pb, 1 - pb).mean() - correct05.mean()),
               mean_conf=float(np.maximum(pb, 1 - pb).mean()),
               false_conf_refused=float(pb[outside].mean()), under_conf_accepted=float(1 - pb[inside].mean()),
               decisive_pay=float(dec_pay), decisive_ratio=float(dec_pay / 0.5), act_rate=float(act_rate),
               refusals_per_act=float(ref_per_act), n_edge=int(len(th_e)))
    if spread is not None:
        out["member_spread"] = float(spread[:, r, ba].std(axis=0).mean())
    return out


def edge_scores(bodies, ag, i, test, rng, noise=0.05):
    th = test["alpha"]; e = is_edge(th)
    o = observe(bodies, ag["body"][i], th[e], test["u"][e], test["kind"][e], rng, noise)
    return _edge_from_p(predict(ag, i, o), th[e])


def edge_scores_ens(bodies, ens, test, rng, noise=0.05):
    th = test["alpha"]; e = is_edge(th)
    o = observe(bodies, ens["body"], th[e], test["u"][e], test["kind"][e], rng, noise)
    mp = ensemble_member_p(ens, o)
    return _edge_from_p(mp.mean(axis=0), th[e], spread=mp)


def ens_home_logloss(world, bodies, ens, rng):
    m = world.kind == 0
    o = observe(bodies, ens["body"], world.alpha[m], world.u[m], world.kind[m], rng, world.noise)
    p = np.clip(ensemble_member_p(ens, o).mean(axis=0), 1e-6, 1 - 1e-6)
    q = accepted_matrix(world.alpha[m])
    return float(-(q * np.log(p) + (1 - q) * np.log(1 - p)).mean())


def log_loss(world, bodies, ag, i, rng, eps_r=0.0, kind=0):
    m = world.kind == kind
    o = observe(bodies, ag["body"][i], world.alpha[m], world.u[m], world.kind[m], rng, world.noise)
    p = np.clip(predict(ag, i, o), 1e-6, 1 - 1e-6)
    q = (1 - eps_r) * accepted_matrix(world.alpha[m])
    return float(-(q * np.log(p) + (1 - q) * np.log(1 - p)).mean())


def kmeans(X, K, seed=0, iters=60):
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    for _ in range(K - 1):  # k-means++
        d2 = np.min(((X[:, None, :] - np.array(C)[None]) ** 2).sum(-1), axis=1)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(C)
    for _ in range(iters):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        for k in range(K):
            if (lab == k).any():
                C[k] = X[lab == k].mean(0)
    return C


def voronoi_entropy(bodies, b, natural, state, K=32):
    """Entropy (bits) of the state's kind-A items over K cells fitted on the natural pool under body b."""
    def obs(w):
        msk = w.kind == 0
        return observe(bodies, b, w.alpha[msk], w.u[msk], w.kind[msk], None, 0.0)
    C = kmeans(obs(natural), K, seed=b)

    def H(X):
        lab = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        pr = np.bincount(lab, minlength=K) / len(lab)
        pr = pr[pr > 0]
        return float(-(pr * np.log2(pr)).sum())
    return H(obs(natural)), H(obs(state))


def appearance_gradient(bodies, ag, world, rng, n=300):
    """Mean |dp/du_b| over random items and offered actions, for each agent's own read coordinate."""
    out = []
    idx = world.idx(0)
    for i in range(ag["W1"].shape[0]):
        b = ag["body"][i]; c = 0 if bodies["types"][b] == 1 else 1
        it = idx[rng.integers(0, len(idx), n)]; a = rng.integers(0, 4, n)
        up, dn = world.u[it].copy(), world.u[it].copy()
        up[:, c] += 0.05; dn[:, c] -= 0.05
        pu = predict(ag, i, observe(bodies, b, world.alpha[it], up, world.kind[it]))[np.arange(n), a]
        pd_ = predict(ag, i, observe(bodies, b, world.alpha[it], dn, world.kind[it]))[np.arange(n), a]
        out.append(float(np.abs(pu - pd_).mean() / 0.1))
    return float(np.mean(out))


def linear_probe(bodies, ag, world, rng, n_train=400, n_test=400, kind=1):
    """Ridge probe from the frozen hidden layer to each action's acceptance on kind-`kind` items."""
    idx = world.idx(kind)
    it = idx[rng.integers(0, len(idx), n_train + n_test)]
    Hh = hidden(ag, 0, observe(bodies, ag["body"][0], world.alpha[it], world.u[it], world.kind[it], rng, world.noise))
    Y = accepted_matrix(world.alpha[it]).astype(float)
    X = np.hstack([Hh, np.ones((len(it), 1))])
    Xtr, Xte, Ytr, Yte = X[:n_train], X[n_train:], Y[:n_train], Y[n_train:]
    Wp = np.linalg.solve(Xtr.T @ Xtr + 1e-2 * np.eye(X.shape[1]), Xtr.T @ Ytr)
    return float(((Xte @ Wp > 0.5) == (Yte > 0.5)).mean())
