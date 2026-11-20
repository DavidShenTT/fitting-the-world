"""Unit tests for src/world.py: every item BRIEF.md section 6 lists, plus the additions in PREREGISTRATION.md.
All data here are SYNTHETIC (random worlds and hand-built predictors), for unit tests only."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "pilot6"))
import world as W  # noqa: E402
import world6 as W6  # noqa: E402

FROZEN_NO_EXPLORE = dict(eps_min=0.0, eps_m=0.0, m=0.0)


def setup(seed=0, n=300, lam=0.4, types=(1, 2, 1), geometry=None, **kw):
    W.set_geometry(geometry or W.UNEQUAL)
    rng = np.random.default_rng(seed)
    return rng, W.World(rng, n_items=n, **kw), W.make_bodies(rng, list(types), lam)


def sharp_pfun(test, width=0.2):
    def pf(ix):
        d = np.abs(W.wrapdiff(test["alpha"][ix][:, None], W.CENTRES[None, :]))
        return 1 / (1 + np.exp((d - W.HALF[None, :]) / width))
    return pf


# ------------------------------------------------------------------ geometry

def test_unequal_arcs():
    W.set_geometry(W.UNEQUAL)
    th = np.arange(0, 360, 0.25)
    acc = W.accepted_matrix(th)
    assert acc.sum(axis=1).min() >= 1
    assert acc[th == 50.0][0].tolist() == [True, False, False, False]
    assert acc[th == 57.0][0].tolist() == [True, True, False, False]
    assert acc[th == 65.0][0].tolist() == [False, True, False, False]
    assert abs((acc.sum(axis=1) >= 2).mean() - 20 / 360) < 0.005
    assert np.allclose(W.BOUNDARIES, [55, 60, 120, 125, 235, 240, 300, 305])


def test_symmetric_world_and_numba_agree():
    W.set_geometry(W.SYMMETRIC)
    th = np.arange(0, 360, 0.25)
    acc = W.accepted_matrix(th)
    assert acc[th == 50.0][0].tolist() == [True, True, False, False]
    assert acc[th == 25.0][0].tolist() == [True, False, False, False]
    assert abs(W.in_overlap(th).mean() - 1 / 3) < 0.005          # a third of the ring is overlap
    assert np.allclose(W.BOUNDARIES, [30, 60, 120, 150, 210, 240, 300, 330])
    rng = np.random.default_rng(13)
    for geom in (W.SYMMETRIC, W.UNEQUAL):
        W.set_geometry(geom)
        al = rng.uniform(0, 360, 2000)
        acc = W.accepted_matrix(al)
        assert all(W._accepts(al[k], a, 0, W.HALF) == acc[k, a] for k in range(2000) for a in range(4))
    W.set_geometry(W.UNEQUAL)


def test_overlap_built_share_and_masks():
    rng, w, B = setup(30, n=3000, geometry=W.SYMMETRIC)
    assert w.built_share_overlap() == 0.0
    assert abs(w.overlap_home().mean() - 1 / 3) < 0.03
    un = w.unambiguous_home()
    assert not (un & w.overlap_home()).any()
    assert abs(un.mean() - (1 - 1 / 3 - 4 * 20 / 360)) < 0.03      # 10-degree margin on both sides of each overlap
    w.alpha = np.array([W.CENTRES[np.argmin(np.abs(W.wrapdiff(a, W.CENTRES)))] for a in w.alpha])
    assert w.built_share_overlap() == 1.0 and w.built_count_overlap() == int(w.overlap_home().sum())
    W.set_geometry(W.UNEQUAL)


# ------------------------------------------------------------------ regression against pilot 6

def test_matches_pilot6_when_builder_policy_equals_learner_policy():
    for passive in (False, True):
        rng, w, B = setup(14)
        W6.set_geometry(W6.UNEQUAL)
        ag = W.make_agents(np.random.default_rng(15), 1, 16)
        fr = W.make_agents(np.random.default_rng(16), 2, 16)
        w6, ag6, fr6 = w.copy(), W.copy_agents(ag), W.copy_agents(fr)
        r = W.step(w, B, ag, fr, 3000, seed=16, passive=passive, rule=W.WEAR, eta=0.15)
        r6 = W6.step(w6, B, ag6, fr6, 3000, seed=16, passive=passive, rule=W6.WEAR, eta=0.15)
        assert all(np.array_equal(ag[k], ag6[k]) for k in ("W1", "b1", "W2", "b2"))
        assert np.array_equal(r["act"], r6["act"]) and np.allclose(w.alpha, w6.alpha)


def test_builder_policy_ignores_learner_exploration():
    rng, w, B = setup(31, delta=0.0, drift_a=0.0, drift_u=0.0)
    fr = W.make_agents(np.random.default_rng(32), 3, 16); fr["W2"][:] = 0.0; fr["b2"][:] = -5.0  # acts only via floor
    w1, w2 = w.copy(), w.copy()
    W.step(w1, B, None, W.copy_agents(fr), 2000, seed=33, rule=W.WEAR, eta=0.15)
    W.step(w2, B, None, W.copy_agents(fr), 2000, seed=33, rule=W.WEAR, eta=0.15, explore=dict(eps_min=0.3, eps_m=0.6, m=0.2))
    assert np.array_equal(w1.alpha, w2.alpha)                    # learner's exploration doesn't reach builders
    w3 = w.copy()
    W.step(w3, B, None, W.copy_agents(fr), 2000, seed=33, rule=W.WEAR, eta=0.15, builder_explore=dict(eps_min=0.3, eps_m=0.6, m=0.2))
    assert (w3.alpha != w.alpha).sum() > (w1.alpha != w.alpha).sum()


# ------------------------------------------------------------------ payoffs, skip rule, feedback

def test_payoff_and_skip_rule():
    rng, w, B = setup(34, delta=0.0, drift_a=0.0, drift_u=0.0)
    ag = W.make_agents(rng, 1, 16); ag["W2"][:] = 0.0; ag["b2"][:] = 10.0       # always acts
    r = W.step(w, B, ag, None, 2000, seed=35, lr=0.0, explore=FROZEN_NO_EXPLORE)
    acc = W.accepted_matrix(r["ang"])[np.arange(2000), r["offer"]]
    assert r["act"].all() and np.array_equal(r["pay"], np.where(acc, 1.0, -2.0))
    ag["b2"][:] = -10.0                                                          # never acts
    r = W.step(w, B, ag, None, 2000, seed=36, lr=0.0, explore=FROZEN_NO_EXPLORE)
    assert not r["act"].any() and np.all(r["pay"] == 0.0)
    assert W._decide(0.67, 0.0, 0.0, 0.0) and not W._decide(0.66, 0.0, 0.0, 0.0)   # breaks even at 2/3


def test_passive_feedback_learns_without_acting():
    rng, w, B = setup(2)
    ag = W.make_agents(rng, 1, 16); ag["W2"][:] = 0.0; ag["b2"][:] = -10.0
    before = ag["W1"].copy()
    W.step(w, B, ag, None, 500, seed=3, explore=FROZEN_NO_EXPLORE)
    assert np.allclose(before, ag["W1"])
    W.step(w, B, ag, None, 500, seed=3, passive=True, explore=FROZEN_NO_EXPLORE)
    assert not np.allclose(before, ag["W1"])


def test_label_share_unweighted_passive():
    rng, w, B = setup(17, n=1500)
    ag = W.make_agents(np.random.default_rng(18), 1, 16)
    r = W.step(w.copy(), B, ag, None, 8000, seed=19, passive=True)
    sw, sn = W.label_share(r, W.WIDE), W.label_share(r, W.NARROW)
    assert abs(sw - 120 / 360) < 0.03 and abs(sn - 70 / 360) < 0.03, (sw, sn)


def test_biased_offers_hit_share_and_keep_outcomes():
    rng, w, B = setup(21, n=1500)
    ag = W.make_agents(np.random.default_rng(22), 1, 16)
    r = W.step(w, B, ag, None, 12000, seed=23, passive=True, offer_share=0.9)
    assert abs(r["y"].mean() - 0.9) < 0.02
    acc = W.accepted_matrix(r["ang"])[np.arange(len(r["ang"])), r["offer"]]
    assert np.array_equal(acc.astype(int), r["y"])


def test_learner_only_random_refusal():
    rng, w, B = setup(37, delta=0.0, drift_a=0.0, drift_u=0.0)
    ag = W.make_agents(rng, 1, 16); ag["W2"][:] = 0.0; ag["b2"][:] = 10.0
    r = W.step(w, B, ag, None, 1000, seed=38, lr=0.0, eps_r=1.0, explore=FROZEN_NO_EXPLORE)
    assert r["y"].sum() == 0                                     # the learner never sees an acceptance
    u0 = w.u.copy()
    fr = W.make_agents(rng, 1, 16, body=0); fr["b2"][:] = 3.0
    W.step(w, B, None, fr, 3000, seed=7, rule=W.EYE, s_u=0.05, kappa=0.25, eps_r=0.9)
    assert np.abs(w.u[:, 0] - u0[:, 0]).max() > 0.04             # builders still shape under eps_r


# ------------------------------------------------------------------ world dynamics

def test_drift_toward_home_every_coordinate_one_rate():
    rng, w, B = setup(39, n=200, delta=0.01, drift_a=0.0, drift_u=0.0)
    w.alpha = (w.home_a + 20.0) % 360; w.u = w.home_u + np.array([0.5, -0.5])
    W.step(w, B, None, None, 100, seed=40)
    f = (1 - 0.01) ** 100
    assert np.allclose(W.wrapdiff(w.alpha, w.home_a), 20.0 * f, atol=1e-6)
    assert np.allclose(w.u - w.home_u, np.array([0.5, -0.5]) * f, atol=1e-9)


def test_wear_moves_fraction_toward_centre_and_control_never_moves():
    rng, w, B = setup(41, n=50, delta=0.0, drift_a=0.0, drift_u=0.0)
    w.alpha[:] = 20.0; w.home_a[:] = 20.0
    fr = W.make_agents(rng, 1, 16); fr["W2"][:] = 0.0; fr["b2"][:] = 10.0      # builder always acts
    w0 = w.copy(); W.step(w0, B, None, W.copy_agents(fr), 500, seed=42, rule=W.NONE)
    assert np.array_equal(w0.alpha, w.alpha) and np.array_equal(w0.u, w.u)    # no-shaping control
    # one accepted use of action 0 (centre 0) at angle 20 moves it to 20 - 0.15*20 = 17
    a = np.array([20.0]); u = np.zeros((1, 2)); hu = u.copy(); kind = np.zeros(1, dtype=np.int64)
    oh = np.empty(16); a1 = np.empty(16); zh = np.empty(4)
    W._shape(0, a, u, kind, 0, W.WEAR, 0.15, 0.0, 1.0, B["Win"], B["bin"], B["Wout"], 0, B["gamma"],
             fr["W1"], fr["b1"], fr["W2"], fr["b2"], 0, oh, a1, zh)
    assert abs(a[0] - 17.0) < 1e-9 and np.array_equal(u, hu)


def test_eye_never_moves_unread_axis():
    rng, w, B = setup(6, delta=0.0, drift_a=0.0, drift_u=0.0)
    u0 = w.u.copy()
    ag = W.make_agents(rng, 1, 16, body=0); ag["b2"][:] = 3.0
    W.step(w, B, None, ag, 3000, seed=7, rule=W.EYE, s_u=0.05, kappa=0.25)
    assert np.allclose(w.u[:, 1], u0[:, 1]) and np.abs(w.u[:, 0] - u0[:, 0]).max() > 0.04
    rng, w, B = setup(6, delta=0.0, drift_a=0.0, drift_u=0.0)
    u0 = w.u.copy()
    ag = W.make_agents(rng, 1, 16, body=1); ag["b2"][:] = 3.0                  # type-2 body reads u2 only
    W.step(w, B, None, ag, 3000, seed=7, rule=W.EYE, s_u=0.05, kappa=0.25)
    assert np.allclose(w.u[:, 0], u0[:, 0]) and np.abs(w.u[:, 1] - u0[:, 1]).max() > 0.04


def test_snap_back():
    rng, w, B = setup(43)
    w.alpha = (w.alpha + 30) % 360; w.u = w.u + 1.0
    w.snap_home()
    assert np.array_equal(w.alpha, w.home_a) and np.array_equal(w.u, w.home_u)


def test_matched_edge_sampler():
    rng, w, B = setup(4, n=2000, delta=0.0, drift_a=0.0, drift_u=0.0)
    ag = W.make_agents(rng, 1, 16)
    r = W.step(w, B, ag, None, 6000, seed=5, edge_share=0.08, chunk=50)
    share = W.is_edge(w.alpha[r["item"]]).mean()
    assert 0.06 < share < 0.10, share


# ------------------------------------------------------------------ ensembles

def test_ensemble_priors_fixed():
    rng, w, B = setup(8)
    ens = W.make_ensemble(rng, 3, [8, 16, 32], body=0, beta=3.0)
    P = {k: ens[k].copy() for k in ("P1", "Pb1", "P2", "Pb2")}
    W.step_ens(w, B, ens, None, 2000, seed=9)
    assert all(np.array_equal(P[k], ens[k]) for k in P)
    assert np.all(ens["W1"][0, 8:] == 0)


def test_thompson_sampling_and_bootstrap_mask():
    rng, w, B = setup(44, delta=0.0, drift_a=0.0, drift_u=0.0)
    K = 4
    ens = W.make_ensemble(rng, K, [16] * K, body=0)
    ens["W2"][:] = 0.0; ens["b2"][:] = -10.0; ens["b2"][0] = 10.0   # only member 0 would act
    acts = W.step_ens(w.copy(), B, ens, None, 8000, seed=45, lr=0.0, eps_min=0.0)
    assert abs(acts.mean() - 1 / K) < 0.02                         # one random member decides each turn
    for k in ("W1", "b1", "W2", "b2"):                             # identical members ...
        ens[k][:] = ens[k][0]
    ens["b2"][:] = 10.0
    W.step_ens(w.copy(), B, ens, None, 3000, seed=46)
    assert not np.allclose(ens["W1"][0], ens["W1"][1])             # ... diverge under the bootstrap mask


def test_ensemble_passive_learns_without_acting():
    rng, w, B = setup(50)
    ens = W.make_ensemble(rng, 3, [16] * 3, body=0)
    ens["W2"][:] = 0.0; ens["b2"][:] = -10.0
    W1 = ens["W1"].copy()
    W.step_ens(w.copy(), B, ens, None, 500, seed=51, eps_min=0.0)
    assert np.array_equal(W1, ens["W1"])
    W.step_ens(w.copy(), B, ens, None, 500, seed=51, eps_min=0.0, passive=True)
    assert not np.allclose(W1, ens["W1"])


# ------------------------------------------------------------------ measures

def test_half_width_measure_recovers_known_boundaries():
    W.set_geometry(W.UNEQUAL)
    rng = np.random.default_rng(1)
    true = np.array([60.0, 35.0, 60.0, 35.0])
    def pf(ang, u):
        d = np.abs(W.wrapdiff(ang[:, None], W.CENTRES[None, :]))
        return 1 / (1 + np.exp((d - true[None, :]) / 1.5))
    assert np.allclose(W._half_widths(pf, rng), true, atol=0.6)
    s = W._shift_summary(np.array([55.0, 40.0, 55.0, 40.0]))
    assert s["shift_wide"] == 5.0 and s["shift_narrow"] == 5.0


def test_confident_error_and_edge_log_loss():
    W.set_geometry(W.UNEQUAL)
    rng = np.random.default_rng(20)
    test = W.natural_test_set(rng, n=6000)
    good = W._scores_from_pfun(sharp_pfun(test), test, rng)
    bad = W._scores_from_pfun(lambda ix: 1 - sharp_pfun(test)(ix), test, rng)
    assert good["wide_conf_err"] < 0.02 and good["narrow_conf_err"] < 0.02
    assert bad["wide_conf_err"] > 0.98 and bad["narrow_conf_err"] > 0.98
    assert good["wide_edge_logloss"] < 0.05 and bad["wide_edge_logloss"] > 5


def test_refusal_sources_counts():
    W.set_geometry(W.UNEQUAL)
    rec = {"upd": np.array([1, 1, 1, 1, 1, 0]), "offer": np.array([0, 0, 0, 0, 1, 0]),
           "y": np.array([1, 0, 0, 0, 0, 0]), "ang": np.array([10.0, 70.0, 90.0, 300.0, 200.0, 90.0])}
    assert W.refusal_sources(rec, (0,)) == dict(acc=1, inside_ref=1, near_ref=1, far_ref=1)


def test_natural_test_set_scoring_neither_learns_nor_shapes():
    rng, w, B = setup(12)
    ag = W.make_agents(rng, 1, 16); before = W.copy_agents(ag)
    t = W.natural_test_set(rng); t0 = {k: v.copy() for k, v in t.items()}
    w0 = w.copy()
    W.edge_scores(B, ag, 0, t, rng); W.boundary_shift(B, ag, 0, rng); W.log_loss(w, B, ag, 0, rng)
    W.competence(w, B, ag, 0, rng); W.chosen_actions(B, ag, 0, t["alpha"], t["u"]); W.hidden_acts(B, ag, 0, w.alpha, w.u)
    assert all(np.array_equal(before[k], ag[k]) for k in ag if isinstance(before[k], np.ndarray))
    assert all(np.array_equal(t0[k], t[k]) for k in t) and np.array_equal(w0.alpha, w.alpha)


def test_built_share_and_code_length():
    rng, w, B = setup(10, n=1000)
    assert w.built_share() == 0.0
    w.alpha = np.array([W.CENTRES[np.argmin(np.abs(W.wrapdiff(a, W.CENTRES)))] for a in w.alpha])
    assert w.built_share() == 1.0
    assert W.code_length_bits(w, B, W.make_agents(rng, 1, 16), 500, seed=11) > 0


def test_cohen_kappa():
    rng = np.random.default_rng(47)
    x = rng.integers(0, 4, 5000)
    assert W.cohen_kappa(x, x) == 1.0
    assert abs(W.cohen_kappa(x, rng.integers(0, 4, 5000))) < 0.03
    # hand example: 3 of 4 agree, frequencies x=(2,2,0,0)/4, y=(3,1,0,0)/4 -> pe = 0.5, kappa = (0.75-0.5)/0.5
    assert abs(W.cohen_kappa(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 0])) - 0.5) < 1e-12


def test_e_k():
    rng = np.random.default_rng(48)
    X = rng.normal(size=(300, 16))
    e_same, _ = W.e_k(X, X)
    e_rand, _ = W.e_k(X, rng.normal(size=(300, 16)))
    assert e_same > 0.99 and abs(e_rand) < 0.05


def test_chosen_action_is_argmax():
    rng, w, B = setup(49)
    ag = W.make_agents(rng, 1, 16)
    ch = W.chosen_actions(B, ag, 0, w.alpha, w.u)
    p = W.predict(ag, 0, W.observe(B, 0, w.alpha, w.u, w.kind, None, 0.0))
    assert np.array_equal(ch, p.argmax(axis=1))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
