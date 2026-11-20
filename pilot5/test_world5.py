"""Sanity tests with known answers for the version 6 world (synthetic data, unit tests only).
Pilot 4's nine tests are kept; the new ones cover the symmetric arcs, reweighting, label shares and confident error."""
import sys, os
import numpy as np
import world5 as W
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "pilot4"))
import world4 as W4


def setup(seed=0, n=300, lam=0.4, types=(1, 2, 1), geometry=None, **kw):
    W.set_geometry(geometry or W.UNEQUAL)
    rng = np.random.default_rng(seed)
    return rng, W.World(rng, n_items=n, **kw), W.make_bodies(rng, list(types), lam)


def test_unequal_arcs():
    W.set_geometry(W.UNEQUAL)
    th = np.arange(0, 360, 0.25)
    acc = W.accepted_matrix(th)
    assert acc.sum(axis=1).min() >= 1                       # every angle accepted by some action
    assert acc[th == 50.0][0].tolist() == [True, False, False, False]   # 50: wide 0 yes, narrow 1 no (edge at 55)
    assert acc[th == 57.0][0].tolist() == [True, True, False, False]    # 55-60 overlap
    assert acc[th == 65.0][0].tolist() == [False, True, False, False]
    assert abs((acc.sum(axis=1) >= 2).mean() - 20 / 360) < 0.005       # four 5-degree overlaps
    assert np.allclose(W.BOUNDARIES, [55, 60, 120, 125, 235, 240, 300, 305])


def test_kind_b_sectors():
    W.set_geometry(W.UNEQUAL)
    th = np.array([10.0, 50.0, 100.0, 190.0, 230.0])
    acc = W.accepted_matrix(th, np.ones(5, dtype=int))
    assert acc.argmax(axis=1).tolist() == [0, 1, 2, 0, 1] and acc.sum(axis=1).tolist() == [1] * 5


def test_half_width_measure_recovers_known_boundaries():
    W.set_geometry(W.UNEQUAL)
    rng = np.random.default_rng(1)
    true = np.array([60.0, 35.0, 60.0, 35.0])
    def pf(ang, u):  # a hand-built predictor: logistic in distance from each centre, edge at `true`
        d = np.abs(W.wrapdiff(ang[:, None], W.CENTRES[None, :]))
        return 1 / (1 + np.exp((d - true[None, :]) / 1.5))
    hw = W._half_widths(pf, rng)
    assert np.allclose(hw, true, atol=0.6), hw
    s = W._shift_summary(np.array([55.0, 40.0, 55.0, 40.0]))
    assert s["shift_wide"] == 5.0 and s["shift_narrow"] == 5.0  # both toward the 45-degree midpoint


def test_passive_feedback_learns_without_acting():
    rng, w, B = setup(2)
    ag = W.make_agents(rng, 1, 16); ag["W2"][:] = 0.0; ag["b2"][:] = -10.0  # never acts
    before = ag["W1"].copy()
    W.step(w, B, ag, None, 500, seed=3, explore=dict(eps_min=0.0, eps_m=0.0, m=0.0))
    assert np.allclose(before, ag["W1"])        # skip rule: no acts, no learning
    W.step(w, B, ag, None, 500, seed=3, passive=True, explore=dict(eps_min=0.0, eps_m=0.0, m=0.0))
    assert not np.allclose(before, ag["W1"])    # passive: learns anyway


def test_matched_edge_sampler():
    rng, w, B = setup(4, n=2000, delta=0.0, drift_a=0.0, drift_u=0.0)
    ag = W.make_agents(rng, 1, 16)
    r = W.step(w, B, ag, None, 6000, seed=5, edge_share=0.08)
    share = W.is_edge(w.alpha[r["item"]]).mean()
    assert 0.06 < share < 0.10, share


def test_eye_never_moves_unread_axis_and_refusal_learner_only():
    rng, w, B = setup(6, delta=0.0, drift_a=0.0, drift_u=0.0)
    u0 = w.u.copy()
    ag = W.make_agents(rng, 1, 16, body=0); ag["b2"][:] = 3.0
    W.step(w, B, None, ag, 3000, seed=7, rule=W.EYE, s_u=0.05, kappa=0.25, eps_r=0.9)
    assert np.allclose(w.u[:, 1], u0[:, 1]) and np.abs(w.u[:, 0] - u0[:, 0]).max() > 0.04


def test_ensemble_priors_fixed():
    rng, w, B = setup(8)
    ens = W.make_ensemble(rng, 3, [8, 16, 32], body=0, beta=3.0)
    P = {k: ens[k].copy() for k in ("P1", "Pb1", "P2", "Pb2")}
    W.step_ens(w, B, ens, None, 2000, seed=9)
    assert all(np.array_equal(P[k], ens[k]) for k in P)
    assert np.all(ens["W1"][0, 8:] == 0)


def test_built_share_and_code_length():
    rng, w, B = setup(10, n=1000)
    assert w.built_share() == 0.0
    w.alpha = np.array([W.CENTRES[np.argmin(np.abs(W.wrapdiff(a, W.CENTRES)))] for a in w.alpha])  # all moved to centres
    assert w.built_share() == 1.0
    ag = W.make_agents(rng, 1, 16)
    bits = W.code_length_bits(w, B, ag, 500, seed=11)
    assert bits > 0


def test_scoring_does_not_learn():
    rng, w, B = setup(12)
    ag = W.make_agents(rng, 1, 16); before = W.copy_agents(ag)
    t = W.natural_test_set(rng)
    W.edge_scores(B, ag, 0, t, rng); W.boundary_shift(B, ag, 0, rng); W.log_loss(w, B, ag, 0, rng); W.competence(w, B, ag, 0, rng)
    assert all(np.array_equal(before[k], ag[k]) for k in ag if isinstance(before[k], np.ndarray))


def test_symmetric_geometry_and_numba_agree():
    W.set_geometry(W.SYMMETRIC)
    th = np.arange(0, 360, 0.25)
    acc = W.accepted_matrix(th)
    assert acc[th == 50.0][0].tolist() == [True, True, False, False]   # 30-60 overlap
    assert acc[th == 25.0][0].tolist() == [True, False, False, False]
    assert np.allclose(W.BOUNDARIES, [30, 60, 120, 150, 210, 240, 300, 330])
    s = W._shift_summary(np.array([55.0, 55.0, 55.0, 55.0]))
    assert s["shift_wide"] == 5.0 and s["shift_narrow"] == 5.0           # inward is toward 45 for every arc
    rng = np.random.default_rng(13)
    for geom in (W.SYMMETRIC, W.UNEQUAL):
        W.set_geometry(geom)
        al = rng.uniform(0, 360, 2000)
        acc = W.accepted_matrix(al)
        assert all(W._accepts(al[k], a, 0, W.HALF) == acc[k, a] for k in range(2000) for a in range(4))
    W.set_geometry(W.UNEQUAL)


def test_unweighted_matches_pilot4_exactly():
    rng, w, B = setup(14)
    ag = W.make_agents(np.random.default_rng(15), 1, 16)
    w4, ag4 = w.copy(), W.copy_agents(ag)
    r5 = W.step(w, B, ag, None, 3000, seed=16, passive=True)
    r4 = W4.step(w4, B, ag4, None, 3000, seed=16, passive=True)
    assert all(np.array_equal(ag[k], ag4[k]) for k in ("W1", "b1", "W2", "b2"))
    assert np.array_equal(r5["act"], r4["act"]) and np.allclose(w.alpha, w4.alpha)
    assert np.all(r5["w"][r5["upd"] == 1] == 1.0)


def test_reweighting_hits_target_share():
    rng, w, B = setup(17, n=1500)
    for s_target in (None, 0.5, 0.9):
        ag = W.make_agents(np.random.default_rng(18), 1, 16)
        r = W.step(w.copy(), B, ag, None, 8000, seed=19, passive=True, target_share=s_target)
        sw, sn = W.label_share(r, W.WIDE), W.label_share(r, W.NARROW)
        if s_target is None:  # unweighted passive: the share is what the arcs accept
            assert abs(sw - 120 / 360) < 0.03 and abs(sn - 70 / 360) < 0.03, (sw, sn)
        else:
            assert abs(sw - s_target) < 0.03 and abs(sn - s_target) < 0.03, (s_target, sw, sn)


def test_confident_error_measure():
    W.set_geometry(W.UNEQUAL)
    rng = np.random.default_rng(20)
    test = W.natural_test_set(rng, n=6000)
    def sharp(ix):
        d = np.abs(W.wrapdiff(test["alpha"][ix][:, None], W.CENTRES[None, :]))
        return 1 / (1 + np.exp((d - W.HALF[None, :]) / 0.2))
    good = W._scores_from_pfun(sharp, test, rng)
    bad = W._scores_from_pfun(lambda ix: 1 - sharp(ix), test, rng)
    assert good["wide_conf_err"] < 0.02 and good["narrow_conf_err"] < 0.02
    assert bad["wide_conf_err"] > 0.98 and bad["narrow_conf_err"] > 0.98


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
