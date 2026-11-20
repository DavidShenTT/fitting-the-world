"""Sanity tests with known answers for the version 5 world (synthetic data, unit tests only)."""
import numpy as np
import world4 as W


def setup(seed=0, n=300, lam=0.4, types=(1, 2, 1), **kw):
    rng = np.random.default_rng(seed)
    return rng, W.World(rng, n_items=n, **kw), W.make_bodies(rng, list(types), lam)


def test_unequal_arcs():
    th = np.arange(0, 360, 0.25)
    acc = W.accepted_matrix(th)
    assert acc.sum(axis=1).min() >= 1                       # every angle accepted by some action
    assert acc[th == 50.0][0].tolist() == [True, False, False, False]   # 50: wide 0 yes, narrow 1 no (edge at 55)
    assert acc[th == 57.0][0].tolist() == [True, True, False, False]    # 55-60 overlap
    assert acc[th == 65.0][0].tolist() == [False, True, False, False]
    assert abs((acc.sum(axis=1) >= 2).mean() - 20 / 360) < 0.005       # four 5-degree overlaps
    assert np.allclose(W.BOUNDARIES, [55, 60, 120, 125, 235, 240, 300, 305])


def test_kind_b_sectors():
    th = np.array([10.0, 50.0, 100.0, 190.0, 230.0])
    acc = W.accepted_matrix(th, np.ones(5, dtype=int))
    assert acc.argmax(axis=1).tolist() == [0, 1, 2, 0, 1] and acc.sum(axis=1).tolist() == [1] * 5


def test_half_width_measure_recovers_known_boundaries():
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
