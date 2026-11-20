"""Sanity tests with known answers for the version 3 world (synthetic data, unit tests only)."""
import numpy as np
import world2 as W


def setup(seed=0, n=300, **kw):
    rng = np.random.default_rng(seed)
    return rng, W.World(rng, n_items=n, **kw), W.make_bodies(rng, 3)


def test_arcs():
    th = np.arange(0, 360, 0.25)
    assert abs(W.in_overlap(th).mean() - 1 / 3) < 0.01
    assert list(W.boundary_action(np.array([25.0, 58.0, 65.0, 125.0]))) == [1, 0, 0, 2]
    assert list(W.nearest_action(np.array([10.0, 50.0, 100.0]))) == [0, 1, 1]


def test_skip_rule_pays_zero():
    rng, w, B = setup(1)
    ag = W.make_agents(rng, 1, 16)
    ag["W2"][:] = 0.0
    ag["b2"][:] = -10.0  # predicted acceptance ~0: always skip unless exploring
    r = W.step(w, B, ag, None, 2000, seed=1, eps=0.0, learn=False)
    assert r["act"].sum() == 0 and np.all(r["pay"] == 0)


def test_payoffs_and_random_refusal():
    rng, w, B = setup(2)
    w.alpha[:] = 0.0; w.home_a[:] = 0.0  # only action 0 accepted
    ag = W.make_agents(rng, 1, 16)
    ag["W2"][:] = 0.0
    ag["b2"][:] = 10.0  # always act
    r = W.step(w, B, ag, None, 4000, seed=2, eps=0.0, learn=False)
    acc0 = r["pay"][r["offer"] == 0]
    assert np.all(acc0 == 1.0) and np.all(r["pay"][r["offer"] != 0] == -2.0)
    r = W.step(w, B, ag, None, 4000, seed=3, eps=0.0, learn=False, eps_r=0.2)
    refused = (r["pay"][r["offer"] == 0] == -2.0).mean()
    assert 0.15 < refused < 0.25, refused


def test_no_shaping_control_and_drift_home():
    rng, w, B = setup(3)
    ag = W.make_agents(rng, 3, 16)
    W.step(w, B, ag, None, 3000, seed=4)
    assert np.abs(W.wrapdiff(w.alpha, w.home_a)).max() < 10 and np.abs(w.u - w.home_u).max() < 0.5
    w.alpha = (w.alpha + 30) % 360; w.u = w.u + 1.0
    w.drift_a = w.drift_u = 0.0
    W.step(w, B, None, None, 20000, seed=5)
    assert np.abs(W.wrapdiff(w.alpha, w.home_a)).max() < 1.0 and np.abs(w.u - w.home_u).max() < 0.05


def test_wear_moves_angle_only():
    rng, w, B = setup(4, delta=0.0, drift_a=0.0, drift_u=0.0)
    w.alpha[:] = 20.0; w.home_a[:] = 20.0
    u0 = w.u.copy()
    ag = W.make_agents(rng, 1, 16); ag["W2"][:] = 0.0; ag["b2"][:] = 10.0
    W.step(w, B, None, ag, 8000, seed=6, rule=W.WEAR, eta=0.2, eps=0.0)  # as a frozen (shaping) agent; ~6-7 accepted uses per item
    assert np.abs(W.wrapdiff(w.alpha, 0.0)).mean() < 10 and np.allclose(w.u, u0)


def test_eye_moves_appearance():
    rng, w, B = setup(5, drift_a=0.0, drift_u=0.0)
    u0 = w.u.copy()
    ag = W.make_agents(rng, 1, 16)
    ag["b2"][:] = 3.0  # acts often
    W.step(w, B, None, ag, 2000, seed=7, rule=W.EYE, s_u=0.05, kappa=0.0, eps=0.0)
    assert np.abs(w.u - u0).max() > 0.04  # appearance moved
    a0 = w.alpha.copy()
    W.step(w, B, None, ag, 2000, seed=8, rule=W.EYE, s_u=0.05, kappa=0.0, eps=0.0)
    assert np.allclose(w.alpha, a0)  # kappa = 0: the angle never moves


def test_filtered_sampler_never_offers_edges():
    rng, w, B = setup(6, delta=0.0, drift_a=0.0, drift_u=0.0)  # angles stay at home, so home = current
    ag = W.make_agents(rng, 1, 16)
    r = W.step(w, B, ag, None, 3000, seed=9, filtered=True)
    assert not W.is_edge(w.home_a[r["item"]]).any()


def test_kind_b_never_shaped():
    rng, w, B = setup(7, drift_a=0.0, drift_u=0.0)
    w.add_kind_b(rng, 100)
    b = w.kind == 1
    a0, u0 = w.alpha[b].copy(), w.u[b].copy()
    ag = W.make_agents(rng, 1, 16); ag["W2"][:] = 0.0; ag["b2"][:] = 10.0
    W.step(w, B, ag, None, 3000, seed=10, rule=W.WEAR, eta=0.3, pB=0.5, learner_shapes=True)
    assert np.allclose(w.alpha[b], a0) and np.allclose(w.u[b], u0)


def test_snap_and_scoring_do_not_learn():
    rng, w, B = setup(8)
    w.alpha = (w.alpha + 50) % 360
    w.snap_home()
    assert np.allclose(w.alpha, w.home_a)
    ag = W.make_agents(rng, 1, 16)
    before = W.copy_agents(ag)
    t = W.natural_test_set(rng)
    W.edge_scores(B, ag, 0, t, rng); W.log_loss(w, B, ag, 0, rng); W.competence(w, B, ag, 0, rng)
    assert all(np.array_equal(before[k], ag[k]) for k in ag)


def test_forestry_same_state_zero_drop():
    rng, w, B = setup(9)
    h0, h1 = W.forestry_entropy(B, 0, w, w.copy())
    assert abs(h0 - h1) < 1e-12 and h0 > 3.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
