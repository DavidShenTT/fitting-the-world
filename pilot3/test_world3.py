"""Sanity tests with known answers for the version 4 world (synthetic data, unit tests only)."""
import numpy as np
import world3 as W


def setup(seed=0, n=300, lam=0.5, types=(1, 2, 1), **kw):
    rng = np.random.default_rng(seed)
    return rng, W.World(rng, n_items=n, **kw), W.make_bodies(rng, list(types), lam)


def test_arcs():
    th = np.arange(0, 360, 0.25)
    assert abs(W.in_overlap(th).mean() - 1 / 3) < 0.01
    assert list(W.boundary_action(np.array([25.0, 58.0, 65.0, 125.0]))) == [1, 0, 0, 2]


def test_bodies_read_only_their_axis():
    rng, w, B = setup(1)
    a, u, k = w.alpha[:50], w.u[:50].copy(), w.kind[:50]
    u2 = u.copy(); u2[:, 1] += 1.0
    u1 = u.copy(); u1[:, 0] += 1.0
    assert np.allclose(W.observe(B, 0, a, u, k), W.observe(B, 0, a, u2, k))      # type 1 ignores u2
    assert not np.allclose(W.observe(B, 0, a, u, k), W.observe(B, 0, a, u1, k))
    assert np.allclose(W.observe(B, 1, a, u, k), W.observe(B, 1, a, u1, k))      # type 2 ignores u1


def test_eye_never_moves_unread_axis():
    rng, w, B = setup(2, delta=0.0, drift_a=0.0, drift_u=0.0)
    u0 = w.u.copy()
    ag = W.make_agents(rng, 1, 16, body=0)  # a type-1 builder
    ag["b2"][:] = 3.0
    W.step(w, B, None, ag, 3000, seed=3, rule=W.EYE, s_u=0.05, kappa=0.25)
    assert np.allclose(w.u[:, 1], u0[:, 1]) and np.abs(w.u[:, 0] - u0[:, 0]).max() > 0.04


def test_random_refusal_is_learner_only():
    finals = []
    for er in (0.0, 0.9):
        rng, w, B = setup(3, delta=0.0, drift_a=0.0, drift_u=0.0)
        w.alpha[:] = 20.0; w.home_a[:] = 20.0
        ag = W.make_agents(rng, 1, 16); ag["W2"][:] = 0.0; ag["b2"][:] = 10.0
        W.step(w, B, None, ag, 3000, seed=4, rule=W.WEAR, eta=0.2, eps_r=er)
        finals.append(w.alpha.copy())
    assert np.allclose(finals[0], finals[1])
    rng, w, B = setup(4)
    w.alpha[:] = 0.0; w.home_a[:] = 0.0
    ag = W.make_agents(rng, 1, 16); ag["W2"][:] = 0.0; ag["b2"][:] = 10.0
    r = W.step(w, B, ag, None, 4000, seed=5, eps_r=0.2, learn=False)
    refused = (r["pay"][r["offer"] == 0] == -2.0).mean()
    assert 0.15 < refused < 0.25, refused


def test_exploration_floor():
    rng, w, B = setup(5)
    ag = W.make_agents(rng, 1, 16); ag["W2"][:] = 0.0; ag["b2"][:] = -10.0  # p ~ 0, outside the margin
    r = W.step(w, B, ag, None, 10000, seed=6, learn=False, explore=dict(eps_min=0.02, eps_m=0.5, m=0.2))
    assert 0.012 < r["act"].mean() < 0.028, r["act"].mean()


def test_margin_exploration():
    rng, w, B = setup(6)
    ag = W.make_agents(rng, 1, 16); ag["W2"][:] = 0.0; ag["b2"][:] = 0.0  # p = 0.5, inside a margin of 0.2
    r = W.step(w, B, ag, None, 10000, seed=7, learn=False, explore=dict(eps_min=0.0, eps_m=0.5, m=0.2))
    assert 0.45 < r["act"].mean() < 0.55, r["act"].mean()


def test_drift_home_all_coordinates():
    rng, w, B = setup(7, drift_a=0.0, drift_u=0.0)
    w.alpha = (w.alpha + 30) % 360; w.u = w.u + 1.0
    W.step(w, B, None, None, 20000, seed=8)
    assert np.abs(W.wrapdiff(w.alpha, w.home_a)).max() < 1.0 and np.abs(w.u - w.home_u).max() < 0.05


def test_ensemble_priors_fixed_padding_dead():
    rng, w, B = setup(8)
    ens = W.make_ensemble(rng, 3, [8, 16, 32], body=0, beta=1.0)
    P = {k: ens[k].copy() for k in ("P1", "Pb1", "P2", "Pb2")}
    W0 = ens["W1"].copy()
    acts = W.step_ens(w, B, ens, None, 3000, seed=9)
    assert all(np.array_equal(P[k], ens[k]) for k in P)       # priors never trained
    assert np.all(ens["W1"][0, 8:] == 0) and np.all(ens["W2"][1, :, 16:] == 0)  # padded units stay dead
    assert not np.allclose(W0, ens["W1"]) and acts.mean() > 0.0


def test_filtered_sampler_never_offers_edges():
    rng, w, B = setup(9, delta=0.0, drift_a=0.0, drift_u=0.0)
    ag = W.make_agents(rng, 1, 16)
    r = W.step(w, B, ag, None, 3000, seed=10, filtered=True)
    assert not W.is_edge(w.home_a[r["item"]]).any()


def test_scoring_does_not_learn_and_snap():
    rng, w, B = setup(10)
    w.alpha = (w.alpha + 50) % 360; w.snap_home()
    assert np.allclose(w.alpha, w.home_a)
    ag = W.make_agents(rng, 1, 16); before = W.copy_agents(ag)
    t = W.natural_test_set(rng)
    W.edge_scores(B, ag, 0, t, rng); W.log_loss(w, B, ag, 0, rng); W.competence(w, B, ag, 0, rng)
    W.appearance_gradient(B, ag, w, rng); W.linear_probe(B, ag, w, rng, kind=0)
    assert all(np.array_equal(before[k], ag[k]) for k in ag if isinstance(before[k], np.ndarray))


def test_voronoi_same_state_zero_drop():
    rng, w, B = setup(11)
    h0, h1 = W.voronoi_entropy(B, 0, w, w.copy(), K=16)
    assert abs(h0 - h1) < 1e-12 and h0 > 3.5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
