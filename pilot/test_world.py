"""Sanity tests with known answers (synthetic data, unit tests only)."""
import numpy as np
import world as W


def test_arcs_and_overlap():
    th = np.arange(0, 360, 0.25)
    acc = W.accepted_matrix(th)
    assert acc.sum(axis=1).min() >= 1  # every angle has at least one accepted action
    assert abs(W.in_overlap(th).mean() - 1 / 3) < 0.01
    assert acc[th == 45.0].sum() == 2 and acc[th == 0.0].sum() == 1
    assert acc[th == 60.0][0, 0] and not acc[th == 60.25][0, 0]


def test_edges():
    th = np.array([25.0, 45.0, 70.0, 75.0, 90.0])
    assert list(W.is_edge(th)) == [True, False, True, False, False]


def test_control_stays_home():
    rng = np.random.default_rng(1)
    w = W.World(rng)
    ag = W.make_agents(rng, 3, 32)
    W.step(w, ag, None, 3000, eta=0.0, seed=1)
    d = np.abs(W.wrapdiff(w.theta, w.home))
    assert d.max() < 15.0, d.max()  # drift noise only: items never leave a narrow band


def test_wear_moves_toward_centre_and_drift_returns():
    rng = np.random.default_rng(2)
    w = W.World(rng, n_items=50, drift_noise=0.0)
    w.home[:] = 20.0
    w.theta[:] = 20.0  # only action 0 is accepted at 20 degrees, so only wear toward 0 can happen
    ag = W.make_agents(rng, 1, 32)
    ag["W2"][:] = 0.0
    ag["b2"][0] = np.array([10.0, -10.0, -10.0, -10.0])  # picks action 0 whenever it is offered
    W.step(w, ag, None, 400, eta=0.2, learn=False, eps=0.0, seed=2, drift=False)
    d = np.abs(W.wrapdiff(w.theta, 0.0))
    assert np.all(d <= 20.0 + 1e-9) and d.mean() < 10, d.mean()
    W.step(w, None, None, 20000, eta=0.0, seed=3)  # nobody acts; only drift back home
    assert np.abs(W.wrapdiff(w.theta, w.home)).max() < 1.0


def test_boundary_action():
    th = np.array([25.0, 58.0, 65.0, 125.0])
    # 25: action 1's lower edge (30); 58: action 0's upper edge (60); 65: action 0 (60); 125: action 2's lower edge (120)
    assert list(W.boundary_action(th)) == [1, 0, 0, 2]


def test_snap_home():
    rng = np.random.default_rng(3)
    w = W.World(rng)
    w.theta = (w.theta + 40) % 360
    w.snap_home()
    assert np.allclose(w.theta, w.home)


def test_test_set_scoring_does_not_learn():
    rng = np.random.default_rng(4)
    w = W.World(rng)
    ag = W.make_agents(rng, 1, 32)
    before = {k: v.copy() for k, v in ag.items()}
    W.edge_scores(w.body, w.noise, ag, 0, W.natural_test_set(rng), rng)
    W.home_log_loss(w, ag, 0, rng)
    assert all(np.array_equal(before[k], ag[k]) for k in ag)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
