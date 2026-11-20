"""Layer 2 pilot runner: design checks C2-C5 and readouts for E1, E8, E9, E10 (PILOT_SPEC.md, PILOT_LOG.md).

Writes one CSV per part into results/. Seeds 1000-1002 only.
"""
import itertools
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import world as W

OUT = "results"
SEEDS = (1000, 1001, 1002)
WIDTHS = (8, 32, 128)
BUDGETS = (5000, 15000)
M, BUILDER_W, WARMUP = 6, 32, 30000
ETA, LR, EPS = 0.2, 0.05, 0.05
T2 = 15000


class Seeder:
    """Distinct numba seeds for every call within one world seed."""
    def __init__(self, base):
        self.n = base * 1000

    def __call__(self):
        self.n += 1
        return self.n


def warm_builders(seed, rule, eta, sd):
    rng = np.random.default_rng(seed)
    nat = W.World(rng)
    w = nat.copy()
    builders = W.make_agents(np.random.default_rng(seed + 1), M, BUILDER_W)
    t0 = time.time()
    trace = []
    for c in range(WARMUP // 5000):
        W.step(w, builders, None, 5000, eta=eta, rule=rule, lr=LR, eps=EPS, seed=sd())
        trace.append(w.built_share())
    return nat, w, builders, trace, time.time() - t0


def scores(agent, home_world, nat, test, seed_eval):
    r = np.random.default_rng(seed_eval)
    e = W.edge_scores(nat.body, nat.noise, agent, 0, test, r)
    return dict(home_log_loss=W.home_log_loss(home_world, agent, 0, r),
                competence_home=W.greedy_acceptance(home_world, agent, 0, r),
                edge_acc=e["edge_acc"], calib_gap=e["calib_gap"], n_edge=e["n_edge"])


def raise_agent(kind, width, budget, nat, built, builders_frozen, seed, sd):
    agent = W.make_agents(np.random.default_rng(seed * 7 + width), 1, width)
    if kind == "wild":
        w = nat.copy()
        W.step(w, agent, None, budget, eta=0.0, lr=LR, eps=EPS, seed=sd())
    else:
        w = built.copy()
        W.step(w, agent, builders_frozen, budget, eta=ETA, lr=LR, eps=EPS, seed=sd())
    return agent, w


def snap_test(agent, nat, sd, seed_eval):
    """E9: every item back at home, nobody else acting, eta = 0; the agent keeps learning."""
    a = W.copy_agents(agent)
    w = nat.copy()
    w.snap_home()
    acc = W.step(w, a, None, 500, eta=0.0, lr=LR, eps=EPS, seed=sd())
    r = np.random.default_rng(seed_eval)
    comp0 = None
    recover = None
    w2 = nat.copy(); w2.snap_home(); a2 = W.copy_agents(agent)
    comp0 = W.greedy_acceptance(w2, a2, 0, np.random.default_rng(seed_eval))
    for c in range(40):
        if W.greedy_acceptance(w2, a2, 0, np.random.default_rng(seed_eval + c)) >= 0.95:
            recover = c * 250
            break
        W.step(w2, a2, None, 250, eta=0.0, lr=LR, eps=EPS, seed=sd())
    return dict(first500_accept=float(acc.mean()), competence_at_snap=comp0, steps_to_95=recover)


def run_seed(seed):
    sd = Seeder(seed)
    rows = dict(e1=[], raised=[], e8=[], e9=[])
    test = W.natural_test_set(np.random.default_rng(seed + 77))

    # E1 / C2 / C4 / C5: building under wear, the eta = 0 control, and the shaping-by-eye smoke test
    nat, built, builders, trace, secs = warm_builders(seed, W.WEAR, ETA, sd)
    _, ctrl, _, trace_c, _ = warm_builders(seed, W.WEAR, 0.0, sd)
    _, eye, _, trace_e, secs_e = warm_builders(seed, W.EYE, 1.0, sd)
    for name, w, tr, s in (("wear", built, trace, secs), ("control", ctrl, trace_c, None), ("eye", eye, trace_e, secs_e)):
        rows["e1"].append(dict(seed=seed, condition=name, built_share=w.built_share(), edge_share=w.edge_share(),
                               natural_edge_share=nat.edge_share(), trace=";".join(f"{x:.3f}" for x in tr), secs=s))
    frozen = W.copy_agents(builders)

    # E10 grid (and the agents E9 uses)
    raised = {}
    for kind, width, budget in itertools.product(("wild", "built"), WIDTHS, BUDGETS):
        t0 = time.time()
        agent, w_end = raise_agent(kind, width, budget, nat, built, frozen, seed, sd)
        sc = scores(agent, w_end, nat, test, seed + 900 + width + budget)
        rows["raised"].append(dict(seed=seed, kind=kind, width=width, budget=budget, secs=time.time() - t0, **sc))
        raised[(kind, width, budget)] = agent

    # E9: snap-back for the full-budget agents
    for kind, width in itertools.product(("wild", "built"), WIDTHS):
        res = snap_test(raised[(kind, width, 15000)], nat, sd, seed + 500 + width)
        rows["e9"].append(dict(seed=seed, kind=kind, width=width, **res))

    # E8: pretrain 15,000 natural steps (D2), then T2 in the built world (treatment) or natural world (control)
    for width in WIDTHS:
        pre = raised[("wild", width, 15000)]
        before = W.edge_scores(nat.body, nat.noise, pre, 0, test, np.random.default_rng(seed + 31))
        for arm in ("treatment", "control"):
            a = W.copy_agents(pre)
            if arm == "treatment":
                w = built.copy()
                W.step(w, a, frozen, T2, eta=ETA, lr=LR, eps=EPS, seed=sd())
            else:
                w = nat.copy()
                W.step(w, a, None, T2, eta=0.0, lr=LR, eps=EPS, seed=sd())
            after = W.edge_scores(nat.body, nat.noise, a, 0, test, np.random.default_rng(seed + 31))
            rows["e8"].append(dict(seed=seed, width=width, arm=arm, edge_before=before["edge_acc"],
                                   edge_after=after["edge_acc"], fall=before["edge_acc"] - after["edge_acc"],
                                   built_share_end=w.built_share() if arm == "treatment" else np.nan))
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    W.step(W.World(np.random.default_rng(0), n_items=20), W.make_agents(np.random.default_rng(0), 1, 8), None, 10, eta=0.1)
    t0 = time.time()
    with Pool(2) as p:
        out = p.map(run_seed, SEEDS)
    for part in ("e1", "raised", "e8", "e9"):
        pd.DataFrame([r for o in out for r in o[part]]).to_csv(os.path.join(OUT, f"{part}.csv"), index=False)
    print(f"done in {time.time() - t0:.0f}s")
