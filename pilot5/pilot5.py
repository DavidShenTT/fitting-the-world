"""Pilot 5 runner (PILOT5_SPEC.md, PILOT5_LOG.md). Seeds 5000-5002; CSVs into results/.

E10 only, single networks of width 32 raised for 60k steps, three agents per seed per cell.
Settings as pilot 4 (lambda 0.4, noise 0.01, lr 0.05, exploration m 0.2 / eps_m 0.6 / eps_min 0.02, wear eta 0.15,
shaping by eye s_u 0.3 at kappa 1). Dose levels s_u 0.1 and 0.03 (tuning step 1).
"""
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import world5 as W

OUT = "results"
SEEDS = (5000, 5001, 5002)
LAM, NOISE, LR = 0.4, 0.01, 0.05
M, BW, WARM, WIDTH, STEPS, N_AG = 6, 32, 30000, 32, 60000, 3
RULES = {"wear": dict(rule=W.WEAR, eta=0.15), "eye1": dict(rule=W.EYE, s_u=0.3, kappa=1.0),
         "dose010": dict(rule=W.EYE, s_u=0.1, kappa=1.0), "dose003": dict(rule=W.EYE, s_u=0.03, kappa=1.0)}
CONDS = {"skip": dict(), "passive": dict(passive=True), "s05": dict(passive=True, target_share=0.5),
         "s09": dict(passive=True, target_share=0.9)}
PLAN = [  # (geometry, kinds, conditions)
    ("unequal", ("wild", "wear", "eye1"), ("skip", "passive", "s05", "s09")),
    ("unequal", ("dose010", "dose003"), ("skip", "passive")),
    ("symmetric", ("wild", "wear"), ("skip", "passive", "s05")),
]


class Seeder:
    def __init__(self, base):
        self.n = base * 10000

    def __call__(self):
        self.n += 1
        return self.n


def build(nat, B, kw, sd):
    w = nat.copy()
    bl = W.make_agents(np.random.default_rng(sd()), M, BW, 0)
    W.step(w, B, bl, None, WARM, seed=sd(), learner_shapes=True, lr=LR, **kw)
    return w, W.copy_agents(bl)


def raise_agent(world, B, frozen, kw, cond, sd, test):
    ag = W.make_agents(np.random.default_rng(sd()), 1, WIDTH, 0)
    w = world.copy()
    fr = W.copy_agents(frozen) if frozen is not None else None
    r = W.step(w, B, ag, fr, STEPS, seed=sd(), lr=LR, **(kw or {}), **CONDS[cond])
    s = sd()
    row = W.edge_scores(B, ag, 0, test, np.random.default_rng(s), noise=NOISE)
    row.update(W.boundary_shift(B, ag, 0, np.random.default_rng(s + 1), noise=NOISE))
    row.update(home_logloss=W.log_loss(w, B, ag, 0, np.random.default_rng(s + 2)),
               share_wide=W.label_share(r, W.WIDE), share_narrow=W.label_share(r, W.NARROW),
               updates=int(r["upd"].sum()), acts=int(r["act"].sum()))
    return row


def offered_edge_share(world, B, ag, frozen, kw, sd, steps=30000, chunk=50, **extra):
    seen = []
    for _ in range(steps // chunk):
        r = W.step(world, B, ag, frozen, chunk, seed=sd(), lr=LR, chunk=chunk, **(kw or {}), **extra)
        seen.append(W.is_edge(world.alpha[r["item"]]))
    return float(np.concatenate(seen).mean())


def run_seed(seed):
    t0 = time.time()
    sd = Seeder(seed)
    rows, q = [], []
    for geom_name, kinds, conds in PLAN:
        W.set_geometry(W.UNEQUAL if geom_name == "unequal" else W.SYMMETRIC)
        rng = np.random.default_rng(seed)
        nat = W.World(rng, noise=NOISE)
        B = W.make_bodies(rng, [1, 1, 1, 1, 2, 2, 2, 2], LAM)
        test = W.natural_test_set(np.random.default_rng(seed + 77))
        worlds = {}
        for kind in kinds:
            if kind == "wild":
                worlds[kind] = (nat, None, None)
                q.append(dict(seed=seed, geometry=geom_name, kind=kind, built_share=np.nan, edge_share=nat.edge_share()))
            else:
                w, fr = build(nat, B, RULES[kind], sd)
                worlds[kind] = (w, fr, RULES[kind])
                q.append(dict(seed=seed, geometry=geom_name, kind=kind, built_share=w.built_share(), edge_share=w.edge_share()))
        for kind in kinds:
            w, fr, kw = worlds[kind]
            for cond in conds:
                for k in range(N_AG):
                    row = raise_agent(w, B, fr, kw, cond, sd, test)
                    row.update(seed=seed, geometry=geom_name, kind=kind, cond=cond, agent=k)
                    rows.append(row)
        if geom_name == "unequal" and "wear" in worlds:  # P5: offered edge share, built world vs redrawn filtered sampler
            w, fr, kw = worlds["wear"]
            treat = offered_edge_share(w.copy(), B, W.make_agents(np.random.default_rng(sd()), 1, WIDTH, 0), W.copy_agents(fr), kw, sd)
            filt = offered_edge_share(nat.copy(), B, W.make_agents(np.random.default_rng(sd()), 1, WIDTH, 0), None, None, sd,
                                      edge_share=treat)
            q.append(dict(seed=seed, geometry="p5", kind="treat_wear", built_share=np.nan, edge_share=treat))
            q.append(dict(seed=seed, geometry="p5", kind="filtered", built_share=np.nan, edge_share=filt))
    W.set_geometry(W.UNEQUAL)
    q.append(dict(seed=seed, geometry="seconds", kind="", built_share=time.time() - t0, edge_share=np.nan))
    print(f"seed {seed} done in {time.time() - t0:.0f}s", flush=True)
    return rows, q


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with Pool(len(SEEDS)) as pool:
        res = pool.map(run_seed, SEEDS)
    pd.DataFrame([r for rows, _ in res for r in rows]).to_csv(f"{OUT}/raised.csv", index=False)
    pd.DataFrame([r for _, q in res for r in q]).to_csv(f"{OUT}/q.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")
