"""Pilot 6 runner (PILOT6_SPEC.md, PILOT6_LOG.md). Seeds 6000-6005; CSVs into results/.

E10 only, unequal world: wild, wear (eta 0.15) and shaping by eye (kappa 1, s_u 0.3) agents, single networks of width 32
raised for 60k steps, two agents per seed per cell. Conditions: skip rule at eps_min 0.02 / 0.1 / 0.3, passive feedback,
biased offers at s = 0.9. Angle histograms of the unequal and symmetric worlds (no agents in the symmetric world).
"""
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import world6 as W

OUT = "results"
SEEDS = (6000, 6001, 6002, 6003, 6004, 6005)
LAM, NOISE, LR = 0.4, 0.01, 0.05
M, BW, WARM, WIDTH, STEPS, N_AG = 6, 32, 30000, 32, 60000, 2
RULES = {"wear": dict(rule=W.WEAR, eta=0.15), "eye1": dict(rule=W.EYE, s_u=0.3, kappa=1.0)}
EXP = lambda e: dict(eps_min=e, eps_m=0.6, m=0.2)
CONDS = {"skip002": dict(explore=EXP(0.02)), "skip010": dict(explore=EXP(0.1)), "skip030": dict(explore=EXP(0.3)),
         "passive": dict(passive=True), "biased09": dict(passive=True, offer_share=0.9)}
KINDS = ("wild", "wear", "eye1")


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
               share_all=float(r["y"][r["upd"] == 1].mean()),
               share_wide=W.label_share(r, W.WIDE), share_narrow=W.label_share(r, W.NARROW),
               updates=int(r["upd"].sum()), acts=int(r["act"].sum()))
    for name, acts in (("wide", W.WIDE), ("narrow", W.NARROW)):
        row.update({f"{name}_{k}": v for k, v in W.refusal_sources(r, acts).items()})
    return row


def run_seed(seed):
    t0 = time.time()
    sd = Seeder(seed)
    rows, q, hist = [], [], []
    W.set_geometry(W.UNEQUAL)
    rng = np.random.default_rng(seed)
    nat = W.World(rng, noise=NOISE)
    B = W.make_bodies(rng, [1, 1, 1, 1, 2, 2, 2, 2], LAM)
    test = W.natural_test_set(np.random.default_rng(seed + 77))
    worlds = {"wild": (nat, None, None)}
    for kind in ("wear", "eye1"):
        w, fr = build(nat, B, RULES[kind], sd)
        worlds[kind] = (w, fr, RULES[kind])
    for kind in KINDS:
        w = worlds[kind][0]
        q.append(dict(seed=seed, kind=kind, built_share=w.built_share() if kind != "wild" else np.nan, edge_share=w.edge_share()))
        hist.append(dict(seed=seed, geometry="unequal", kind=kind, **{f"b{5 * k:02d}": v for k, v in enumerate(W.angle_histogram(w))}))
    for kind in KINDS:
        w, fr, kw = worlds[kind]
        for cond in CONDS:
            for k in range(N_AG):
                row = raise_agent(w, B, fr, kw, cond, sd, test)
                row.update(seed=seed, kind=kind, cond=cond, agent=k)
                rows.append(row)
    W.set_geometry(W.SYMMETRIC)  # histogram only
    rng = np.random.default_rng(seed)
    nat_s = W.World(rng, noise=NOISE)
    B_s = W.make_bodies(rng, [1, 1, 1, 1, 2, 2, 2, 2], LAM)
    wear_s, _ = build(nat_s, B_s, RULES["wear"], sd)
    for kind, w in (("wild", nat_s), ("wear", wear_s)):
        hist.append(dict(seed=seed, geometry="symmetric", kind=kind, **{f"b{5 * k:02d}": v for k, v in enumerate(W.angle_histogram(w))}))
    W.set_geometry(W.UNEQUAL)
    q.append(dict(seed=seed, kind="seconds", built_share=time.time() - t0, edge_share=np.nan))
    print(f"seed {seed} done in {time.time() - t0:.0f}s", flush=True)
    return rows, q, hist


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with Pool(len(SEEDS)) as pool:
        res = pool.map(run_seed, SEEDS)
    pd.DataFrame([r for rows, _, _ in res for r in rows]).to_csv(f"{OUT}/raised.csv", index=False)
    pd.DataFrame([r for _, q, _ in res for r in q]).to_csv(f"{OUT}/q.csv", index=False)
    pd.DataFrame([r for _, _, h in res for r in h]).to_csv(f"{OUT}/hist.csv", index=False)
    print(f"done in {time.time() - t0:.0f}s")
