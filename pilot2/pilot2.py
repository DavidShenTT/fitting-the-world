"""Pilot 2 runner (PILOT2_SPEC.md, PILOT2_LOG.md): checks Q1-Q6 and readouts for E3, E8, E9, E10.

Seeds 2000-2002. Writes CSVs into results/. Settings follow the log: eps = 0.2, lr = 0.05 (D3);
competence 0.75 for learning cost, budgets 10k / 30k, E8 pretraining cap 40k and T2 = 20k (D4).
Building settings from the Q4 sweep: wear eta = 0.15; eye s_u = 0.3 at kappa = 1 and 0.25.
"""
import itertools
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import world2 as W

OUT = "results"
SEEDS = (2000, 2001, 2002)
EPS, LR = 0.2, 0.05
M, BW, WARM = 6, 32, 30000
WIDTHS = (8, 32, 128)
BUDGETS = (10000, 30000)
COMP = 0.75
RULES = {"wear": dict(rule=W.WEAR, eta=0.15), "eye1": dict(rule=W.EYE, s_u=0.3, kappa=1.0),
         "eye025": dict(rule=W.EYE, s_u=0.3, kappa=0.25)}
EPS_R = (0.05, 0.1, 0.2)


class Seeder:
    def __init__(self, base):
        self.n = base * 10000

    def __call__(self):
        self.n += 1
        return self.n


def new_agent(seed, width, body=0):
    return W.make_agents(np.random.default_rng(seed), 1, width, body)


def build(nat, B, kw, sd):
    w = nat.copy()
    bl = W.make_agents(np.random.default_rng(sd()), M, BW, 0)
    W.step(w, B, bl, None, WARM, seed=sd(), learner_shapes=True, eps=EPS, lr=LR, **kw)
    return w, W.copy_agents(bl)


def live(world, B, agent, frozen, kw, steps, sd, eps_r=0.0, **extra):
    """The agent learns in `world`; frozen builders (if any) keep shaping with rule `kw`; the agent doesn't shape."""
    kw = kw or {}
    return W.step(world, B, agent, frozen, steps, seed=sd(), eps=EPS, lr=LR, eps_r=eps_r, **kw, **extra)


def run_seed(seed):
    t0 = time.time()
    sd = Seeder(seed)
    rng = np.random.default_rng(seed)
    nat = W.World(rng)
    B = W.make_bodies(rng, 3)  # body 0: builders; 1 and 2: outsiders
    test = W.natural_test_set(np.random.default_rng(seed + 77))
    out = dict(e3=[], forestry=[], raised=[], e9=[], e8=[], ens=[], q=[])

    worlds = {name: build(nat, B, kw, sd) for name, kw in RULES.items()}
    for name, (w, _) in worlds.items():
        out["q"].append(dict(seed=seed, item="built_share", world=name, value=w.built_share()))
        out["q"].append(dict(seed=seed, item="edge_share", world=name, value=w.edge_share()))
    out["q"].append(dict(seed=seed, item="edge_share", world="natural", value=nat.edge_share()))

    # ---------------- E3: who benefits; L3c forestry
    for body in (0, 1, 2):
        for wname in ("natural",) + tuple(RULES):
            if wname == "natural":
                w, fr, kw = nat.copy(), None, None
            else:
                w, fr = worlds[wname][0].copy(), worlds[wname][1]
                kw = RULES[wname]
            ag = new_agent(seed * 31 + body, 32, body)
            live(w, B, ag, fr, kw, 2000, sd)
            ll2k = W.log_loss(w, B, ag, 0, np.random.default_rng(seed + 5))
            cost = None
            steps = 2000
            while steps <= 30000:
                if W.competence(w, B, ag, 0, np.random.default_rng(seed + steps)) >= COMP:
                    cost = steps
                    break
                live(w, B, ag, fr, kw, 500, sd)
                steps += 500
            out["e3"].append(dict(seed=seed, body=body, world=wname, logloss_2k=ll2k, cost=cost))
    for wname in RULES:
        for body in (0, 1, 2):
            h_nat, h_built = W.forestry_entropy(B, body, nat, worlds[wname][0])
            out["forestry"].append(dict(seed=seed, world=wname, body=body, h_natural=h_nat, h_built=h_built,
                                        drop=h_nat - h_built))

    # ---------------- E10 grid (single networks) and the agents E9 uses
    raised = {}
    for kind, width, budget in itertools.product(("wild",) + tuple(RULES), WIDTHS, BUDGETS):
        ag = new_agent(seed * 97 + width + budget, width)
        if kind == "wild":
            w = nat.copy(); live(w, B, ag, None, None, budget, sd)
        else:
            w = worlds[kind][0].copy(); live(w, B, ag, worlds[kind][1], RULES[kind], budget, sd)
        r = np.random.default_rng(seed + 900 + width)
        es = W.edge_scores(B, ag, 0, test, r)
        out["raised"].append(dict(seed=seed, kind=kind, width=width, budget=budget, eps_r=0.0,
                                  home_logloss=W.log_loss(w, B, ag, 0, r), competence_home=W.competence(w, B, ag, 0, r), **es))
        raised[(kind, width, budget)] = (ag, w)

    # Q3: random refusal in built worlds (width 32, full budget), home log loss only for the choice
    for kind, er in itertools.product(("wear", "eye1"), EPS_R):
        ag = new_agent(seed * 53 + int(er * 100), 32)
        w = worlds[kind][0].copy(); live(w, B, ag, worlds[kind][1], RULES[kind], 30000, sd, eps_r=er)
        r = np.random.default_rng(seed + 901)
        es = W.edge_scores(B, ag, 0, test, r)
        out["raised"].append(dict(seed=seed, kind=kind, width=32, budget=30000, eps_r=er,
                                  home_logloss=W.log_loss(w, B, ag, 0, r, eps_r=er), competence_home=W.competence(w, B, ag, 0, r, eps_r=er), **es))

    # ensembles (width 32, full budget): five independently raised networks
    for kind in ("wild", "wear", "eye1"):
        members = []
        for m in range(5):
            ag = new_agent(seed * 7919 + m, 32)
            if kind == "wild":
                w = nat.copy(); live(w, B, ag, None, None, 30000, sd)
            else:
                w = worlds[kind][0].copy(); live(w, B, ag, worlds[kind][1], RULES[kind], 30000, sd)
            members.append(ag)
        es = W.edge_scores_ensemble(B, members, test, np.random.default_rng(seed + 902))
        out["ens"].append(dict(seed=seed, kind=kind, **es))

    # ---------------- E9: snap-back, full-budget agents
    for kind, width in itertools.product(("wild",) + tuple(RULES), WIDTHS):
        ag0, w_home = raised[(kind, width, 30000)]
        home_ll = W.log_loss(w_home, B, ag0, 0, np.random.default_rng(seed + 3))
        ag = W.copy_agents(ag0)
        w = nat.copy(); w.snap_home()
        snap_ll = W.log_loss(w, B, ag, 0, np.random.default_rng(seed + 3))
        pays = []
        for k in range(3):
            pays.append(W.edge_scores(B, ag, 0, test, np.random.default_rng(seed + 4))["decisive_pay"])
            if k < 2:
                live(w, B, ag, None, None, 250, sd)
        out["e9"].append(dict(seed=seed, kind=kind, width=width, decisive_pay_0=pays[0], decisive_pay_500=pays[2],
                              decisive_pay_mean=float(np.mean(pays)), home_logloss=home_ll, snap_logloss=snap_ll,
                              d_nll=snap_ll - home_ll))

    # ---------------- E8: pretrain to an edge-accuracy plateau, then three arms with kind-B items
    for width in WIDTHS:
        ag = new_agent(seed * 211 + width, width)
        w = nat.copy()
        prev, steps, plateau_at = None, 0, None
        while steps < 40000:
            live(w, B, ag, None, None, 2000, sd); steps += 2000
            ea = W.edge_scores(B, ag, 0, test, np.random.default_rng(seed + 6))["edge_acc"]
            if prev is not None and abs(ea - prev) < 0.01:
                plateau_at = steps
                break
            prev = ea
        before = W.edge_scores(B, ag, 0, test, np.random.default_rng(seed + 7))["edge_acc"]
        for arm in ("treat_wear", "treat_eye025", "filtered", "natural"):
            a = W.copy_agents(ag)
            if arm.startswith("treat"):
                kname = arm.split("_")[1]
                wa = worlds[kname][0].copy(); fr = worlds[kname][1]; kw = RULES[kname]
            else:
                wa = nat.copy(); fr = None; kw = None
            wa.add_kind_b(np.random.default_rng(seed + 8), 300)
            b_cost = None
            for c in range(20):
                live(wa, B, a, fr, kw, 1000, sd, pB=0.3, filtered=(arm == "filtered"))
                if b_cost is None and W.competence(wa, B, a, 0, np.random.default_rng(seed + 10 + c), kind=1) >= COMP:
                    b_cost = (c + 1) * 1000
            after = W.edge_scores(B, a, 0, test, np.random.default_rng(seed + 7))["edge_acc"]
            out["e8"].append(dict(seed=seed, width=width, arm=arm, plateau_at=plateau_at, edge_before=before,
                                  edge_after=after, fall=before - after, b_cost=b_cost,
                                  b_competence_end=W.competence(wa, B, a, 0, np.random.default_rng(seed + 11), kind=1),
                                  built_share_end=wa.built_share() if arm.startswith("treat") else np.nan))
    out["q"].append(dict(seed=seed, item="seconds", world="all", value=time.time() - t0))
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    w = W.World(np.random.default_rng(0), n_items=20)
    W.step(w, W.make_bodies(np.random.default_rng(0), 1), W.make_agents(np.random.default_rng(0), 1, 8), None, 10, seed=1)
    t0 = time.time()
    with Pool(2) as p:
        res = p.map(run_seed, SEEDS)
    for part in res[0]:
        pd.DataFrame([r for o in res for r in o[part]]).to_csv(os.path.join(OUT, f"{part}.csv"), index=False)
    print("done in %.0fs" % (time.time() - t0))
