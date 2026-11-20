"""Pilot 3 runner (PILOT3_SPEC.md, PILOT3_LOG.md). Seeds 3000-3002; CSVs into results/.

Settings from the log: lambda = 0.4, observation noise 0.01 (D1); exploration m = 0.2, eps_m = 0.6,
eps_min = 0.02, lr = 0.05 (step 2); budgets 30k / 60k, E8 pretraining cap 80k, T2 = 30k (D2);
wear eta = 0.15, shaping by eye s_u = 0.3 at kappa = 1 and 0.25 (step 3).
Bodies per seed: 0 and 1 are type-1 maps, 2 and 3 type-2 maps.
"""
import itertools
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import world3 as W

OUT = "results"
SEEDS = (3000, 3001, 3002)
LAM, NOISE, LR = 0.4, 0.01, 0.05
M, BW, WARM = 6, 32, 30000
WIDTHS = (8, 32, 128)
BUDGETS = (30000, 60000)
COMP = 0.9
WEAR = dict(rule=W.WEAR, eta=0.15)
EYE1 = dict(rule=W.EYE, s_u=0.3, kappa=1.0)
EYE025 = dict(rule=W.EYE, s_u=0.3, kappa=0.25)
EPS_R = (0.05, 0.1, 0.2)


class Seeder:
    def __init__(self, base):
        self.n = base * 10000

    def __call__(self):
        self.n += 1
        return self.n


def agent(seed, width, body=0):
    return W.make_agents(np.random.default_rng(seed), 1, width, body)


def build(nat, B, kw, body, sd):
    w = nat.copy()
    bl = W.make_agents(np.random.default_rng(sd()), M, BW, body)
    W.step(w, B, bl, None, WARM, seed=sd(), learner_shapes=True, lr=LR, **kw)
    return w, W.copy_agents(bl)


def live(world, B, ag, frozen, kw, steps, sd, **extra):
    return W.step(world, B, ag, frozen, steps, seed=sd(), lr=LR, **(kw or {}), **extra)


def edge(B, ag, test, seed):
    return W.edge_scores(B, ag, 0, test, np.random.default_rng(seed), noise=NOISE)


def cost_to(world, B, ag, frozen, kw, sd, cap=20000, seed=0):
    """Steps to competence 0.9, checked every 250 steps from step 0; also acted turns used."""
    acts = 0
    for s in range(0, cap + 1, 250):
        if s > 0 and W.competence(world, B, ag, 0, np.random.default_rng(seed + s)) >= COMP:
            return s, acts
        if s < cap:
            acts += int(live(world, B, ag, frozen, kw, 250, sd)["act"].sum())
    return None, acts


def run_seed(seed):
    t0 = time.time()
    sd = Seeder(seed)
    rng = np.random.default_rng(seed)
    nat = W.World(rng, noise=NOISE)
    B = W.make_bodies(rng, [1, 1, 2, 2], LAM)
    test = W.natural_test_set(np.random.default_rng(seed + 77))
    out = {k: [] for k in ("q", "e3", "raised", "ens", "e9", "e8", "upd")}

    # ---------------- worlds
    worlds = {("eye025", 0): build(nat, B, EYE025, 0, sd), ("eye025", 2): build(nat, B, EYE025, 2, sd),
              ("wear", 0): build(nat, B, WEAR, 0, sd), ("wear", 2): build(nat, B, WEAR, 2, sd),
              ("eye1", 0): build(nat, B, EYE1, 0, sd)}
    rules = {"eye025": EYE025, "wear": WEAR, "eye1": EYE1}
    for (r, bb), (w, _) in worlds.items():
        out["q"].append(dict(seed=seed, world=r, builder_body=bb, built_share=w.built_share(), edge_share=w.edge_share(),
                             u1_move=w.u_move(0), u2_move=w.u_move(1)))
    out["q"].append(dict(seed=seed, world="natural", builder_body=-1, built_share=np.nan, edge_share=nat.edge_share(),
                         u1_move=nat.u_move(0), u2_move=nat.u_move(1)))

    # ---------------- E3
    nat_cost, nat_ll, nat_code = {}, {}, {}
    for body in range(4):
        ag = agent(seed * 31 + body, 32, body); w = nat.copy()
        live(w, B, ag, None, None, 2000, sd); nat_ll[body] = W.log_loss(w, B, ag, 0, np.random.default_rng(seed + 5))
        nat_cost[body] = cost_to(nat.copy(), B, agent(seed * 37 + body, 32, body), None, None, sd, seed=seed)
        nb = agent(seed * 41 + body, 32, body)
        nat_code[body] = float(live(nat.frozen(), B, nb, None, None, 2000, sd)["ll"].mean())
    for (r, bb) in [("eye025", 0), ("eye025", 2), ("wear", 0), ("wear", 2)]:
        w0, fr = worlds[(r, bb)]
        arms = {"same_body": bb, "same_type": bb + 1, "outsider": 2 if bb == 0 else 0}
        for arm, body in arms.items():
            ag = agent(seed * 43 + body + bb, 32, body); w = w0.copy()
            live(w, B, ag, fr, rules[r], 2000, sd)
            ll2k = W.log_loss(w, B, ag, 0, np.random.default_rng(seed + 5))
            c, acts = cost_to(w0.copy(), B, agent(seed * 47 + body + bb, 32, body), fr, rules[r], sd, seed=seed)
            h_nat, h_built = W.voronoi_entropy(B, body, nat, w0)
            nb = agent(seed * 53 + body + bb, 32, body)
            code_built = float(live(w0.frozen(), B, nb, None, None, 2000, sd)["ll"].mean())
            out["e3"].append(dict(seed=seed, world=r, builder_body=bb, arm=arm, body=body,
                                  cost_natural=nat_cost[body][0], cost_built=c, acts_natural=nat_cost[body][1], acts_built=acts,
                                  ll2k_natural=nat_ll[body], ll2k_built=ll2k, h_natural=h_nat, h_built=h_built,
                                  code_natural=nat_code[body], code_built=code_built))

    # ---------------- E10 grid (single networks, body 0)
    raised = {}
    for kind, width, budget in itertools.product(("wild", "wear", "eye1"), WIDTHS, BUDGETS):
        ag = agent(seed * 97 + width + budget, width)
        if kind == "wild":
            w = nat.copy(); rec = live(w, B, ag, None, None, budget, sd)
        else:
            w0, fr = worlds[(kind, 0)]; w = w0.copy(); rec = live(w, B, ag, fr, rules[kind], budget, sd)
        r_ = np.random.default_rng(seed + 900 + width)
        out["raised"].append(dict(seed=seed, kind=kind, width=width, budget=budget, eps_r=0.0, acts=int(rec["act"].sum()),
                                  home_logloss=W.log_loss(w, B, ag, 0, r_), competence_home=W.competence(w, B, ag, 0, r_),
                                  **edge(B, ag, test, seed + 901)))
        raised[(kind, width, budget)] = (ag, w, int(rec["act"].sum()))
    for kind, er in itertools.product(("wear", "eye1"), EPS_R):
        ag = agent(seed * 59 + int(er * 100), 32)
        w0, fr = worlds[(kind, 0)]; w = w0.copy(); rec = live(w, B, ag, fr, rules[kind], 60000, sd, eps_r=er)
        r_ = np.random.default_rng(seed + 902)
        out["raised"].append(dict(seed=seed, kind=kind, width=32, budget=60000, eps_r=er, acts=int(rec["act"].sum()),
                                  home_logloss=W.log_loss(w, B, ag, 0, r_, eps_r=er), competence_home=W.competence(w, B, ag, 0, r_, eps_r=er),
                                  **edge(B, ag, test, seed + 901)))

    # update-count check: a wild agent run until its acted turns match the built-raised (wear, width 32, 60k) agent's
    target = raised[("wear", 32, 60000)][2]
    ag = agent(seed * 97 + 32 + 60000, 32); w = nat.copy(); acts = steps = 0
    while acts < target and steps < 240000:
        acts += int(live(w, B, ag, None, None, 2000, sd)["act"].sum()); steps += 2000
    out["upd"].append(dict(seed=seed, target_acts=target, wild_acts=acts, wild_steps=steps,
                           wild60k_acts=raised[("wild", 32, 60000)][2], **edge(B, ag, test, seed + 901)))

    # ---------------- ensembles (width 32 base, 60k)
    variants = {"standard": ([32] * 5, 0.0), "diverse": ([6, 16, 32, 48, 64], 0.0), "prior": ([32] * 5, 1.0)}
    for kind, (vname, (widths, beta)) in itertools.product(("wild", "wear", "eye1"), variants.items()):
        ens = W.make_ensemble(np.random.default_rng(seed * 7919 + len(vname)), 5, widths, body=0, beta=beta)
        if kind == "wild":
            w = nat.copy(); acts = W.step_ens(w, B, ens, None, 60000, seed=sd(), lr=LR)
        else:
            w0, fr = worlds[(kind, 0)]; w = w0.copy()
            acts = W.step_ens(w, B, ens, fr, 60000, seed=sd(), lr=LR, **rules[kind])
        es = W.edge_scores_ens(B, ens, test, np.random.default_rng(seed + 903), noise=NOISE)
        out["ens"].append(dict(seed=seed, kind=kind, variant=vname, acts=int(acts.sum()),
                               home_logloss=W.ens_home_logloss(w, B, ens, np.random.default_rng(seed + 904)), **es))

    # ---------------- E9: snap-back for the 60k single networks
    for kind, width in itertools.product(("wild", "wear", "eye1"), WIDTHS):
        ag0, w_home, _ = raised[(kind, width, 60000)]
        home_ll = W.log_loss(w_home, B, ag0, 0, np.random.default_rng(seed + 3))
        ag = W.copy_agents(ag0); w = nat.copy(); w.snap_home()
        snap_ll = W.log_loss(w, B, ag, 0, np.random.default_rng(seed + 3))
        sc = []
        for k in range(3):
            sc.append(edge(B, ag, test, seed + 4))
            if k < 2:
                live(w, B, ag, None, None, 250, sd)
        out["e9"].append(dict(seed=seed, kind=kind, width=width, home_logloss=home_ll, snap_logloss=snap_ll, d_nll=snap_ll - home_ll,
                              decisive_pay_0=sc[0]["decisive_pay"], decisive_pay_500=sc[2]["decisive_pay"],
                              decisive_pay_mean=float(np.mean([x["decisive_pay"] for x in sc])),
                              act_rate_0=sc[0]["act_rate"], refusals_per_act_0=sc[0]["refusals_per_act"]))

    # ---------------- E8
    for width in WIDTHS:
        ag = agent(seed * 211 + width, width); w = nat.copy()
        prev, steps, plateau = None, 0, None
        while steps < 80000:
            live(w, B, ag, None, None, 2000, sd); steps += 2000
            ea = edge(B, ag, test, seed + 6)["edge_acc"]
            if prev is not None and abs(ea - prev) < 0.01:
                plateau = steps; break
            prev = ea
        before = edge(B, ag, test, seed + 7)["edge_acc"]
        n1, n2 = np.linalg.norm(ag["W1"]), np.linalg.norm(ag["W2"])
        for arm in ("treat_wear", "treat_eye1", "filtered", "natural"):
            a = W.copy_agents(ag)
            if arm.startswith("treat"):
                wa, fr = worlds[(arm.split("_")[1], 0)]; wa = wa.copy(); kw = rules[arm.split("_")[1]]
            else:
                wa, fr, kw = nat.copy(), None, None
            wa.add_kind_b(np.random.default_rng(seed + 8), 300)
            comps, acts = [], 0
            for c in range(30):
                acts += int(live(wa, B, a, fr, kw, 1000, sd, pB=0.3, filtered=(arm == "filtered"))["act"].sum())
                comps.append(W.competence(wa, B, a, 0, np.random.default_rng(seed + 10 + c), kind=1))
            out["e8"].append(dict(seed=seed, width=width, arm=arm, plateau_at=plateau, edge_before=before,
                                  edge_after=edge(B, a, test, seed + 7)["edge_acc"], acts_T2=acts,
                                  b_auc=float(np.nanmean(comps)), b_competence_end=comps[-1],
                                  probe_b=W.linear_probe(B, a, wa, np.random.default_rng(seed + 12)),
                                  w1_norm_before=n1, w1_norm_after=np.linalg.norm(a["W1"]),
                                  w2_norm_before=n2, w2_norm_after=np.linalg.norm(a["W2"]),
                                  built_share_end=wa.built_share() if arm.startswith("treat") else np.nan))
    out["q"].append(dict(seed=seed, world="seconds", builder_body=-1, built_share=time.time() - t0, edge_share=np.nan,
                         u1_move=np.nan, u2_move=np.nan))
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    w = W.World(np.random.default_rng(0), n_items=20)
    Bq = W.make_bodies(np.random.default_rng(0), [1], LAM)
    W.step(w, Bq, W.make_agents(np.random.default_rng(0), 1, 8), None, 10, seed=1)
    W.step_ens(w, Bq, W.make_ensemble(np.random.default_rng(0), 2, [4, 8]), None, 10, seed=1)
    t0 = time.time()
    with Pool(2) as p:
        res = p.map(run_seed, SEEDS)
    for part in res[0]:
        pd.DataFrame([r for o in res for r in o[part]]).to_csv(os.path.join(OUT, f"{part}.csv"), index=False)
    print("done in %.0fs" % (time.time() - t0), flush=True)
