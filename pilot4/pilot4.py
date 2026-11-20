"""Pilot 4 runner (PILOT4_SPEC.md, PILOT4_LOG.md). Seeds 4000-4002; CSVs into results/.

Settings: lambda = 0.4, noise 0.01 (step 1); exploration as pilot 3; wear eta = 0.15 (step 2, D1); shaping by eye
s_u = 0.3 at kappa = 1 and 0.25; randomized priors beta = 1 (S5 rule); L8c dropped (S4).
Bodies per seed: maps 0-3 type 1, maps 4-7 type 2. Builders: map 0 (type 1) or map 4 (type 2).
"""
import itertools
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import world4 as W

OUT = "results"
SEEDS = (4000, 4001, 4002)
LAM, NOISE, LR = 0.4, 0.01, 0.05
M, BW, WARM = 6, 32, 30000
WIDTHS = (8, 32, 128)
BUDGETS = (30000, 60000)
COMP = 0.9
WEAR = dict(rule=W.WEAR, eta=0.15)
EYE1 = dict(rule=W.EYE, s_u=0.3, kappa=1.0)
EYE025 = dict(rule=W.EYE, s_u=0.3, kappa=0.25)
RULES = {"wear": WEAR, "eye1": EYE1, "eye025": EYE025}
EPS_R = (0.01, 0.02, 0.03, 0.05)
BETA = 1.0


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


def scores(B, ag, test, seed):
    out = W.edge_scores(B, ag, 0, test, np.random.default_rng(seed), noise=NOISE)
    out.update(W.boundary_shift(B, ag, 0, np.random.default_rng(seed + 1), noise=NOISE))
    return out


def cost_to(world, B, ag, frozen, kw, sd, cap=20000, seed=0):
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
    B = W.make_bodies(rng, [1, 1, 1, 1, 2, 2, 2, 2], LAM)
    test = W.natural_test_set(np.random.default_rng(seed + 77))
    out = {k: [] for k in ("q", "e3", "raised", "ens", "e9", "e8", "upd")}

    worlds = {("eye025", 0): build(nat, B, EYE025, 0, sd), ("eye025", 4): build(nat, B, EYE025, 4, sd),
              ("wear", 0): build(nat, B, WEAR, 0, sd), ("wear", 4): build(nat, B, WEAR, 4, sd),
              ("eye1", 0): build(nat, B, EYE1, 0, sd)}
    for (r, bb), (w, _) in worlds.items():
        out["q"].append(dict(seed=seed, world=r, builder_body=bb, built_share=w.built_share(), edge_share=w.edge_share(),
                             u1_move=w.u_move(0), u2_move=w.u_move(1)))
    out["q"].append(dict(seed=seed, world="natural", builder_body=-1, built_share=np.nan, edge_share=nat.edge_share(),
                         u1_move=nat.u_move(0), u2_move=nat.u_move(1)))

    # ---------------- E3: same body, three same-type maps, four outsider maps
    nat_cost, nat_ll, nat_code = {}, {}, {}
    for body in range(8):
        ag = agent(seed * 31 + body, 32, body); w = nat.copy()
        live(w, B, ag, None, None, 2000, sd); nat_ll[body] = W.log_loss(w, B, ag, 0, np.random.default_rng(seed + 5))
        nat_cost[body] = cost_to(nat.copy(), B, agent(seed * 37 + body, 32, body), None, None, sd, seed=seed)
        nat_code[body] = W.code_length_bits(nat, B, agent(seed * 41 + body, 32, body), 8000, sd(), lr=LR)
    for (r, bb) in [("eye025", 0), ("eye025", 4), ("wear", 0), ("wear", 4)]:
        w0, fr = worlds[(r, bb)]
        same_type = [b for b in range(8) if (b < 4) == (bb < 4) and b != bb]
        outsiders = [b for b in range(8) if (b < 4) != (bb < 4)]
        arms = [("same_body", bb)] + [("same_type", b) for b in same_type] + [("outsider", b) for b in outsiders]
        for arm, body in arms:
            ag = agent(seed * 43 + body + bb, 32, body); w = w0.copy()
            live(w, B, ag, fr, RULES[r], 2000, sd)
            ll2k = W.log_loss(w, B, ag, 0, np.random.default_rng(seed + 5))
            c, acts = cost_to(w0.copy(), B, agent(seed * 47 + body + bb, 32, body), fr, RULES[r], sd, seed=seed)
            h_nat, h_built = W.voronoi_entropy(B, body, nat, w0)
            code_built = W.code_length_bits(w0, B, agent(seed * 53 + body + bb, 32, body), 8000, sd(), lr=LR)
            out["e3"].append(dict(seed=seed, world=r, builder_body=bb, arm=arm, body=body,
                                  cost_natural=nat_cost[body][0], cost_built=c, acts_built=acts,
                                  ll2k_natural=nat_ll[body], ll2k_built=ll2k, h_natural=h_nat, h_built=h_built,
                                  code_natural=nat_code[body], code_built=code_built))

    # ---------------- E10 grid (single networks, body 0)
    raised = {}
    for kind, width, budget in itertools.product(("wild", "wear", "eye1"), WIDTHS, BUDGETS):
        ag = agent(seed * 97 + width + budget, width)
        if kind == "wild":
            w = nat.copy(); rec = live(w, B, ag, None, None, budget, sd)
        else:
            w0, fr = worlds[(kind, 0)]; w = w0.copy(); rec = live(w, B, ag, fr, RULES[kind], budget, sd)
        r_ = np.random.default_rng(seed + 900 + width)
        out["raised"].append(dict(seed=seed, kind=kind, width=width, budget=budget, eps_r=0.0, acts=int(rec["act"].sum()),
                                  home_logloss=W.log_loss(w, B, ag, 0, r_), competence_home=W.competence(w, B, ag, 0, r_),
                                  **scores(B, ag, test, seed + 901)))
        raised[(kind, width, budget)] = (ag, w, int(rec["act"].sum()))
    for kind, er in itertools.product(("wear", "eye1"), EPS_R):
        ag = agent(seed * 59 + int(er * 100), 32)
        w0, fr = worlds[(kind, 0)]; w = w0.copy(); rec = live(w, B, ag, fr, RULES[kind], 60000, sd, eps_r=er)
        r_ = np.random.default_rng(seed + 902)
        out["raised"].append(dict(seed=seed, kind=kind, width=32, budget=60000, eps_r=er, acts=int(rec["act"].sum()),
                                  home_logloss=W.log_loss(w, B, ag, 0, r_, eps_r=er), competence_home=W.competence(w, B, ag, 0, r_, eps_r=er),
                                  **scores(B, ag, test, seed + 901)))
    target = raised[("wear", 32, 60000)][2]
    ag = agent(seed * 97 + 32 + 60000, 32); w = nat.copy(); acts = steps = 0
    while acts < target and steps < 240000:
        acts += int(live(w, B, ag, None, None, 2000, sd)["act"].sum()); steps += 2000
    out["upd"].append(dict(seed=seed, target_acts=target, wild_acts=acts, wild_steps=steps,
                           wild60k_acts=raised[("wild", 32, 60000)][2], **scores(B, ag, test, seed + 901)))

    # ---------------- ensembles (width 32 base, 60k)
    variants = {"standard": ([32] * 5, 0.0), "diverse": ([6, 16, 32, 48, 64], 0.0), "prior": ([32] * 5, BETA)}
    for kind, (vname, (widths, beta)) in itertools.product(("wild", "wear", "eye1"), variants.items()):
        ens = W.make_ensemble(np.random.default_rng(seed * 7919 + len(vname)), 5, widths, body=0, beta=beta)
        if kind == "wild":
            w = nat.copy(); acts = W.step_ens(w, B, ens, None, 60000, seed=sd(), lr=LR)
        else:
            w0, fr = worlds[(kind, 0)]; w = w0.copy()
            acts = W.step_ens(w, B, ens, fr, 60000, seed=sd(), lr=LR, **RULES[kind])
        es = W.edge_scores_ens(B, ens, test, np.random.default_rng(seed + 903), noise=NOISE)
        es.update(W.boundary_shift_ens(B, ens, np.random.default_rng(seed + 904), noise=NOISE))
        out["ens"].append(dict(seed=seed, kind=kind, variant=vname, acts=int(acts.sum()),
                               home_logloss=W.ens_home_logloss(w, B, ens, np.random.default_rng(seed + 905)), **es))

    # ---------------- E9: snap-back for the 60k single networks
    for kind, width in itertools.product(("wild", "wear", "eye1"), WIDTHS):
        ag0, w_home, _ = raised[(kind, width, 60000)]
        home_ll = W.log_loss(w_home, B, ag0, 0, np.random.default_rng(seed + 3))
        ag = W.copy_agents(ag0); w = nat.copy(); w.snap_home()
        snap_ll = W.log_loss(w, B, ag, 0, np.random.default_rng(seed + 3))
        sc = []
        for k in range(3):
            sc.append(W.edge_scores(B, ag, 0, test, np.random.default_rng(seed + 4), noise=NOISE))
            if k < 2:
                live(w, B, ag, None, None, 250, sd)
        row = dict(seed=seed, kind=kind, width=width, home_logloss=home_ll, snap_logloss=snap_ll, d_nll=snap_ll - home_ll,
                   decisive_pay_mean=float(np.mean([x["decisive_pay"] for x in sc])))
        for t in ("wide", "narrow"):
            row[f"{t}_pay_mean"] = float(np.mean([x[f"{t}_decisive_pay"] for x in sc]))
            row[f"{t}_act_rate_0"] = sc[0][f"{t}_act_rate"]; row[f"{t}_refusals_per_act_0"] = sc[0][f"{t}_refusals_per_act"]
        out["e9"].append(row)

    # ---------------- E8 (L8c dropped after S4)
    for width in WIDTHS:
        ag = agent(seed * 211 + width, width); w = nat.copy()
        prev, steps, plateau = None, 0, None
        while steps < 80000:
            live(w, B, ag, None, None, 2000, sd); steps += 2000
            sc = W.edge_scores(B, ag, 0, test, np.random.default_rng(seed + 6), noise=NOISE)
            ea = 0.5 * (sc["wide_edge_acc"] + sc["narrow_edge_acc"])
            if prev is not None and abs(ea - prev) < 0.01:
                plateau = steps; break
            prev = ea
        before = scores(B, ag, test, seed + 7)
        wear_edge = worlds[("wear", 0)][0].edge_share()
        for arm in ("treat_wear", "treat_eye1", "filtered", "natural"):
            a = W.copy_agents(ag)
            if arm.startswith("treat"):
                wa, fr = worlds[(arm.split("_")[1], 0)]; wa = wa.copy(); kw = RULES[arm.split("_")[1]]
                acts = int(live(wa, B, a, fr, kw, 30000, sd)["act"].sum()); edge_seen = wa.edge_share()
            elif arm == "filtered":
                wa = nat.copy(); r = live(wa, B, a, None, None, 30000, sd, edge_share=wear_edge)
                acts = int(r["act"].sum()); edge_seen = float(W.is_edge(wa.alpha[r["item"]]).mean())
            else:
                wa = nat.copy(); acts = int(live(wa, B, a, None, None, 30000, sd)["act"].sum()); edge_seen = nat.edge_share()
            after = scores(B, a, test, seed + 7)
            out["e8"].append(dict(seed=seed, width=width, arm=arm, plateau_at=plateau, acts_T2=acts, edge_seen=edge_seen,
                                  edge_acc_before=0.5 * (before["wide_edge_acc"] + before["narrow_edge_acc"]),
                                  edge_acc_after=0.5 * (after["wide_edge_acc"] + after["narrow_edge_acc"]),
                                  shift_wide_before=before["shift_wide"], shift_wide_after=after["shift_wide"],
                                  shift_narrow_before=before["shift_narrow"], shift_narrow_after=after["shift_narrow"]))
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
