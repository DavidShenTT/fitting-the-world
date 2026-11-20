"""Experiment tasks (PREREGISTRATION.md). Each task is a tuple; run_task(exp, task) returns {table: [row dicts]}.
Every random stream is derived from (seed, tag, ...) so that a world built in one task is identical in another."""
import zlib

import numpy as np

import config as C
import world as W

GEOM = {"unequal": W.UNEQUAL, "symmetric": W.SYMMETRIC}
CONDS = {"skip": {}, "passive": dict(passive=True),
         "skip010": dict(explore=dict(C.EXPLORE, eps_min=0.1)), "skip030": dict(explore=dict(C.EXPLORE, eps_min=0.3))}


# ------------------------------------------------------------------ seeding

def _ints(key):
    return [k if isinstance(k, (int, np.integer)) and k >= 0 else zlib.crc32(repr(k).encode()) for k in key]


def sseed(*key):
    """A step seed (numba reseeds with seed * 1000 + chunk, so keep it below 4e6)."""
    return int(np.random.SeedSequence(_ints(key)).generate_state(1)[0] % 4_000_000)


def rng(*key):
    return np.random.default_rng(np.random.SeedSequence(_ints(key)))


class Seeder:
    def __init__(self, *key):
        self.key, self.n = key, 0

    def __call__(self):
        self.n += 1
        return sseed(*self.key, self.n)


# ------------------------------------------------------------------ shared pieces

def setup(seed, geom, lam=None):
    W.set_geometry(GEOM[geom])
    r = rng(seed, "world")
    nat = W.World(r, n_items=C.N_ITEMS, delta=C.DELTA, drift_a=C.DRIFT_A, drift_u=C.DRIFT_U, noise=C.NOISE)
    B = W.make_bodies(r, C.BODY_TYPES, C.LAM[geom] if lam is None else lam)
    test = W.natural_test_set(rng(seed, "test"))
    return nat, B, test


def world_row(w, nat, **key):
    return dict(**key, built_share=w.built_share(), built_share_overlap=w.built_share_overlap(),
                built_count_overlap=w.built_count_overlap(), edge_share=w.edge_share(),
                edge_share_natural=nat.edge_share(), u1_move=w.u_move(0), u2_move=w.u_move(1))


def build(nat, B, rule, seed, tag, body=0, M=C.M_BUILD, warm=C.WARM, width=C.BUILD_WIDTH, every=None,
          delta=None, record=None):
    """M builders on one map learn and shape from random init for `warm` steps. Returns (world, frozen copies, traj)."""
    w = nat.copy()
    if delta is not None:
        w.delta = delta
    bl = W.make_agents(rng(seed, "builders", tag), M, width, body)
    sd = Seeder(seed, "build", tag)
    traj, done = [], 0
    chunk = every or warm
    while done < warm:
        n = min(chunk, warm - done)
        W.step(w, B, bl, None, n, seed=sd(), learner_shapes=True, lr=C.LR, explore=C.EXPLORE, **rule)
        done += n
        if record is not None:
            traj.append(dict(step=done, **record(w, bl)))
    return w, W.copy_agents(bl), traj


def live(world, B, ag, frozen, rule, steps, seed, cond="skip", eps_r=0.0, **extra):
    kw = dict(rule) if frozen is not None else {}
    return W.step(world, B, ag, frozen, steps, seed=seed, lr=C.LR, eps_r=eps_r, **kw, **CONDS[cond], **extra)


def score(B, ag, world, test, key, eps_r=0.0, recs=None):
    r = rng(*key)
    row = W.edge_scores(B, ag, 0, test, r, noise=C.NOISE)
    row.update(W.boundary_shift(B, ag, 0, r, noise=C.NOISE))
    row["shift_unsigned"] = 0.5 * (abs(row["shift_wide"]) + abs(row["shift_narrow"]))
    row["home_logloss"] = W.log_loss(world, B, ag, 0, r, eps_r=eps_r)
    row["competence_home"] = W.competence(world, B, ag, 0, r, eps_r=eps_r)
    row["edge_acc"] = 0.5 * (row["wide_edge_acc"] + row["narrow_edge_acc"])
    row["conf_err"] = 0.5 * (row["wide_conf_err"] + row["narrow_conf_err"])
    row["edge_logloss"] = 0.5 * (row["wide_edge_logloss"] + row["narrow_edge_logloss"])
    if recs:
        rec = {k: np.concatenate([x[k] for x in recs]) for k in recs[0]}
        row.update(acts=int(rec["act"].sum()), updates=int(rec["upd"].sum()),
                   share_wide=W.label_share(rec, W.WIDE), share_narrow=W.label_share(rec, W.NARROW),
                   offered_edge_share=float(W.is_edge(rec["ang"]).mean()))
        for name, acts in (("wide", W.WIDE), ("narrow", W.NARROW)):
            row.update({f"{name}_{k}": v for k, v in W.refusal_sources(rec, acts).items()})
    return row


def stack_agents(*ags):
    H = max(a["W1"].shape[1] for a in ags)
    assert all(a["W1"].shape[1] == H for a in ags), "stacked learners need equal widths"
    return {k: np.concatenate([a[k] for a in ags]) for k in ("W1", "b1", "W2", "b2", "body")}


def take(ag, i):
    return {k: ag[k][i:i + 1].copy() for k in ("W1", "b1", "W2", "b2", "body")}


# ------------------------------------------------------------------ E1 and the wear sanity checks (symmetric)

def _e1_record(B, seed, tag):
    def rec(w, bl):
        d = dict(built_share_overlap=w.built_share_overlap(), built_count_overlap=w.built_count_overlap(),
                 u1_move=w.u_move(0), u2_move=w.u_move(1), edge_share=w.edge_share())
        d["app_grad"] = W.appearance_gradient(B, bl, w, rng(seed, "appgrad", tag, w.alpha.sum()), n=200)
        return d
    return rec


def e1(seed, M, rname, delta=None):
    nat, B, _ = setup(seed, "symmetric")
    tag = ("e1", M, rname, delta)
    _, _, traj = build(nat, B, C.RULES[rname], seed, tag, M=M, warm=C.E1_STEPS, every=C.E1_EVERY, delta=delta,
                       record=_e1_record(B, seed, tag))
    key = dict(seed=seed, M=M, rule=rname, delta=delta if delta is not None else C.DELTA)
    return {"traj": [dict(key, **t) for t in traj]}


# ------------------------------------------------------------------ E2 (symmetric, shaping by eye kappa = 1)

def e2(seed, M, s_u, delta, width, steps=C.E1_STEPS):
    nat, B, _ = setup(seed, "symmetric")
    rule = dict(rule=2, s_u=s_u, kappa=1.0)
    rec = lambda w, bl: dict(built_share_overlap=w.built_share_overlap(), built_count_overlap=w.built_count_overlap(),
                             n_overlap_home=int(w.overlap_home().sum()))
    _, _, traj = build(nat, B, rule, seed, ("e2", M, s_u, delta, width), M=M, warm=steps, every=C.E1_EVERY,
                       delta=delta, width=width, record=rec)
    key = dict(seed=seed, M=M, s_u=s_u, delta=delta, width=width)
    return {"traj": [dict(key, **t) for t in traj]}


# ------------------------------------------------------------------ E3 (unequal)

def _curve(world, B, ag, frozen, rule, seed_key, sd):
    """Competence every 250 steps to the cap; home log loss at 2,000 steps."""
    comps, ll2k, reached, acts, s = [], np.nan, np.nan, 0, 0
    while True:
        c = W.competence(world, B, ag, 0, rng(*seed_key, "comp", s))
        comps.append(c)
        if s == C.E3_LL_AT:
            ll2k = W.log_loss(world, B, ag, 0, rng(*seed_key, "ll"))
        if np.isnan(reached) and s > 0 and c >= C.COMP:
            reached = s
        if s >= C.E3_CAP or (not np.isnan(reached) and s >= max(C.E3_EARLY, C.E3_LL_AT)):
            break
        acts += int(live(world, B, ag, frozen, rule, C.E3_CHECK, sd())["act"].sum())
        s += C.E3_CHECK
    early = float(np.mean(comps[: C.E3_EARLY // C.E3_CHECK + 1]))
    return dict(area=early, steps_to_crit=reached, ll2k=ll2k, acts=acts)


def e3(seed, rname):
    nat, B, test = setup(seed, "unequal")
    bb = 0 if seed % 2 == 0 else 4
    rule = C.RULES[rname]
    w0, fr, _ = build(nat, B, rule, seed, ("e3", rname, bb), body=bb)
    out = {"worlds": [world_row(w0, nat, seed=seed, exp="e3", rule=rname, builder_body=bb)], "e3": []}
    same_type = [b for b in range(8) if (b < 4) == (bb < 4) and b != bb]
    outsiders = [b for b in range(8) if (b < 4) != (bb < 4)]
    arms = [("same_body", bb)] + [("same_type", b) for b in same_type] + [("outsider", b) for b in outsiders]
    ent = {b: W.voronoi_entropy(B, b, nat, w0) for b in range(8)}
    for arm, body in arms:
        for j in range(C.E3_NEWBORNS_PER_MAP):
            row = dict(seed=seed, rule=rname, builder_body=bb, arm=arm, body=body, j=j,
                       h_natural=ent[body][0], h_built=ent[body][1])
            for wk, world, frz in (("natural", nat, None), ("built", w0, fr)):
                ag = W.make_agents(rng(seed, "e3nb", body, j), 1, C.BASE_WIDTH, body)
                c = _curve(world.copy(), B, ag, frz, rule, (seed, "e3", wk, body, j), Seeder(seed, "e3", rname, wk, body, j))
                code = W.code_length_bits(world, B, W.make_agents(rng(seed, "e3nb", body, j), 1, C.BASE_WIDTH, body),
                                          C.CODE_PAIRS, seed=sseed(seed, "e3code", rname, wk, body, j), lr=C.LR)
                row.update({f"{k}_{wk}": v for k, v in c.items()}, **{f"code_{wk}": code})
            out["e3"].append(row)
    return out


# ------------------------------------------------------------------ E4-E6 (symmetric)

def pair_row(B, nat, test, a, wa, b, wb, **key):
    ov, un = nat.overlap_home(), nat.unambiguous_home()
    ca = W.chosen_actions(B, a, 0, wa.alpha, wa.u)
    cb = W.chosen_actions(B, b, 0, wb.alpha, wb.u)
    tov, tun = W.in_overlap(test["alpha"]), W.unambiguous(test["alpha"])
    ta = W.chosen_actions(B, a, 0, test["alpha"], test["u"])
    tb = W.chosen_actions(B, b, 0, test["alpha"], test["u"])
    ek, k = W.e_k(W.hidden_acts(B, a, 0, wa.alpha, wa.u), W.hidden_acts(B, b, 0, wb.alpha, wb.u))
    return dict(**key, kappa_overlap=W.cohen_kappa(ca[ov], cb[ov]), kappa_unamb=W.cohen_kappa(ca[un], cb[un]),
                agree_overlap=float((ca[ov] == cb[ov]).mean()), agree_unamb=float((ca[un] == cb[un]).mean()),
                kappa_overlap_test=W.cohen_kappa(ta[tov], tb[tov]), kappa_unamb_test=W.cohen_kappa(ta[tun], tb[tun]),
                e_k=ek, k_raw=k, n_overlap=int(ov.sum()), n_unamb=int(un.sum()),
                acc_a=float(W.accepted_matrix(wa.alpha)[np.arange(len(ca)), ca].mean()),
                acc_b=float(W.accepted_matrix(wb.alpha)[np.arange(len(cb)), cb].mean()))


def raise_joint(world, B, frozen, rule, bodies, key, steps=C.RAISE_STEPS):
    ag = stack_agents(*[W.make_agents(rng(*key, i), 1, C.BASE_WIDTH, b) for i, b in enumerate(bodies)])
    w = world.copy()
    live(w, B, ag, frozen, rule, steps, sseed(*key, "live"))
    return ag, w


def e4(seed, rname):
    """E4 and E5 for one shaping rule; natural pairs are shared (identical in both rule tasks)."""
    nat, B, test = setup(seed, "symmetric")
    rule = C.RULES[rname]
    wA, frA, _ = build(nat, B, rule, seed, ("e4A", rname))
    wB, frB, _ = build(nat, B, rule, seed, ("e4B", rname))
    out = {"worlds": [world_row(w, nat, seed=seed, exp="e4", rule=rname, world=t) for w, t in ((wA, "A"), (wB, "B"))],
           "pairs": [], "decay": []}
    P = lambda a, wa, b, wb, kind, p, phase: pair_row(B, nat, test, a, wa, b, wb, seed=seed, rule=rname, kind=kind,
                                                       pair=p, phase=phase)
    for p in range(C.E4_PAIRS):
        ag, w = raise_joint(wA, B, frA, rule, (0, 0), (seed, "e4same", rname, p))
        out["pairs"].append(P(take(ag, 0), w, take(ag, 1), w, "same", p, "kept"))
        a, wa = raise_joint(wA, B, frA, rule, (0,), (seed, "e4diffA", rname, p))
        b, wb = raise_joint(wB, B, frB, rule, (0,), (seed, "e4diffB", rname, p))
        out["pairs"].append(P(a, wa, b, wb, "different", p, "kept"))
        ag, w = raise_joint(nat, B, None, rule, (0, 0), (seed, "e4nat", p))
        out["pairs"].append(P(take(ag, 0), w, take(ag, 1), w, "natural", p, "kept"))
    # E5: abandonment, then the frozen-builders control
    ov = nat.overlap_home()
    ab = {}
    for wX, frX, t in ((wA, frA, "A"), (wB, frB, "B")):
        wa, wf = wX.copy(), wX.copy()
        sd = Seeder(seed, "e5", rname, t)
        disp0 = float(np.abs(W.wrapdiff(wa.alpha[ov], wa.home_a[ov])).mean())
        for s in range(0, C.ABANDON + 1, 100):
            if s > 0:
                W.step(wa, B, None, None, 100, seed=sd())
                live(wf, B, None, frX, rule, 100, sd())
            out["decay"].append(dict(seed=seed, rule=rname, world=t, step=s, disp0=disp0,
                                     disp=float(np.abs(W.wrapdiff(wa.alpha[ov], wa.home_a[ov])).mean()),
                                     built_share_overlap=wa.built_share_overlap(),
                                     built_share_overlap_frozen=wf.built_share_overlap()))
        ab[t] = wa
    for p in range(C.E4_PAIRS):
        ag, w = raise_joint(ab["A"], B, None, rule, (0, 0), (seed, "e5same", rname, p))
        out["pairs"].append(P(take(ag, 0), w, take(ag, 1), w, "same", p, "abandoned"))
        a, wa = raise_joint(ab["A"], B, None, rule, (0,), (seed, "e5diffA", rname, p))
        b, wb = raise_joint(ab["B"], B, None, rule, (0,), (seed, "e5diffB", rname, p))
        out["pairs"].append(P(a, wa, b, wb, "different", p, "abandoned"))
    return out


def e6(seed):
    nat, B, test = setup(seed, "symmetric")
    out = {"pairs": []}
    kinds = {"cross_type": [(i % 4, 4 + (i + i // 4) % 4) for i in range(5)],
             "same_map": [(i, i) for i in range(5)],
             "same_type": [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5)]}
    for kind, pairs in kinds.items():
        for p, (b1, b2) in enumerate(pairs):
            ag, w = raise_joint(nat, B, None, {}, (b1, b2), (seed, "e6", kind, p))
            out["pairs"].append(pair_row(B, nat, test, take(ag, 0), w, take(ag, 1), w, seed=seed, kind=kind, pair=p,
                                         body_a=b1, body_b=b2))
    return out


# ------------------------------------------------------------------ E8 (unequal)

def lifecycle_worlds(nat, B, seed, rules=("wear", "eye1")):
    return {r: build(nat, B, C.RULES[r], seed, ("lc", r))[:2] for r in rules}


def e8(seed, width):
    nat, B, test = setup(seed, "unequal")
    worlds = lifecycle_worlds(nat, B, seed)
    out = {"worlds": [world_row(w, nat, seed=seed, exp="lc", rule=r) for r, (w, _) in worlds.items()], "e8": [], "e8_traj": []}
    def ev(ag):
        sc = W.edge_scores(B, ag, 0, test, rng(seed, "e8eval"), noise=C.NOISE)
        return 0.5 * (sc["wide_edge_acc"] + sc["narrow_edge_acc"])
    for j in range(C.N_AG_GRID):
        sd = Seeder(seed, "e8", width, j)
        ag = W.make_agents(rng(seed, "e8", width, j), 1, width, 0)
        w = nat.copy(); steps, prev, plateau = 0, None, np.nan
        while steps < C.PRETRAIN_CAP:
            live(w, B, ag, None, {}, C.PRETRAIN_EVERY, sd()); steps += C.PRETRAIN_EVERY
            ea = ev(ag)
            if prev is not None and abs(ea - prev) < C.PLATEAU_TOL:
                plateau = steps
                break
            prev = ea
        before = score(B, ag, w, test, (seed, "e8score"))
        offered = {}
        for arm in ("treat_wear", "filtered_wear", "treat_eye1", "filtered_eye1", "natural"):
            a = W.copy_agents(ag); recs = []
            r = arm.split("_")[-1]
            if arm.startswith("treat"):
                wa = worlds[r][0].copy(); fr = worlds[r][1]
                for s in range(0, C.T2, 2000):
                    recs.append(live(wa, B, a, fr, C.RULES[r], 2000, sd()))
                    out["e8_traj"].append(dict(seed=seed, width=width, j=j, arm=arm, step=s + 2000,
                                               built_share=wa.built_share(), edge_share=wa.edge_share(), edge_acc=ev(a)))
                offered[r] = float(W.is_edge(np.concatenate([x["ang"] for x in recs])).mean())
            elif arm.startswith("filtered"):
                wa = nat.copy()
                recs.append(live(wa, B, a, None, {}, C.T2, sd(), edge_share=offered[r], chunk=50))
            else:
                wa = nat.copy()
                recs.append(live(wa, B, a, None, {}, C.T2, sd()))
            after = score(B, a, wa, test, (seed, "e8score"), recs=recs)
            out["e8"].append(dict(seed=seed, width=width, j=j, arm=arm, plateau_at=plateau,
                                  edge_acc_before=before["edge_acc"], edge_acc_after=after["edge_acc"],
                                  acts_T2=after["acts"], offered_edge_share=after["offered_edge_share"],
                                  target_edge_share=offered.get(r, np.nan) if arm.startswith("filtered") else np.nan,
                                  shift_unsigned_before=before["shift_unsigned"], shift_unsigned_after=after["shift_unsigned"]))
    return out


# ------------------------------------------------------------------ E9 and E10 (unequal)

def raise_scored(world, B, frozen, rule, test, ag, key, cond, eps_r=0.0, budgets=C.BUDGETS):
    """Raise one agent, scoring it at each budget. Returns (rows, agent, world)."""
    w = world.copy(); sd = Seeder(*key, "live"); rows, recs, done = [], [], 0
    for b in budgets:
        recs.append(live(w, B, ag, frozen, rule, b - done, sd(), cond=cond, eps_r=eps_r)); done = b
        rows.append(dict(budget=b, **score(B, ag, w, test, (*key, "score", b), eps_r=eps_r, recs=recs)))
    return rows, ag, w


def e9_snap(nat, B, test, ag, w_home, cond, key):
    home_ll = W.log_loss(w_home, B, ag, 0, rng(*key, "ll"))
    a = W.copy_agents(ag); ws = nat.copy(); ws.snap_home()
    snap_ll = W.log_loss(ws, B, a, 0, rng(*key, "ll"))
    sd = Seeder(*key, "e9"); sc = []
    for s in range(0, C.E9_STEPS + 1, C.E9_EVERY):
        if s > 0:
            live(ws, B, a, None, {}, C.E9_EVERY, sd(), cond=cond)
        sc.append(W.edge_scores(B, a, 0, test, rng(*key, "e9score"), noise=C.NOISE))
    row = dict(home_logloss=home_ll, snap_logloss=snap_ll, d_ll=snap_ll - home_ll,
               decisive_pay_mean=float(np.mean([x["decisive_pay"] for x in sc])))
    for t in ("wide", "narrow"):
        row[f"{t}_pay_mean"] = float(np.mean([x[f"{t}_decisive_pay"] for x in sc]))
        row[f"{t}_act_rate_mean"] = float(np.mean([x[f"{t}_act_rate"] for x in sc]))
        row[f"{t}_refusals_per_act_mean"] = float(np.nanmean([x[f"{t}_refusals_per_act"] for x in sc]))
    return row


def _kind_world(nat, worlds, kind):
    return (nat, None, {}) if kind == "wild" else (worlds[kind][0], worlds[kind][1], C.RULES[kind])


def e10_grid(seed, kind, cond):
    nat, B, test = setup(seed, "unequal")
    worlds = lifecycle_worlds(nat, B, seed, rules=(kind,)) if kind != "wild" else {}
    world, fr, rule = _kind_world(nat, worlds, kind)
    out = {"raised": [], "e9": [], "upd": []}
    prim_acts = []
    for width in C.WIDTH_LADDER:
        n = C.N_AG_PRIMARY if width == C.BASE_WIDTH else C.N_AG_GRID
        for j in range(n):
            key = (seed, "e10", kind, cond, width, j)
            rows, ag, w = raise_scored(world, B, fr, rule, test, W.make_agents(rng(*key), 1, width, 0), key, cond)
            for r in rows:
                out["raised"].append(dict(seed=seed, kind=kind, cond=cond, arm="grid", width=width, j=j, eps_r=0.0, **r))
            if width == C.BASE_WIDTH:
                prim_acts.append(rows[-1]["acts"])
                out["e9"].append(dict(seed=seed, kind=kind, cond=cond, j=j,
                                      **e9_snap(nat, B, test, ag, w, cond, key), **{k: rows[-1][k] for k in
                                      ("shift_wide", "shift_narrow", "shift_unsigned")}))
    if kind != "wild" and cond == "skip":   # update-count control (skip rule only; under passive steps = updates)
        target = float(np.mean(prim_acts))
        for j in range(C.N_AG_GRID):
            key = (seed, "upd", kind, j)
            ag = W.make_agents(rng(*key), 1, C.BASE_WIDTH, 0); w = nat.copy(); sd = Seeder(*key)
            acts = steps = 0; recs = []
            while acts < target and steps < C.UPD_CAP:
                recs.append(live(w, B, ag, None, {}, 2000, sd())); acts += int(recs[-1]["act"].sum()); steps += 2000
            out["upd"].append(dict(seed=seed, kind=kind, j=j, target_acts=target, wild_steps=steps,
                                   **score(B, ag, w, test, (*key, "score"), recs=recs)))
    return out


def e10_arm(seed, arm, kind, cond, n_ag, eps_r=0.0):
    """Mechanism (skip010/skip030), dose (eye1_su01/eye1_su003) and matched-surprise arms at the primary cell."""
    nat, B, test = setup(seed, "unequal")
    rules = (kind,) if kind != "wild" else ()
    worlds = {}
    for r in rules:
        worlds[r] = build(nat, B, C.RULES[r], seed, ("lc", r))[:2]
    world, fr, rule = _kind_world(nat, worlds, kind)
    out = {"raised": [], "worlds": [world_row(w, nat, seed=seed, exp="lc", rule=r) for r, (w, _) in worlds.items()]}
    for j in range(n_ag):
        key = (seed, "e10" + arm, kind, cond, j)
        rows, _, _ = raise_scored(world, B, fr, rule, test, W.make_agents(rng(*key), 1, C.BASE_WIDTH, 0), key, cond,
                                  eps_r=eps_r, budgets=(C.BUDGETS[-1],))
        out["raised"].append(dict(seed=seed, kind=kind, cond=cond, arm=arm, width=C.BASE_WIDTH, j=j, eps_r=eps_r, **rows[0]))
    return out


def e10_ens(seed, kind, cond):
    nat, B, test = setup(seed, "unequal")
    worlds = lifecycle_worlds(nat, B, seed, rules=(kind,)) if kind != "wild" else {}
    world, fr, rule = _kind_world(nat, worlds, kind)
    out = {"ens": []}
    for vname, (widths, beta) in C.ENS_VARIANTS.items():
        for j in range(C.N_ENS):
            key = (seed, "ens", kind, cond, vname, j)
            ens = W.make_ensemble(rng(*key), C.ENS_K, widths, body=0, beta=beta)
            w = world.copy()
            acts = W.step_ens(w, B, ens, fr, C.BUDGETS[-1], seed=sseed(*key, "live"), lr=C.LR, passive=(cond == "passive"),
                              eps_min=C.EXPLORE["eps_min"], **(rule if fr is not None else {}))
            r = rng(*key, "score")
            es = W.edge_scores_ens(B, ens, test, r, noise=C.NOISE)
            es.update(W.boundary_shift_ens(B, ens, r, noise=C.NOISE))
            es["shift_unsigned"] = 0.5 * (abs(es["shift_wide"]) + abs(es["shift_narrow"]))
            es["conf_err"] = 0.5 * (es["wide_conf_err"] + es["narrow_conf_err"])
            es["edge_logloss"] = 0.5 * (es["wide_edge_logloss"] + es["narrow_edge_logloss"])
            out["ens"].append(dict(seed=seed, kind=kind, cond=cond, variant=vname, j=j, acts=int(acts.sum()),
                                   home_logloss=W.ens_home_logloss(w, B, ens, r), **es))
    return out


def e10_sym(seed):
    nat, B, test = setup(seed, "symmetric")
    worlds = lifecycle_worlds(nat, B, seed, rules=("wear",))
    hist = W.angle_histogram(worlds["wear"][0])
    out = {"sym": [], "sym_hist": [dict(seed=seed, bin_lo=5 * i, share=float(h)) for i, h in enumerate(hist)],
           "worlds": [world_row(worlds["wear"][0], nat, seed=seed, exp="sym", rule="wear")]}
    for kind in ("wild", "wear"):
        world, fr, rule = _kind_world(nat, worlds, kind)
        for cond in ("skip", "passive"):
            for j in range(C.N_AG_GRID):
                key = (seed, "sym", kind, cond, j)
                rows, _, _ = raise_scored(world, B, fr, rule, test, W.make_agents(rng(*key), 1, C.BASE_WIDTH, 0), key,
                                          cond, budgets=(C.BUDGETS[-1],))
                out["sym"].append(dict(seed=seed, kind=kind, cond=cond, j=j, **rows[0]))
    W.set_geometry(W.UNEQUAL)
    return out


# ------------------------------------------------------------------ task lists

def tasks(exp, seeds):
    T = []
    for s in seeds:
        if exp == "e1":
            T += [(s, M, r) for M in C.E1_M for r in C.E1_RULES]
        elif exp == "e2":
            rg = C.E2_RANGE
            T += [(s, M, su, d, 32) for M in rg["M"] for su in rg["s_u"] for d in rg["delta"]]
            T += [(s, rg["M_mid"], rg["s_u_mid"], rg["delta_mid"], wd) for wd in C.E2_WIDTHS if wd != 32]
        elif exp == "e3":
            T += [(s, r) for r in ("eye025", "wear")]
        elif exp == "e4":
            T += [(s, r) for r in ("wear", "eye1")]
        elif exp == "e6":
            T += [(s,)]
        elif exp == "e8":
            T += [(s, wd) for wd in C.WIDTH_LADDER]
        elif exp == "e10":
            T += [("grid", s, k, c) for k in ("wild", "wear", "eye1") for c in ("skip", "passive")]
            T += [("arm", s, c, k, c, C.N_AG_GRID) for c in ("skip010", "skip030") for k in ("wild", "wear", "eye1")]
            T += [("arm", s, "dose", r, "skip", C.N_AG_GRID) for r in ("eye1_su01", "eye1_su003")]
            T += [("arm", s, "matched", r, c, C.N_AG_PRIMARY, C.EPS_R[(r, c)]) for r in ("wear", "eye1")
                  for c in ("skip", "passive")]
            T += [("ens", s, k, c) for k in ("wild", "wear", "eye1") for c in ("skip", "passive")]
            T += [("sym", s)]
    return T


def run_task(exp, t):
    if exp == "e1":
        return e1(*t)
    if exp == "e2":
        return e2(*t)
    if exp == "e3":
        return e3(*t)
    if exp == "e4":
        return e4(*t)
    if exp == "e6":
        return e6(*t)
    if exp == "e8":
        return e8(*t)
    if exp == "e10":
        kind, rest = t[0], t[1:]
        return {"grid": e10_grid, "arm": e10_arm, "ens": e10_ens, "sym": e10_sym}[kind](*rest)
    raise ValueError(exp)


# ------------------------------------------------------------------ stage-2 tuning and sanity tasks (seeds 7000-7099)

def tune_leg(seed, geom, width, lam=None):
    """Legibility and exploration check: a lone wild agent, skip rule, 60k steps (wild agents only)."""
    nat, B, test = setup(seed, geom, lam)
    lam = C.LAM[geom] if lam is None else lam
    key = (seed, "tune_leg", geom, width) if lam == 0.4 else (seed, "tune_leg", geom, width, lam)
    ag = W.make_agents(rng(*key), 1, width, 0); w = nat.copy(); sd = Seeder(*key)
    comp, reached = [], np.nan
    for s in range(5000, 60001, 5000):
        live(w, B, ag, None, {}, 5000, sd())
        c = W.competence(w, B, ag, 0, rng(*key, "comp", s), n=2000)
        comp.append(c)
        if np.isnan(reached) and c >= C.COMP:
            reached = s
    row = score(B, ag, w, test, (*key, "score"))
    W.set_geometry(W.UNEQUAL)
    return {"leg": [dict(seed=seed, geom=geom, width=width, lam=lam, steps_to_comp=reached,
                         comp_60k=comp[-1], **row)]}


def tune_epsr(seed, kind, cond, grid=tuple(round(0.01 * i, 2) for i in range(21)), n_ag=2):
    """Home log loss only (L10a-matched's eps_r); kind 'wild' gives the eps_r = 0 comparator."""
    nat, B, test = setup(seed, "unequal")
    worlds = lifecycle_worlds(nat, B, seed, rules=(kind,)) if kind != "wild" else {}
    world, fr, rule = _kind_world(nat, worlds, kind)
    rows = []
    for er in (grid if kind != "wild" else (0.0,)):
        for j in range(n_ag):
            key = (seed, "tune_epsr", kind, cond, er, j)
            ag = W.make_agents(rng(*key), 1, C.BASE_WIDTH, 0); w = world.copy()
            live(w, B, ag, fr, rule, C.BUDGETS[-1], sseed(*key), cond=cond, eps_r=er)
            rows.append(dict(seed=seed, kind=kind, cond=cond, eps_r=er, j=j,
                             home_logloss=W.log_loss(w, B, ag, 0, rng(*key, "ll"), eps_r=er)))
    return {"epsr": rows}


def tune_leg_diag(seed, geom, lam, j):
    """Diagnostic for the legibility check: agent j on map 0, scored at 60k and 120k steps (wild agents only)."""
    nat, B, test = setup(seed, geom, lam)
    key = (seed, "tune_leg_diag", geom, lam, j)
    ag = W.make_agents(rng(*key), 1, C.BASE_WIDTH, 0); w = nat.copy(); sd = Seeder(*key)
    rows = []
    for s in (60000, 120000):
        live(w, B, ag, None, {}, 60000, sd())
        rows.append(dict(seed=seed, geom=geom, lam=lam, j=j, steps=s, **score(B, ag, w, test, (*key, "score", s))))
    W.set_geometry(W.UNEQUAL)
    return {"diag": rows}


def tasks_tuning(exp, seeds):
    T = []
    for s in seeds:
        if exp == "tune_leg":
            T += [(s, "symmetric", 32)] + [(s, "unequal", wd) for wd in (4, 8, 16, 32)]
        elif exp == "tune_leg_lam":
            T += [(s, g, wd, lam) for lam in (0.35, 0.3, 0.25, 0.2, 0.15, 0.1)
                  for g, wd in [("symmetric", 32)] + [("unequal", w_) for w_ in (4, 8, 16, 32)]]
        elif exp == "tune_leg_diag":
            T += [(s, g, lam, j) for g in ("symmetric", "unequal") for lam in (0.4, 0.3, 0.2) for j in range(3)]
        elif exp == "tune_epsr":
            T += [(s, k, c) for k in ("wild", "wear", "eye1") for c in ("skip", "passive")]
        elif exp == "tune_e2":
            sw = C.E2_SWEEP
            T += [(s, M, su, d, 32) for M in sw["M"] for su in sw["s_u"] for d in sw["delta"]]
        elif exp == "sanity_wear":
            T += [(s, 6, "none", None)] + [(s, 6, "wear", d) for d in (0.00025, 0.0005, 0.001, 0.002, 0.004)]
    return T


_tasks_main, _run_main = tasks, run_task


def tasks(exp, seeds):  # noqa: F811
    return tasks_tuning(exp, seeds) if exp.startswith(("tune_", "sanity")) else _tasks_main(exp, seeds)


def run_task(exp, t):  # noqa: F811
    if exp in ("tune_leg", "tune_leg_lam"):
        return tune_leg(*t)
    if exp == "tune_epsr":
        return tune_epsr(*t)
    if exp == "tune_leg_diag":
        return tune_leg_diag(*t)
    if exp == "tune_e2":
        return e2(*t)
    if exp == "sanity_wear":
        return e1(*t)
    return _run_main(exp, t)
