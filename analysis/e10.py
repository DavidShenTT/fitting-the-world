"""E10 and E9 against PREREGISTRATION.md section 8 (with DEVIATIONS.md). Reads results/e10/*.csv.

    python analysis/e10.py [results subdir, default e10]
Writes results/<dir>/summary_*.csv and prints every table.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONF_ITEMS, ROOT, boot_ci, diff_means, paired_ratio, save, show, world_means  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "e10"
P = os.path.join(ROOT, "results", D)
R = pd.read_csv(os.path.join(P, "raised.csv"))
R["stretch"] = 0.5 * (R.shift_narrow - R.shift_wide)
for t in ("wide", "narrow"):
    lab = R[f"{t}_acc"] + R[f"{t}_inside_ref"] + R[f"{t}_near_ref"] + R[f"{t}_far_ref"]
    ref = R[f"{t}_inside_ref"] + R[f"{t}_near_ref"] + R[f"{t}_far_ref"]
    R[f"{t}_near_share"] = R[f"{t}_near_ref"] / ref.replace(0, np.nan)
    R[f"{t}_far_share"] = R[f"{t}_far_ref"] / ref.replace(0, np.nan)
    R[f"{t}_acc_per_far"] = R[f"{t}_acc"] / R[f"{t}_far_ref"].replace(0, np.nan)
COLS = ["conf_err", "edge_logloss", "home_logloss", "shift_unsigned", "shift_wide", "shift_narrow", "stretch", "edge_acc",
        "decisive_pay", "acts", "updates", "share_wide", "share_narrow", "wide_near_share", "wide_far_share",
        "wide_acc_per_far", "narrow_near_share", "narrow_far_share", "narrow_acc_per_far", "competence_home"]
KEYS = ["seed", "kind", "cond", "arm", "width", "budget"]
WM = world_means(R, KEYS, COLS)
PRIM = dict(width=32, budget=60000)


def cell(kind, cond, arm="grid", width=32, budget=60000):
    return WM[(WM.kind == kind) & (WM.cond == cond) & (WM.arm == arm) & (WM.width == width) & (WM.budget == budget)]


def l10a_row(b, w, **key):
    cr = paired_ratio(b, w, "conf_err", CONF_ITEMS)
    lr = paired_ratio(b, w, "edge_logloss")
    hb, hw = b.home_logloss.mean(), w.home_logloss.mean()
    ok = cr[0] >= 1.5 and lr[0] >= 1.5 and hb <= hw
    against = cr[0] <= 1 or lr[0] <= 1 or hb > hw
    return dict(**key, conf_ratio=cr[0], conf_lo=cr[1], conf_hi=cr[2], edge_ll_ratio=lr[0], ll_lo=lr[1], ll_hi=lr[2],
                home_ll_built=hb, home_ll_wild=hw, conf_built=b.conf_err.mean(), conf_wild=w.conf_err.mean(),
                n_worlds=cr[3], zero_wild_worlds=cr[4], verdict="pass" if ok else ("against" if against else "neither"))


# ---------------- L10a (primary cell) and every secondary width x budget cell
rows, sec = [], []
for cond in ("skip", "passive"):
    for rule in ("wear", "eye1"):
        rows.append(l10a_row(cell(rule, cond), cell("wild", cond), cond=cond, rule=rule))
        for (width, budget), _ in WM[WM.arm == "grid"].groupby(["width", "budget"]):
            r = l10a_row(cell(rule, cond, width=width, budget=budget), cell("wild", cond, width=width, budget=budget),
                         cond=cond, rule=rule, width=width, budget=budget)
            r["shift_diff"] = diff_means(cell(rule, cond, width=width, budget=budget),
                                         cell("wild", cond, width=width, budget=budget), "shift_unsigned")[0]
            sec.append(r)
L10A = pd.DataFrame(rows); save(L10A, P, "summary_L10a.csv"); show(L10A, "L10a, primary cell (width 32, 60k)")
SEC = pd.DataFrame(sec); save(SEC, P, "summary_L10a_cells.csv")
show(SEC.groupby(["cond", "rule"]).apply(lambda g: pd.Series(dict(cells=len(g), L10a_pass=(g.verdict == "pass").sum(),
     shift_ge_2_5=(g.shift_diff >= 2.5).sum())), include_groups=False).reset_index(), "L10a and shift across all width x budget cells")
show(SEC[["cond", "rule", "width", "budget", "conf_ratio", "edge_ll_ratio", "home_ll_built", "home_ll_wild", "shift_diff", "verdict"]],
     "secondary cells")

# ---------------- update-count control (skip rule): wild agents with as many acted turns as built-raised ones
U = pd.read_csv(os.path.join(P, "upd.csv"))
urows = []
for rule in ("wear", "eye1"):
    uw = world_means(U[U.kind == rule].assign(kind="wild_upd"), ["seed"], ["conf_err", "edge_logloss", "home_logloss",
                                                                          "shift_unsigned", "acts", "wild_steps"])
    r = l10a_row(cell(rule, "skip"), uw, cond="skip", rule=rule, comparator="wild, equal acted turns")
    r["shift_diff"], r["shift_lo"], r["shift_hi"], _ = diff_means(cell(rule, "skip"), uw, "shift_unsigned")
    r["wild_steps_mean"] = uw.wild_steps.mean(); r["acts_built"] = cell(rule, "skip").acts.mean(); r["acts_wild_upd"] = uw.acts.mean()
    urows.append(r)
UPD = pd.DataFrame(urows); save(UPD, P, "summary_update_control.csv"); show(UPD, "update-count control (skip rule)")

# ---------------- L10-shift (skip rule)
srows = []
for rule in ("wear", "eye1"):
    m, lo, hi, n = diff_means(cell(rule, "skip"), cell("wild", "skip"), "shift_unsigned")
    srows.append(dict(rule=rule, shift_diff=m, lo=lo, hi=hi, n_worlds=n, shift_built=cell(rule, "skip").shift_unsigned.mean(),
                      shift_wild=cell("wild", "skip").shift_unsigned.mean(),
                      verdict="pass" if m >= 2.5 else ("against" if m < 1 else "neither")))
for rule in ("wear", "eye1"):   # passive, reported only
    m, lo, hi, n = diff_means(cell(rule, "passive"), cell("wild", "passive"), "shift_unsigned")
    srows.append(dict(rule=rule + " (passive, no criterion)", shift_diff=m, lo=lo, hi=hi, n_worlds=n,
                      shift_built=cell(rule, "passive").shift_unsigned.mean(), shift_wild=cell("wild", "passive").shift_unsigned.mean(),
                      verdict=""))
SH = pd.DataFrame(srows); save(SH, P, "summary_L10shift.csv"); show(SH, "L10-shift (pooled unsigned shift, built - wild)")

# ---------------- L10-dose (skip rule)
Wd = pd.read_csv(os.path.join(P, "worlds.csv")).drop_duplicates(["seed", "exp", "rule"])
Wd["removed"] = Wd.edge_share_natural - Wd.edge_share
drows = []
for lvl, (kind, arm) in {"eye1_su003": ("eye1_su003", "dose"), "eye1_su01": ("eye1_su01", "dose"), "eye1": ("eye1", "grid")}.items():
    b = cell(kind, "skip", arm=arm)
    drows.append(dict(level=lvl, removed=Wd[(Wd.exp == "lc") & (Wd.rule == lvl)].removed.mean(),
                      edge_share_left=Wd[(Wd.exp == "lc") & (Wd.rule == lvl)].edge_share.mean(),
                      shift_diff=diff_means(b, cell("wild", "skip"), "shift_unsigned")[0],
                      conf_ratio=paired_ratio(b, cell("wild", "skip"), "conf_err", CONF_ITEMS)[0]))
DO = pd.DataFrame(drows).sort_values("removed")
DO["verdict"] = "pass" if (np.all(np.diff(DO.shift_diff) > 0) and np.all(np.diff(DO.conf_ratio) > 0)) else "fail"
save(DO, P, "summary_L10dose.csv"); show(DO, "L10-dose (ordered by edge items removed)")

# ---------------- L10m (skip rule, eps_min 0.02 / 0.1 / 0.3)
mrows = []
for rule in ("wear", "eye1"):
    vals = []
    for eps, arm, cond in ((0.02, "grid", "skip"), (0.1, "skip010", "skip010"), (0.3, "skip030", "skip030")):
        b, w = cell(rule, cond, arm=arm), cell("wild", cond, arm=arm)
        st = diff_means(b, w, "stretch")
        vals.append(st[0])
        mrows.append(dict(rule=rule, eps_min=eps, stretch_diff=st[0], lo=st[1], hi=st[2], raw_stretch_built=b.stretch.mean(),
                          raw_stretch_wild=w.stretch.mean(), wide_near_share_built=b.wide_near_share.mean(),
                          wide_far_share_built=b.wide_far_share.mean(), wide_acc_per_far_built=b.wide_acc_per_far.mean(),
                          wide_near_share_wild=w.wide_near_share.mean(), wide_acc_per_far_wild=w.wide_acc_per_far.mean()))
    for r in mrows[-3:]:
        r["verdict"] = "pass" if vals[0] > vals[1] > vals[2] else "against"
LM = pd.DataFrame(mrows); save(LM, P, "summary_L10m.csv"); show(LM, "L10m (mean stretch, built - wild)")
sp = []
for rule in ("wear", "eye1"):
    pts = [(cell(rule, c, arm=a).stretch.mean(), cell(rule, c, arm=a).wide_acc_per_far.mean())
           for c, a in (("skip", "grid"), ("skip010", "skip010"), ("skip030", "skip030"), ("passive", "grid"))]
    rho = spearmanr([p[0] for p in pts], [p[1] for p in pts]).statistic
    sp.append(dict(rule=rule, spearman_raw_stretch_vs_acc_per_far=rho, n_conditions=len(pts)))
show(pd.DataFrame(sp), "rank correlation across conditions (exploratory)")

# ---------------- L10a-matched
mt = []
for rule in ("wear", "eye1"):
    for cond in ("skip", "passive"):
        b = R[(R.kind == rule) & (R.cond == cond) & (R.arm == "matched")].assign(k="built")
        w = R[(R.kind == "wild") & (R.cond == cond) & (R.arm == "grid") & (R.width == 32) & (R.budget == 60000)].assign(k="wild")
        a = pd.concat([b, w])
        for nb in (3, 2, 1):
            a["bin"] = pd.qcut(a.home_logloss, nb, labels=False, duplicates="drop") if nb > 1 else 0
            cnt = a.groupby(["bin", "k"]).size().unstack(fill_value=0)
            if len(cnt) == nb and (cnt.min(axis=1) >= 10).all() and cnt.shape[1] == 2:
                break
        runnable = nb >= 3 or (len(cnt) == nb and nb >= 3)
        for bn, g in a.groupby("bin"):
            gb, gw = g[g.k == "built"], g[g.k == "wild"]
            cr = gb.conf_err.mean() / gw.conf_err.mean() if gw.conf_err.mean() > 0 else np.inf
            lr = gb.edge_logloss.mean() / gw.edge_logloss.mean()
            mt.append(dict(rule=rule, cond=cond, eps_r=b.eps_r.iloc[0], n_bins=nb, runnable=nb >= 3, bin=bn,
                           n_built=len(gb), n_wild=len(gw), home_ll_built=gb.home_logloss.mean(), home_ll_wild=gw.home_logloss.mean(),
                           conf_ratio=cr, edge_ll_ratio=lr,
                           bin_ok=bool(cr >= 1.5 and lr >= 1.5 and gb.home_logloss.mean() <= gw.home_logloss.mean() + 0.005)))
MT = pd.DataFrame(mt)
MT["verdict"] = MT.groupby(["rule", "cond"]).bin_ok.transform(lambda s: "pass" if s.all() else "fail")
MT.loc[~MT.runnable, "verdict"] = "not runnable (< 3 bins with >= 10 of each kind)"
save(MT, P, "summary_L10a_matched.csv"); show(MT, "L10a-matched (home log-loss bins)")

# ---------------- ensembles
E = pd.read_csv(os.path.join(P, "ens.csv"))
E["spread_ratio"] = E.spread_edges / E.spread_centres
EW = world_means(E, ["seed", "kind", "cond", "variant"], ["conf_err", "edge_logloss", "home_logloss", "shift_unsigned", "spread_ratio"])
man = E[E.variant == "prior"].groupby("kind").spread_ratio.mean()
show(man.reset_index(), "randomized-prior manipulation check: member spread, edges / centres (need >= 2)")
erows = []
for v in ("standard", "diverse", "prior"):
    for cond in ("skip", "passive"):
        for rule in ("wear", "eye1"):
            b = EW[(EW.kind == rule) & (EW.cond == cond) & (EW.variant == v)]
            w = EW[(EW.kind == "wild") & (EW.cond == cond) & (EW.variant == v)]
            r = l10a_row(b.assign(n_agents=3), w.assign(n_agents=3), variant=v, cond=cond, rule=rule)
            r["shift_diff"] = diff_means(b, w, "shift_unsigned")[0]
            erows.append(r)
EN = pd.DataFrame(erows); EN["prior_arm_kept"] = bool((E[E.variant == "prior"].spread_ratio.mean()) >= 2)
save(EN, P, "summary_L10a_ens.csv"); show(EN, "L10a-ens")

# ---------------- E9
E9 = pd.read_csv(os.path.join(P, "e9.csv"))
E9W = world_means(E9, ["seed", "kind", "cond"], ["decisive_pay_mean", "d_ll", "wide_pay_mean", "narrow_pay_mean",
                                                 "wide_act_rate_mean", "narrow_act_rate_mean", "wide_refusals_per_act_mean",
                                                 "narrow_refusals_per_act_mean", "shift_wide", "shift_narrow"])
e9 = []
for cond in ("skip", "passive"):
    w = E9W[(E9W.kind == "wild") & (E9W.cond == cond)]
    for rule in ("wear", "eye1"):
        b = E9W[(E9W.kind == rule) & (E9W.cond == cond)]
        gap = (w.decisive_pay_mean.mean() - b.decisive_pay_mean.mean()) / w.decisive_pay_mean.mean()
        dll = diff_means(b, w, "d_ll")
        row = dict(cond=cond, rule=rule, pay_wild=w.decisive_pay_mean.mean(), pay_built=b.decisive_pay_mean.mean(), rel_gap=gap,
                   d_ll_diff=dll[0], d_ll_lo=dll[1], d_ll_hi=dll[2],
                   L9=("pass" if gap >= 0.30 else ("against" if gap < 0.10 else "neither")) if cond == "skip" else "(reported)",
                   L9b=("pass" if dll[0] >= 0.1 else "fail") if cond == "skip" else "(reported)")
        for t in ("wide", "narrow"):
            for k in ("act_rate", "refusals_per_act", "pay"):
                row[f"{t}_{k}_built"] = b[f"{t}_{k}_mean"].mean(); row[f"{t}_{k}_wild"] = w[f"{t}_{k}_mean"].mean()
            row[f"{t}_shift_built"] = b[f"shift_{t}"].mean()
        e9.append(row)
E9S = pd.DataFrame(e9); save(E9S, P, "summary_E9.csv"); show(E9S, "E9 (snap-back, first 500 steps)")

# ---------------- exploratory: symmetric check; home log loss vs confident error within built-raised agents
S = pd.read_csv(os.path.join(P, "sym.csv"))
show(S.groupby(["kind", "cond"])[["shift_wide", "shift_narrow", "shift_unsigned", "share_wide", "share_narrow", "conf_err",
                                  "home_logloss"]].mean().reset_index(), "symmetric-arc check (exploratory; partial gap)")
ex = []
for cond in ("skip", "passive"):
    for rule in ("wear", "eye1"):
        a = R[(R.kind == rule) & (R.cond == cond) & (R.arm == "grid") & (R.width == 32) & (R.budget == 60000)]
        ex.append(dict(cond=cond, rule=rule, spearman_home_ll_vs_conf_err=spearmanr(a.home_logloss, a.conf_err).statistic, n=len(a)))
show(pd.DataFrame(ex), "within built-raised agents (exploratory)")
show(pd.concat([cell(k, c).assign(k=k, c=c) for k in ("wild", "wear", "eye1") for c in ("skip", "passive")])
     .groupby(["k", "c"])[["edge_acc", "share_wide", "share_narrow", "acts"]].mean().reset_index(), "context: primary-cell edge accuracy, label shares")
