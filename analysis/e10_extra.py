"""Two post hoc analyses requested in review (24 Nov), from saved data. Neither changes a preregistered result.

1. L10m extras: confident-error ratio (built / wild) and payoff per decisive edge turn across the learner's eps_min,
   from results/e10/raised.csv (the same agents as L10m). The payoff is scored on the natural test set at the end of
   raising, i.e. the zero-step point of the E9 snap-back measure; the 500-step E9 run exists only at eps_min 0.02.
2. DEVIATIONS D5 (exploratory): outsider-body newcomers (body 4) against same-body newcomers (body 0) at the E10
   primary cell, from results/e10_outsider/raised.csv and results/e10/raised.csv.

    python analysis/e10_extra.py
Writes results/e10/summary_L10m_extra.csv and results/e10_outsider/summary_D5.csv.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CONF_ITEMS, ROOT, boot_ci, diff_means, paired_ratio, save, show, world_means  # noqa: E402

COLS = ["conf_err", "edge_logloss", "home_logloss", "shift_unsigned", "decisive_pay"]
KEYS = ["seed", "kind", "cond", "arm", "width", "budget"]
P10 = os.path.join(ROOT, "results", "e10")
WM = world_means(pd.read_csv(os.path.join(P10, "raised.csv")), KEYS, COLS)


def cell(wm, kind, cond, arm="grid", width=32, budget=60000):
    return wm[(wm.kind == kind) & (wm.cond == cond) & (wm.arm == arm) & (wm.width == width) & (wm.budget == budget)]


# ---------------- 1. L10m extras
rows = []
e9 = pd.read_csv(os.path.join(P10, "summary_E9.csv"))
for rule in ("wear", "eye1"):
    for eps, arm, cond in ((0.02, "grid", "skip"), (0.1, "skip010", "skip010"), (0.3, "skip030", "skip030")):
        b, w = cell(WM, rule, cond, arm), cell(WM, "wild", cond, arm)
        cr = paired_ratio(b, w, "conf_err", CONF_ITEMS)
        pb, pw = b.decisive_pay.mean(), w.decisive_pay.mean()
        rows.append(dict(rule=rule, eps_min=eps, conf_ratio=cr[0], conf_lo=cr[1], conf_hi=cr[2],
                         conf_built=b.conf_err.mean(), conf_wild=w.conf_err.mean(),
                         pay_at_snap_built=pb, pay_at_snap_wild=pw, pay_at_snap_gap=1 - pb / pw, n_worlds=cr[3]))
LX = pd.DataFrame(rows); save(LX, P10, "summary_L10m_extra.csv"); show(LX, "L10m extras (skip rule, primary cell)")


# ---------------- 2. D5: outsider-body newcomers
def log_ratios(built, wild, col, count_items=None):
    """Per-world log(built / wild), with paired_ratio's zero-wild adjustment. Returns a Series indexed by seed."""
    m = built[["seed", col, "n_agents"]].merge(wild[["seed", col, "n_agents"]], on="seed", suffixes=("_b", "_w"))
    b, w = m[f"{col}_b"].to_numpy(float), m[f"{col}_w"].to_numpy(float)
    zero = w <= 0
    if count_items is not None and zero.any():
        nb, nw = m.n_agents_b.to_numpy() * count_items, m.n_agents_w.to_numpy() * count_items
        b = np.where(zero, (b * nb + 0.5) / nb, b)
        w = np.where(zero, 0.5 / nw, w)
    return pd.Series(np.log(b / w), index=m.seed.to_numpy())


PO = os.path.join(ROOT, "results", "e10_outsider")
if os.path.exists(os.path.join(PO, "raised.csv")):
    O = world_means(pd.read_csv(os.path.join(PO, "raised.csv")), KEYS, COLS)
    rows = []
    for cond in ("skip", "passive"):
        for rule in ("wear", "eye1"):
            sb, sw = cell(WM, rule, cond), cell(WM, "wild", cond)
            ob, ow = cell(O, rule, cond, arm="body"), cell(O, "wild", cond, arm="body")
            rs, ro = paired_ratio(sb, sw, "conf_err", CONF_ITEMS), paired_ratio(ob, ow, "conf_err", CONF_ITEMS)
            d = (log_ratios(sb, sw, "conf_err", CONF_ITEMS) - log_ratios(ob, ow, "conf_err", CONF_ITEMS)).dropna()
            lo, hi = boot_ci(d.to_numpy())
            ll = paired_ratio(ob, ow, "edge_logloss")
            sh = diff_means(ob, ow, "shift_unsigned")
            rows.append(dict(cond=cond, rule=rule,
                             same_conf_ratio=rs[0], same_lo=rs[1], same_hi=rs[2],
                             out_conf_ratio=ro[0], out_lo=ro[1], out_hi=ro[2],
                             ratio_of_ratios=float(np.exp(d.mean())), ror_lo=float(np.exp(lo)), ror_hi=float(np.exp(hi)),
                             out_conf_built=ob.conf_err.mean(), out_conf_wild=ow.conf_err.mean(),
                             out_edge_ll_ratio=ll[0], out_home_ll_built=ob.home_logloss.mean(),
                             out_home_ll_wild=ow.home_logloss.mean(), out_shift_diff=sh[0], out_shift_lo=sh[1],
                             out_shift_hi=sh[2], n_worlds=len(d)))
    D5 = pd.DataFrame(rows); save(D5, PO, "summary_D5.csv"); show(D5, "D5: same-body vs outsider-body newcomers")
else:
    print("results/e10_outsider/raised.csv not found: run python runs/run.py e10_outsider 0-9")
