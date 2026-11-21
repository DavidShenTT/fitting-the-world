"""L10a-matched rerun under passive feedback (DEVIATIONS.md D4): the binning rule of PREREGISTRATION.md section 8,
unchanged, on the rerun built agents (results/e10_matched_v2) against the primary-cell wild passive agents (results/e10).
    python analysis/e10_matched_v2.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, save, show  # noqa: E402

R0 = pd.read_csv(os.path.join(ROOT, "results", "e10", "raised.csv"))
R1 = pd.read_csv(os.path.join(ROOT, "results", "e10_matched_v2", "raised.csv"))
UNMATCHED = {"eye1": "unmatched by rule (tuning ratio 0.79)", "wear": "matched (tuning ratio 0.835)"}
rows = []
for rule in ("wear", "eye1"):
    b = R1[(R1.kind == rule) & (R1.cond == "passive")].assign(k="built")
    w = R0[(R0.kind == "wild") & (R0.cond == "passive") & (R0.arm == "grid") & (R0.width == 32) & (R0.budget == 60000)].assign(k="wild")
    a = pd.concat([b, w])
    for nb in (3, 2, 1):
        a["bin"] = pd.qcut(a.home_logloss, nb, labels=False, duplicates="drop") if nb > 1 else 0
        cnt = a.groupby(["bin", "k"]).size().unstack(fill_value=0)
        if len(cnt) == nb and cnt.shape[1] == 2 and (cnt.min(axis=1) >= 10).all():
            break
    for bn, g in a.groupby("bin"):
        gb, gw = g[g.k == "built"], g[g.k == "wild"]
        cr = gb.conf_err.mean() / gw.conf_err.mean() if len(gw) and gw.conf_err.mean() > 0 else np.nan
        lr = gb.edge_logloss.mean() / gw.edge_logloss.mean() if len(gw) else np.nan
        rows.append(dict(rule=rule, status=UNMATCHED[rule], eps_r=b.eps_r.iloc[0], n_bins=nb, runnable=nb >= 3, bin=bn,
                         n_built=len(gb), n_wild=len(gw), home_ll_built=gb.home_logloss.mean(), home_ll_wild=gw.home_logloss.mean(),
                         conf_ratio=cr, edge_ll_ratio=lr,
                         bin_ok=bool(cr >= 1.5 and lr >= 1.5 and gb.home_logloss.mean() <= gw.home_logloss.mean() + 0.005)))
    rows.append(dict(rule=rule, status="all agents", bin="all", n_built=len(b), n_wild=len(w),
                     home_ll_built=b.home_logloss.mean(), home_ll_wild=w.home_logloss.mean(),
                     conf_ratio=b.conf_err.mean() / w.conf_err.mean(), edge_ll_ratio=b.edge_logloss.mean() / w.edge_logloss.mean()))
M = pd.DataFrame(rows)
for rule in ("wear", "eye1"):
    g = M[(M.rule == rule) & (M.bin != "all")]
    v = ("pass" if g.bin_ok.all() else "fail") if g.runnable.all() else "not runnable (< 3 bins with >= 10 of each kind)"
    M.loc[M.rule == rule, "verdict"] = v
save(M, os.path.join(ROOT, "results", "e10_matched_v2"), "summary_L10a_matched_v2.csv")
show(M, "L10a-matched, passive rerun (D4)")
