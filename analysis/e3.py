"""E3, who benefits (PREREGISTRATION.md section 6). Reads results/e3/e3.csv.

Per world (seed x rule): category means over maps and newborns (same body, same type, outsider). Ratios are ratios of
the means over worlds; bootstrap intervals resample worlds.
    python analysis/e3.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, N_BOOT, save, show  # noqa: E402

P = os.path.join(ROOT, "results", sys.argv[1] if len(sys.argv) > 1 else "e3")
d = pd.read_csv(os.path.join(P, "e3.csv"))
d["h_drop"] = d.h_natural - d.h_built
d["area_gain"] = d.area_built - d.area_natural
d["ll_drop"] = d.ll2k_natural - d.ll2k_built
d["code_saving"] = d.code_natural - d.code_built
d["speedup"] = d.steps_to_crit_natural / d.steps_to_crit_built
MEAS = ["h_drop", "area_gain", "ll_drop", "code_saving", "speedup", "area_natural", "area_built",
        "steps_to_crit_natural", "steps_to_crit_built", "code_natural", "code_built"]
WM = d.groupby(["seed", "rule", "arm"])[MEAS].mean().reset_index()
save(WM, P, "world_means.csv")


def ratio_ci(rule, col, num="same_body", den="outsider", seed=0):
    p = WM[WM.rule == rule].pivot(index="seed", columns="arm", values=col)
    est = p[num].mean() / p[den].mean()
    r = np.random.default_rng(seed)
    bs = [p.iloc[i][num].mean() / p.iloc[i][den].mean() for i in r.integers(0, len(p), (N_BOOT, len(p)))]
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5), p


rows = []
for rule in ("eye025", "wear"):
    m = WM[WM.rule == rule].groupby("arm")[MEAS].mean()
    # L3c
    _, lo, hi, p = ratio_ci(rule, "h_drop")
    sb, ot = m.loc["same_body", "h_drop"], m.loc["outsider", "h_drop"]
    rows.append(dict(rule=rule, prediction="L3c (primary): entropy drop, bits", same_body=sb, same_type=m.loc["same_type", "h_drop"],
                     outsider=ot, ratio_same_over_out=sb / ot if ot != 0 else np.inf, ci_lo=lo, ci_hi=hi,
                     verdict=("pass" if sb >= 0.5 and ot <= 0.5 * sb else ("against" if ot >= 0.8 * sb else "neither"))
                     if rule == "eye025" else "(control)"))
    for pred, col in (("L3: early learning-curve area gain", "area_gain"), ("L3b: home log-loss drop at 2,000 steps", "ll_drop"),
                      ("L3d: code-length saving, bits", "code_saving"), ("speed-up (steps to 0.9), reported", "speedup")):
        est, lo, hi, _ = ratio_ci(rule, col)
        if rule == "eye025" and col != "speedup":
            v = "pass" if est >= 1.5 else ("against" if (col == "area_gain" and abs(est - 1) <= 0.2) else "fail")
        elif rule == "wear" and col == "area_gain":
            v = "pass (within 20%)" if abs(est - 1) <= 0.2 else "fail (not within 20%)"
        else:
            v = "(reported)"
        rows.append(dict(rule=rule, prediction=pred, same_body=m.loc["same_body", col], same_type=m.loc["same_type", col],
                         outsider=m.loc["outsider", col], ratio_same_over_out=est, ci_lo=lo, ci_hi=hi, verdict=v))
S = pd.DataFrame(rows)
save(S, P, "summary_E3.csv")
show(S, "E3 (builders at kappa = 0.25; wear as the control)")
show(d.groupby(["rule", "arm"])[["steps_to_crit_natural", "steps_to_crit_built"]].agg(lambda s: s.isna().mean()).reset_index(),
     "share of newborns not reaching competence 0.9 within 20,000 steps")
W = pd.read_csv(os.path.join(P, "worlds.csv"))
show(W.groupby("rule")[["built_share", "edge_share", "edge_share_natural", "u1_move", "u2_move"]].mean().reset_index(),
     "E3 built worlds (u moves: builders read u1 in even seeds, u2 in odd seeds)")
