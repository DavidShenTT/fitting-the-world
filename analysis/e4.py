"""E4 (routes), E5 (decay) and E6 (bodies), PREREGISTRATION.md section 7. Reads results/e4/ and results/e6/.

Agreement is Cohen's kappa on the argmax-chosen action; world means over pairs first, then means over worlds.
    python analysis/e4.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, boot_ci, save, show  # noqa: E402

P4, P6 = (os.path.join(ROOT, "results", x) for x in (sys.argv[1:3] if len(sys.argv) > 2 else ("e4", "e6")))
pr = pd.read_csv(os.path.join(P4, "pairs.csv"))
COLS = ["kappa_overlap", "kappa_unamb", "agree_overlap", "agree_unamb", "kappa_overlap_test", "kappa_unamb_test", "e_k", "acc_a"]
WM = pr.groupby(["seed", "rule", "phase", "kind"])[COLS].mean().reset_index()
save(WM, P4, "world_means.csv")


def piv(rule, phase, col):
    return WM[(WM.rule == rule) & (WM.phase == phase)].pivot(index="seed", columns="kind", values=col)


rows = []
for rule in ("wear", "eye1"):
    ko, ku, ek = piv(rule, "kept", "kappa_overlap"), piv(rule, "kept", "kappa_unamb"), piv(rule, "kept", "e_k")
    same_adv, diff_adv = ko["same"] - ko["natural"], ko["different"] - ko["natural"]
    un = ku.mean()
    l4 = (same_adv.mean() >= 0.2) and (diff_adv.mean() <= 0.05) and (un.min() >= 0.9) and (un.max() - un.min() <= 0.05)
    l5d = ek["same"] - ek["different"]
    lo5, hi5 = boot_ci(l5d)
    ab = piv(rule, "abandoned", "kappa_overlap")
    l6 = ab["same"] - ko["natural"]
    rows.append(dict(rule=rule, k_ov_same=ko["same"].mean(), k_ov_diff=ko["different"].mean(), k_ov_nat=ko["natural"].mean(),
                     same_minus_nat=same_adv.mean(), same_lo=boot_ci(same_adv)[0], same_hi=boot_ci(same_adv)[1],
                     diff_minus_nat=diff_adv.mean(), diff_lo=boot_ci(diff_adv)[0], diff_hi=boot_ci(diff_adv)[1],
                     k_un_same=un["same"], k_un_diff=un["different"], k_un_nat=un["natural"],
                     L4="pass" if l4 else ("against" if same_adv.mean() <= 0 else "fail"),
                     ek_same=ek["same"].mean(), ek_diff=ek["different"].mean(), ek_nat=ek["natural"].mean(),
                     L5_diff=l5d.mean(), L5_lo=lo5, L5_hi=hi5, L5="pass" if (l5d.mean() > 0 and lo5 > 0) else "fail",
                     k_ov_same_abandoned=ab["same"].mean(), k_ov_diff_abandoned=ab["different"].mean(),
                     L6_adv=l6.mean(), L6_lo=boot_ci(l6)[0], L6_hi=boot_ci(l6)[1], L6="pass" if l6.mean() < 0.05 else "fail"))
S = pd.DataFrame(rows)
save(S, P4, "summary_E4_E5.csv")
show(S, "E4 (L4, L5) and E5 (L6)")
show(WM[WM.phase == "kept"].groupby(["rule", "kind"])[["kappa_overlap_test", "kappa_unamb_test", "agree_overlap", "acc_a"]]
     .mean().reset_index(), "secondary readout on the natural test set; raw agreement; acceptance of chosen actions")
W = pd.read_csv(os.path.join(P4, "worlds.csv"))
show(W.groupby(["rule", "world"])[["built_share_overlap", "edge_share"]].mean().reset_index(), "E4 built worlds")

e6 = pd.read_csv(os.path.join(P6, "pairs.csv"))
E6W = e6.groupby(["seed", "kind"])[["kappa_overlap", "kappa_unamb", "e_k"]].mean().reset_index()
rows = []
for k, g in E6W.groupby("kind"):
    lo, hi = boot_ci(g.kappa_overlap)
    rows.append(dict(kind=k, kappa_overlap=g.kappa_overlap.mean(), lo=lo, hi=hi, above_chance=lo > 0,
                     kappa_unamb=g.kappa_unamb.mean(), e_k=g.e_k.mean(), n_worlds=len(g)))
E6 = pd.DataFrame(rows)
save(E6, P6, "summary_E6.csv")
show(E6, "E6 (natural world; no prediction): agreement on overlap-home items")
