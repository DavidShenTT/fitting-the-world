"""E8 against PREREGISTRATION.md section 8 (width ladder 4-64, DEVIATIONS.md D1). Reads results/<dir>/e8.csv.

    python analysis/e8.py [results subdir, default e8]
Writes results/<dir>/summary_E8.csv, summary_E8_matching.csv, summary_E8_traj.csv and prints them.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, boot_ci, save, show, world_means  # noqa: E402

D = sys.argv[1] if len(sys.argv) > 1 else "e8"
P = os.path.join(ROOT, "results", D)
d = pd.read_csv(os.path.join(P, "e8.csv"))
d["fall"] = d.edge_acc_before - d.edge_acc_after
WM = world_means(d, ["seed", "width", "arm"], ["fall", "edge_acc_before", "edge_acc_after", "acts_T2", "offered_edge_share",
                                               "plateau_at"])
piv = WM.pivot_table(index=["seed", "width"], columns="arm", values="fall").reset_index()
widths = sorted(d.width.unique())
rows = []
for rule in ("wear", "eye1"):
    piv[f"extra_{rule}"] = piv[f"treat_{rule}"] - piv["natural"]
    piv[f"vsfilt_{rule}"] = piv[f"treat_{rule}"] - piv[f"filtered_{rule}"]
    for wd in widths:
        g = piv[piv.width == wd]
        ex, vf = g[f"extra_{rule}"], g[f"vsfilt_{rule}"]
        rows.append(dict(rule=rule, width=wd, fall_treat=g[f"treat_{rule}"].mean(), fall_filtered=g[f"filtered_{rule}"].mean(),
                         fall_natural=g["natural"].mean(), extra_fall=ex.mean(), extra_lo=boot_ci(ex)[0], extra_hi=boot_ci(ex)[1],
                         treat_minus_filtered=vf.mean(), tmf_lo=boot_ci(vf)[0], tmf_hi=boot_ci(vf)[1], n_worlds=len(g)))
S = pd.DataFrame(rows)
nar, wid = widths[0], widths[-1]
for rule in ("wear", "eye1"):
    s = S[S.rule == rule].set_index("width")
    e_n, e_w = s.loc[nar, "extra_fall"], s.loc[wid, "extra_fall"]
    tmf = s.loc[nar, "treat_minus_filtered"]
    S.loc[S.rule == rule, "L8a"] = "pass" if e_n >= 0.10 else ("against" if e_n < 0.03 else "neither")
    S.loc[S.rule == rule, "L8a_prime"] = ("within 0.03 (expected)" if abs(tmf) <= 0.03 else
                                          ("building-specific (>= 0.05 more)" if tmf >= 0.05 else "between"))
    S.loc[S.rule == rule, "L8b_ratio"] = e_n / e_w if e_w != 0 else np.inf
    S.loc[S.rule == rule, "L8b"] = "pass" if (e_w > 0 and e_n / e_w >= 2) or (e_w <= 0 < e_n) else "fail"
save(S, P, "summary_E8.csv"); show(S, f"E8 (L8a at width {nar}, L8b {nar} vs {wid}; L8a' judged at width {nar})")
mrows = []
for rule in ("wear", "eye1"):
    for wd in widths:
        t = WM[(WM.width == wd) & (WM.arm == f"treat_{rule}")].set_index("seed")
        f = WM[(WM.width == wd) & (WM.arm == f"filtered_{rule}")].set_index("seed")
        mrows.append(dict(rule=rule, width=wd, edge_offered_treat=t.offered_edge_share.mean(),
                          edge_offered_filtered=f.offered_edge_share.mean(),
                          max_abs_diff_points=float(100 * (t.offered_edge_share - f.offered_edge_share).abs().max()),
                          acts_ratio_treat_filtered=float((t.acts_T2 / f.acts_T2).mean()),
                          plateau_mean=WM[WM.width == wd].plateau_at.mean(), capped=int(d[d.width == wd].plateau_at.isna().sum())))
M = pd.DataFrame(mrows); save(M, P, "summary_E8_matching.csv"); show(M, "E8 matching checks (edge share within 0.5 points; acts within 10%)")
T = pd.read_csv(os.path.join(P, "e8_traj.csv"))
TS = T.groupby(["arm", "width", "step"])[["built_share", "edge_acc"]].mean().reset_index()
save(TS, P, "summary_E8_traj.csv")
show(TS[TS.width == nar].pivot_table(index="step", columns="arm", values=["built_share", "edge_acc"]).reset_index().iloc[::3],
     f"population built share vs individual edge accuracy, width {nar} (every 3rd checkpoint)")
