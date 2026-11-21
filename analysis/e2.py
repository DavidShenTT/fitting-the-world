"""E2, the upkeep law (PREREGISTRATION.md section 5 as changed by DEVIATIONS.md D2). Reads results/e2/traj.csv.

Built items = overlap-home items now outside every overlap zone, steady state = mean over the last 10,000 steps.
Primary fit: OLS of log(count) on log M, log(1/delta), log(width) and log(s_u) (nuisance), world-level points
(seed x cell). L2: slopes on M and 1/delta in [0.75, 1.25]; slope on width within +-0.25. Intervals: bootstrap over
worlds (seeds resampled, refit). Also reported: the fit without log(s_u); the sensitivity fit on the D2 subset; the fit
without any cell whose real-seed mean built share exceeds 0.95.
    python analysis/e2.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROOT, N_BOOT, save, show  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "src"))
import config as C  # noqa: E402

P = os.path.join(ROOT, "results", "e2")
t = pd.read_csv(os.path.join(P, "traj.csv"))
tail = t[t.step > t.step.max() - C.STEADY_TAIL]
pts = tail.groupby(["seed", "M", "s_u", "delta", "width"])[["built_count_overlap", "built_share_overlap", "n_overlap_home"]] \
          .mean().reset_index()
rg = json.load(open(os.path.join(ROOT, "results", "tuning", "e2", "range.json")))
sens = {tuple(c) for c in rg["sensitivity"]}
pts["sens"] = [(int(r.M), float(r.s_u), float(r.delta)) in sens for r in pts.itertuples()]
cellmean = pts.groupby(["M", "s_u", "delta", "width"]).built_share_overlap.transform("mean")
pts["saturated"] = cellmean > 0.95
save(pts, P, "points.csv")


def fit(d, su=True):
    X = [np.log(d.M), np.log(1 / d.delta), np.log(d.width)] + ([np.log(d.s_u)] if su else [])
    X = np.column_stack([np.ones(len(d))] + X)
    y = np.log(d.built_count_overlap)
    return np.linalg.lstsq(X, y, rcond=None)[0][1:]


def fit_ci(d, su=True, seed=0):
    est = fit(d, su)
    r = np.random.default_rng(seed)
    seeds = d.seed.unique()
    bs = []
    for _ in range(min(N_BOOT, 2000)):
        pick = r.choice(seeds, len(seeds))
        bs.append(fit(pd.concat([d[d.seed == s] for s in pick]), su))
    bs = np.array(bs)
    return est, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


rows = []
zero = int((pts.built_count_overlap <= 0).sum())
use = pts[pts.built_count_overlap > 0]
for name, d, su in (("primary (all eligible cells, s_u covariate)", use, True),
                    ("without s_u covariate", use, False),
                    ("sensitivity (tuning share in [0.2, 0.8])", use[use.sens], True),
                    ("without real cells above 0.95", use[~use.saturated], True)):
    est, lo, hi = fit_ci(d, su)
    r = dict(fit=name, n_points=len(d), n_cells=d.groupby(["M", "s_u", "delta", "width"]).ngroups,
             slope_M=est[0], M_lo=lo[0], M_hi=hi[0], slope_inv_delta=est[1], d_lo=lo[1], d_hi=hi[1],
             slope_width=est[2], w_lo=lo[2], w_hi=hi[2])
    if su:
        r.update(slope_s_u=est[3])
    r["L2"] = "pass" if (0.75 <= est[0] <= 1.25 and 0.75 <= est[1] <= 1.25 and abs(est[2]) <= 0.25) else "against"
    rows.append(r)
S = pd.DataFrame(rows)
save(S, P, "summary_E2.csv")
show(S, "E2: log-log slopes of the built-item count")
cells = use.groupby(["M", "s_u", "delta", "width"]).agg(share=("built_share_overlap", "mean"),
                                                       count=("built_count_overlap", "mean")).reset_index()
c32 = cells[cells.width == 32]
print(f"\nzero-count points dropped: {zero}; real cells with mean share > 0.95: {int(pts[pts.saturated].groupby(['M','s_u','delta','width']).ngroups)}")
print(f"corr(log M, log 1/delta) among real cells (width 32): {np.corrcoef(np.log(c32.M), np.log(1 / c32.delta))[0, 1]:.3f}")
show(cells, "cells: real-seed mean built share and count")
