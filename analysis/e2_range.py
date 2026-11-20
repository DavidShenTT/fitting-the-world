"""E2's cells from the tuning sweep (DEVIATIONS.md D2, replacing PREREGISTRATION.md section 5's block rule).

Eligible cell: mean steady built share over tuning seeds 7020-7022 in [0.1, 0.9] and no tuning seed above 0.95. All
eligible cells enter the joint OLS. The sensitivity subset is the eligible cells with tuning mean in [0.2, 0.8]. The width
cell is the eligible cell whose tuning mean is closest to 0.5 (ties: smaller M). Writes results/tuning/e2/cells.csv and
range.json, and prints the corr(log M, log 1/delta) among chosen cells.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import config as C  # noqa: E402

P = os.path.join(ROOT, "results", "tuning", "e2")

t = pd.read_csv(os.path.join(P, "traj.csv"))
tail = t[t.step > t.step.max() - C.STEADY_TAIL]
per_seed = tail.groupby(["M", "s_u", "delta", "seed"]).built_share_overlap.mean().reset_index()
cells = per_seed.groupby(["M", "s_u", "delta"]).built_share_overlap.agg(["mean", "max", "min"]).reset_index()
cells["eligible"] = (cells["mean"] >= 0.1) & (cells["mean"] <= 0.9) & (cells["max"] <= 0.95)
cells["sensitivity"] = cells.eligible & (cells["mean"] >= 0.2) & (cells["mean"] <= 0.8)
cells.to_csv(os.path.join(P, "cells.csv"), index=False)
el = cells[cells.eligible].copy()
mid = el.assign(dist=(el["mean"] - 0.5).abs()).sort_values(["dist", "M"]).iloc[0]


def corr(d):
    return float(np.corrcoef(np.log(d.M), np.log(1 / d.delta))[0, 1])


rg = dict(cells=[[int(r.M), float(r.s_u), float(r.delta)] for r in el.itertuples()],
          sensitivity=[[int(r.M), float(r.s_u), float(r.delta)] for r in el[el.sensitivity].itertuples()],
          width_cell=[int(mid.M), float(mid.s_u), float(mid.delta)])
json.dump(rg, open(os.path.join(P, "range.json"), "w"), indent=1)
print(f"eligible cells: {len(el)} (s_u levels {sorted(el.s_u.unique())}); sensitivity subset: {int(el.sensitivity.sum())}")
print(f"corr(log M, log 1/delta): all eligible {corr(el):.3f}; sensitivity subset {corr(el[el.sensitivity]):.3f}")
for su, g in el.groupby("s_u"):
    if len(g) > 2:
        print(f"  within s_u = {su:g}: n = {len(g)}, corr = {corr(g):.3f}")
print("width cell (M, s_u, delta):", rg["width_cell"], f"tuning mean {mid['mean']:.3f}")
