"""Sanity checks (c) and (d) from E5's decay table (PREREGISTRATION.md section 3).

(c) after abandonment, the signed displacement of overlap-home items, projected on its sign at abandonment (its
expectation decays exactly as (1 - delta)^t, so drift noise doesn't bias it), halves in ln 2 / delta steps (log-linear fit
over steps 0-5,000, within 10%). (d) with builders frozen but acting, the built share stays within 10% of its start.
    python analysis/decay_sanity.py <results dir containing decay.csv>
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import config as C  # noqa: E402

d = pd.read_csv(os.path.join(ROOT, "results", sys.argv[1], "decay.csv"))
rows = []
for (seed, rule, world), g in d.groupby(["seed", "rule", "world"]):
    g = g.sort_values("step"); f = g[g.step <= 5000]
    slope = np.polyfit(f.step, np.log(f.proj), 1)[0]
    half = np.log(2) / -slope
    b0 = g.built_share_overlap_frozen.iloc[0]
    rows.append(dict(seed=seed, rule=rule, world=world, half_time=half, expected=np.log(2) / C.DELTA,
                     c_pass=abs(half / (np.log(2) / C.DELTA) - 1) <= 0.10,
                     frozen_max_rel_change=float((g.built_share_overlap_frozen - b0).abs().max() / b0),
                     d_pass=float((g.built_share_overlap_frozen - b0).abs().max() / b0) <= 0.10,
                     built_share_end_abandoned=g.built_share_overlap.iloc[-1], built_share_start=g.built_share_overlap.iloc[0]))
s = pd.DataFrame(rows)
s.to_csv(os.path.join(ROOT, "results", sys.argv[1], "decay_sanity.csv"), index=False)
print(s.round(3).to_string(index=False))
