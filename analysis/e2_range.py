"""E2's test range from the tuning sweep (PREREGISTRATION.md section 5), seeds 7020-7022.

A cell is eligible if its mean steady built share is in [0.1, 0.9] and no tuning seed exceeds 0.95. The test range is the
largest full factorial block (>= 3 levels of M and >= 3 of delta, >= 1 of s_u) of eligible cells; ties go to more s_u
levels, then more M levels. Writes results/tuning/e2/cells.csv and range.json, and prints both.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import config as C  # noqa: E402

P = os.path.join(ROOT, "results", "tuning", "e2")


def subsets(levels, kmin):
    for k in range(len(levels), kmin - 1, -1):
        yield from itertools.combinations(levels, k)


def main():
    t = pd.read_csv(os.path.join(P, "traj.csv"))
    tail = t[t.step > t.step.max() - C.STEADY_TAIL]
    per_seed = tail.groupby(["M", "s_u", "delta", "seed"]).built_share_overlap.mean().reset_index()
    cells = per_seed.groupby(["M", "s_u", "delta"]).built_share_overlap.agg(["mean", "max", "min"]).reset_index()
    cells["eligible"] = (cells["mean"] >= 0.1) & (cells["mean"] <= 0.9) & (cells["max"] <= 0.95)
    cells.to_csv(os.path.join(P, "cells.csv"), index=False)
    ok = {(r.M, r.s_u, r.delta) for r in cells.itertuples() if r.eligible}
    Ms, Ss, Ds = (sorted(cells[c].unique()) for c in ("M", "s_u", "delta"))
    best = None
    for ms in subsets(Ms, 3):
        for ds in subsets(Ds, 3):
            for ss in subsets(Ss, 1):
                if all((m, s, d) in ok for m in ms for s in ss for d in ds):
                    score = (len(ms) * len(ss) * len(ds), len(ss), len(ms))
                    if best is None or score > best[0]:
                        best = (score, ms, ss, ds)
    print(cells.pivot_table(index=["s_u", "M"], columns="delta", values="mean").round(2).to_string())
    if best is None:
        print("\nNO BLOCK: extend the grid (s_u 0.003, delta 0.008) by the preregistered rule")
        return
    _, ms, ss, ds = best
    mid = lambda xs: [float(v) for v in xs][(len(xs) - 1) // 2]
    rg = dict(M=[int(m) for m in ms], s_u=[float(s) for s in ss], delta=[float(d) for d in ds],
              M_mid=int(mid(ms)), s_u_mid=mid(ss), delta_mid=mid(ds))
    json.dump(rg, open(os.path.join(P, "range.json"), "w"), indent=1)
    print("\nrange:", rg)


if __name__ == "__main__":
    main()
