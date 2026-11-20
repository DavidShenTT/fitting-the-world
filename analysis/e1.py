"""E1 and the wear sanity checks (PREREGISTRATION.md sections 3-4). Reads results/<dir>/traj.csv.

    python analysis/e1.py e1                    # real E1 (seeds 0-9): L1 at kappa = 1 and kappa = 0.25
    python analysis/e1.py tuning/sanity_wear    # sanity checks (a) and (b)
Writes <dir>/runs_steady.csv and <dir>/summary.csv, and prints them.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import config as C  # noqa: E402

pd.set_option("display.width", 200)


def steady(g):
    """Steady-state rule: first 5,000-step window whose mean built share differs from the previous window's by < 0.02;
    value = mean over the last 10,000 steps."""
    g = g.sort_values("step")
    last = g.step.max()
    win = ((g.step - 1) // C.STEADY_WINDOW).astype(int)
    means = g.groupby(win).built_share_overlap.mean()
    reached = np.nan
    for k in range(1, len(means)):
        if abs(means.iloc[k] - means.iloc[k - 1]) < C.STEADY_TOL:
            reached = (means.index[k] + 1) * C.STEADY_WINDOW
            break
    tail = g[g.step > last - C.STEADY_TAIL]
    return pd.Series(dict(steady_step=reached, settled=not np.isnan(reached),
                          built_share=tail.built_share_overlap.mean(), built_count=tail.built_count_overlap.mean(),
                          u1_move=tail.u1_move.mean(), u2_move=tail.u2_move.mean(), edge_share=tail.edge_share.mean(),
                          app_grad=g.app_grad.mean()))


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    m = x[r.integers(0, len(x), (n, len(x)))].mean(axis=1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def main(d):
    path = os.path.join(ROOT, "results", d)
    t = pd.read_csv(os.path.join(path, "traj.csv"))
    runs = t.groupby(["seed", "M", "rule", "delta"]).apply(steady, include_groups=False).reset_index()
    runs.to_csv(os.path.join(path, "runs_steady.csv"), index=False)
    rows = []
    if "eye1" in set(runs.rule):
        for M, g in runs.groupby("M"):
            p = g.pivot(index="seed", columns="rule", values="built_share")
            diff = p["eye1"] - p["none"]
            lo, hi = boot_ci(diff)
            u = g[g.rule == "eye025"].u1_move
            ulo, uhi = boot_ci(u)
            rows.append(dict(M=M, n_worlds=len(diff), built_eye1=p["eye1"].mean(), built_control=p["none"].mean(),
                             L1_diff=diff.mean(), L1_ci_lo=lo, L1_ci_hi=hi,
                             L1_verdict="pass" if diff.mean() >= 0.3 else ("against" if diff.mean() <= 0.1 else "neither"),
                             u1_move_eye025=u.mean(), u1_ci_lo=ulo, u1_ci_hi=uhi,
                             L1_025_verdict="pass" if u.mean() >= 0.3 else ("against" if u.mean() < 0.1 else "neither"),
                             u2_move_eye025=g[g.rule == "eye025"].u2_move.mean(),
                             app_grad_eye025=g[g.rule == "eye025"].app_grad.mean(),
                             settled_share=g.settled.mean(), steady_step_max=g.steady_step.max()))
    else:  # sanity checks (a) wear vs control at M = 6, delta 0.0005; (b) built share falls as delta rises
        base = runs[(runs.rule == "wear") & np.isclose(runs.delta, C.DELTA)].set_index("seed").built_share
        ctrl = runs[runs.rule == "none"].set_index("seed").built_share
        bydelta = runs[runs.rule == "wear"].groupby("delta").built_share.mean()
        rows.append(dict(check="a_wear_minus_control", value=float((base - ctrl).mean()),
                         passed=bool((base - ctrl).mean() >= 0.3)))
        rows.append(dict(check="b_wear_falls_with_delta", value=";".join(f"{k:g}:{v:.3f}" for k, v in bydelta.items()),
                         passed=bool(np.all(np.diff(bydelta.values) <= 0))))
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(path, "summary.csv"), index=False)
    print(s.round(3).to_string(index=False))
    print("\nper-run (built share, u1 move):")
    print(runs.groupby(["M", "rule", "delta"])[["built_share", "u1_move", "u2_move", "settled"]].mean().round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
