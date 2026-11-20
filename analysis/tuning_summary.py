"""Summaries of the stage-2 tuning runs (seeds 7000-7099). Prints compact tables; writes results/tuning/*_summary.csv."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
R = os.path.join(ROOT, "results", "tuning")
pd.set_option("display.width", 200)


def leg():
    d = pd.concat([pd.read_csv(os.path.join(R, x, "leg.csv")) for x in ("leg", "leg_lam") if os.path.exists(os.path.join(R, x, "leg.csv"))])
    true = {"symmetric": (60, 60), "unequal": (60, 35)}
    d["hw_ok"] = [abs(r.hw_wide - true[r.geom][0]) <= 3 and abs(r.hw_narrow - true[r.geom][1]) <= 3 for r in d.itertuples()]
    d["pass"] = (d.wide_edge_acc >= 0.85) & (d.narrow_edge_acc >= 0.85) & (d.wide_decisive_ratio >= 0.6) & \
                (d.narrow_decisive_ratio >= 0.6) & d.hw_ok
    d["comp_ok"] = d.steps_to_comp.notna()
    cols = ["geom", "width", "seed", "wide_edge_acc", "narrow_edge_acc", "wide_decisive_ratio", "narrow_decisive_ratio",
            "hw_wide", "hw_narrow", "steps_to_comp", "pass"]
    if len(sys.argv) > 2:
        print(d[cols + ["lam"]].sort_values(["geom", "width", "lam", "seed"]).round(3).to_string(index=False))
    s = d.groupby(["geom", "width", "lam"]).agg(n_pass=("pass", "sum"), all_pass=("pass", "all"), comp_all=("comp_ok", "all")).reset_index()
    print(s.to_string(index=False))
    s.to_csv(os.path.join(R, "leg_summary.csv"), index=False)


def epsr():
    d = pd.read_csv(os.path.join(R, "epsr", "epsr.csv"))
    wild = d[d.kind == "wild"].groupby("cond").home_logloss.mean()
    m = d[d.kind != "wild"].groupby(["kind", "cond", "eps_r"]).home_logloss.mean().reset_index()
    rows = []
    for (k, c), g in m.groupby(["kind", "cond"]):
        g = g.assign(ratio=g.home_logloss / wild[c])
        best = g.iloc[(g.ratio - 1).abs().argmin()]
        rows.append(dict(kind=k, cond=c, wild_ll=wild[c], eps_r=best.eps_r, built_ll=best.home_logloss, ratio=best.ratio,
                         within_20pct=abs(best.ratio - 1) <= 0.2))
    s = pd.DataFrame(rows)
    print(s.round(4).to_string(index=False))
    s.to_csv(os.path.join(R, "epsr_summary.csv"), index=False)



def diag():
    """Legibility diagnostic: 3 agents per seed, 60k and 120k steps, lambda 0.4 / 0.3 / 0.2."""
    d = pd.read_csv(os.path.join(R, "leg_diag", "diag.csv"))
    tn = d.geom.map({"symmetric": 60, "unequal": 35})
    d["ok"] = (d.wide_edge_acc >= .85) & (d.narrow_edge_acc >= .85) & (d.wide_decisive_ratio >= .6) & \
              (d.narrow_decisive_ratio >= .6) & ((d.hw_wide - 60).abs() <= 3) & ((d.hw_narrow - tn).abs() <= 3)
    a = d.groupby(["geom", "lam", "steps"]).agg(agents_passing=("ok", "mean"), wide_acc=("wide_edge_acc", "mean"),
                                                narrow_acc=("narrow_edge_acc", "mean"), wide_dec=("wide_decisive_ratio", "mean"),
                                                narrow_dec=("narrow_decisive_ratio", "mean")).round(3)
    sm = d.groupby(["geom", "lam", "steps", "seed"])[["wide_edge_acc", "narrow_edge_acc", "wide_decisive_ratio",
                                                       "narrow_decisive_ratio", "hw_wide", "hw_narrow"]].mean().reset_index()
    tn = sm.geom.map({"symmetric": 60, "unequal": 35})
    sm["ok"] = (sm.wide_edge_acc >= .85) & (sm.narrow_edge_acc >= .85) & (sm.wide_decisive_ratio >= .6) & \
               (sm.narrow_decisive_ratio >= .6) & ((sm.hw_wide - 60).abs() <= 3) & ((sm.hw_narrow - tn).abs() <= 3)
    b = sm.groupby(["geom", "lam", "steps"]).ok.sum().rename("seeds_passing_on_3agent_mean")
    txt = a.join(b).to_string()
    print(txt)
    open(os.path.join(R, "leg_diag_summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    globals()[sys.argv[1]]()
