"""Print every figure used in PILOT6.md straight from the saved CSVs (and pilot 5's, where stated)."""
import numpy as np
import pandas as pd

pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
ra, q, hist = pd.read_csv("results/raised.csv"), pd.read_csv("results/q.csv"), pd.read_csv("results/hist.csv")
for d in (ra,):
    d["unsigned"] = 0.5 * (d.shift_wide.abs() + d.shift_narrow.abs())
    d["conf_err"] = 0.5 * (d.wide_conf_err + d.narrow_conf_err)
    d["edge_ll"] = 0.5 * (d.wide_edge_logloss + d.narrow_edge_logloss)
    d["stretch"] = 0.5 * (d.shift_narrow - d.shift_wide)   # mean amount arcs are wider than true, toward-45 convention
MAIN = ("skip002", "passive", "biased09")
CONDS_ORDER = ("skip002", "skip010", "skip030", "passive", "biased09")
cell = ra.groupby(["kind", "cond", "seed"]).mean(numeric_only=True)


def diff(kind, cond, col):
    return cell.loc[(kind, cond)][col] - cell.loc[("wild", cond)][col]   # per seed


print("== Worlds")
print(q[q.kind != "seconds"].groupby("kind")[["built_share", "edge_share"]].agg(["min", "max"]).round(3).to_string())
print("seconds per seed:", q[q.kind == "seconds"].built_share.round(0).tolist())

print("\n== Q1: biased offers, acceptance share of labels (per agent: overall min-max; per type mean)")
b = ra[ra.cond == "biased09"]
print(f"  overall {b.share_all.min():.3f}-{b.share_all.max():.3f}; wide {b.share_wide.mean():.3f}, narrow {b.share_narrow.mean():.3f}")
print("  label shares by condition (all kinds), wide / narrow:")
for cond, g in ra.groupby("cond"):
    print(f"    {cond:8s} {g.share_wide.mean():.3f} / {g.share_narrow.mean():.3f}")

print("\n== Q2: wild half-widths, biased offers minus passive feedback (per seed, mean of two agents)")
for col in ("hw_wide", "hw_narrow"):
    d = cell.loc[("wild", "biased09")][col] - cell.loc[("wild", "passive")][col]
    print(f"  {col}: mean {d.mean():+.2f} [{', '.join(f'{v:+.1f}' for v in d)}]; max |.| over seeds {d.abs().max():.2f}")
print("  wild half-widths by condition (wide / narrow): " + "; ".join(
    f"{c} {cell.loc[('wild', c)].hw_wide.mean():.1f}/{cell.loc[('wild', c)].hw_narrow.mean():.1f}" for c in CONDS_ORDER))

print("\n== Q3: refusal sources (sums over agents), acceptances per far-side refusal, and near-side share of refusals")
for kind in ("wild", "wear", "eye1"):
    for cond in CONDS_ORDER:
        g = ra[(ra.kind == kind) & (ra.cond == cond)]
        parts = []
        for t in ("wide", "narrow"):
            acc, far, near = g[f"{t}_acc"].sum(), g[f"{t}_far_ref"].sum(), g[f"{t}_near_ref"].sum()
            parts.append(f"{t} {acc / max(far, 1):6.1f}:1 (near {near / max(near + far, 1):.2f})")
        print(f"  {kind:5s} {cond:8s} " + "  ".join(parts))

print("\n== Q4: folded angle histograms, % of items per 5-degree bin (mean over seeds); 0 = wide centre, 90 = narrow centre")
cols = [c for c in hist.columns if c.startswith("b")]
hm = hist.groupby(["geometry", "kind"])[cols].mean() * 100
print(hm.round(1).to_string())
sw = hm.loc[("symmetric", "wear")]
dens = lambda lo, hi: sw[[f"b{x:02d}" for x in range(lo, hi, 5)]].mean()
print(f"  symmetric wear: mean % per bin in [50,60) {dens(50, 60):.2f}, [60,90) {dens(60, 90):.2f}, [30,50) {dens(30, 50):.2f}, [0,30) {dens(0, 30):.2f}")

print("\n== Cell means (twelve agents)")
print(ra.groupby(["kind", "cond"])[["shift_wide", "shift_narrow", "unsigned", "stretch", "conf_err", "edge_ll", "home_logloss", "acts"]]
      .mean().round(3).to_string())

print("\n== L10a (primary), built vs wild under the three main conditions: mean over seeds")
for kind in ("wear", "eye1"):
    for cond in MAIN:
        cb, cw = cell.loc[(kind, cond)].conf_err.mean(), cell.loc[("wild", cond)].conf_err.mean()
        lb, lw = cell.loc[(kind, cond)].edge_ll.mean(), cell.loc[("wild", cond)].edge_ll.mean()
        hb, hw = cell.loc[(kind, cond)].home_logloss.mean(), cell.loc[("wild", cond)].home_logloss.mean()
        rc = cell.loc[(kind, cond)].conf_err / cell.loc[("wild", cond)].conf_err
        print(f"  {kind:5s} {cond:8s} conf err {cb:.3f} vs {cw:.3f} ({cb / cw:.2f}x; per seed {rc.min():.2f}-{rc.max():.2f})  "
              f"edge log loss {lb:.3f} vs {lw:.3f} ({lb / lw:.2f}x)  home ll {hb:.3f} vs {hw:.3f}")

print("\n== L10-shift (skip rule): pooled unsigned shift, built minus wild")
for kind in ("wear", "eye1"):
    d = diff(kind, "skip002", "unsigned")
    print(f"  {kind}: {d.mean():.2f} [{', '.join(f'{v:.1f}' for v in d)}]")

print("\n== L10m: mean stretch, built minus wild (degrees; positive = built arcs wider)")
for kind in ("wear", "eye1"):
    st = {c: diff(kind, c, "stretch") for c in CONDS_ORDER}
    print(f"  {kind}: " + "  ".join(f"{c} {st[c].mean():+.2f}" for c in CONDS_ORDER))
    pos = (st["biased09"].mean() - st["passive"].mean()) / (st["skip002"].mean() - st["passive"].mean())
    print(f"    biased offers' position on passive (0) -> skip (1): {pos:.2f}; "
          f"eps_min 0.02 > 0.1 > 0.3: {st['skip002'].mean() > st['skip010'].mean() > st['skip030'].mean()}")
    print(f"    wide / narrow shifts toward 45 (built minus wild): " + "  ".join(
        f"{c} {diff(kind, c, 'shift_wide').mean():+.1f}/{diff(kind, c, 'shift_narrow').mean():+.1f}" for c in CONDS_ORDER))

print("\n== L10-dose, confident-error condition, from pilot 5's saved dose arms (skip rule)")
p5 = pd.read_csv("../pilot5/results/raised.csv")
p5 = p5[(p5.geometry == "unequal") & (p5.cond == "skip")].copy()
p5["conf_err"] = 0.5 * (p5.wide_conf_err + p5.narrow_conf_err)
cw = p5[p5.kind == "wild"].conf_err.mean()
for kind in ("dose003", "dose010", "eye1"):
    print(f"  {kind}: confident error {p5[p5.kind == kind].conf_err.mean():.3f} vs wild {cw:.3f} ({p5[p5.kind == kind].conf_err.mean() / cw:.2f}x)")

print("\n== Q5: variance split for built-minus-wild contrasts (main conditions, wear and eye1)")
def split(d, n_per_seed, label):
    d = d.copy()
    d["unsigned"] = 0.5 * (d.shift_wide.abs() + d.shift_narrow.abs())
    d["stretch"] = 0.5 * (d.shift_narrow - d.shift_wide)
    d["conf_err"] = 0.5 * (d.wide_conf_err + d.narrow_conf_err)
    for col in ("unsigned", "stretch", "conf_err"):
        g = d.groupby(["kind", "cond", "seed"])[col]
        within = np.sqrt((g.std() ** 2).mean())
        seedmeans = g.mean().groupby(level=["kind", "cond"]).std()
        sm = np.sqrt((seedmeans ** 2).mean())
        between = np.sqrt(max(sm ** 2 - within ** 2 / n_per_seed, 0.0))
        print(f"  {label} {col:9s} agent-to-agent sd {within:.3f}; sd of seed means {sm:.3f}; implied world-to-world sd {between:.3f}")
split(ra[ra.cond.isin(MAIN) & ra.kind.isin(["wild", "wear", "eye1"])], 2, "pilot 6")
p5all = pd.read_csv("../pilot5/results/raised.csv")
split(p5all[(p5all.geometry == "unequal") & p5all.cond.isin(["skip", "passive"]) & p5all.kind.isin(["wild", "wear", "eye1"])], 3, "pilot 5")

print("\n== Exploratory: raw mean stretch (not differenced) against acceptances per far-side refusal (wide actions), by condition")
for kind in ("wild", "wear", "eye1"):
    st, ratio = [], []
    for cond in CONDS_ORDER:
        g = ra[(ra.kind == kind) & (ra.cond == cond)]
        st.append(g.stretch.mean()); ratio.append(g.wide_acc.sum() / max(g.wide_far_ref.sum(), 1))
    rho = pd.Series(st).rank().corr(pd.Series(ratio).rank())
    print(f"  {kind:5s} " + "  ".join(f"{c} {s:+.2f} @ {r:.1f}:1" for c, s, r in zip(CONDS_ORDER, st, ratio)) + f"   Spearman {rho:.2f}")

print("\n== Design: standard error of a built-minus-wild contrast with 8 seeds x 3 agents per kind (pilot 6 variance components)")
d6 = ra[ra.cond.isin(MAIN)].copy()
for col in ("unsigned", "stretch"):
    g = d6.groupby(["kind", "cond", "seed"])[col]
    within = np.sqrt((g.std() ** 2).mean()); sm = np.sqrt((g.mean().groupby(level=["kind", "cond"]).std() ** 2).mean())
    between2 = max(sm ** 2 - within ** 2 / 2, 0.0)
    print(f"  {col}: {np.sqrt(2 * (between2 / 8 + within ** 2 / 24)):.2f} deg")
