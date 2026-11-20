"""Print every figure used in PILOT5.md straight from the saved CSVs."""
import numpy as np
import pandas as pd

pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
ra, q = pd.read_csv("results/raised.csv"), pd.read_csv("results/q.csv")
ra["unsigned"] = 0.5 * (ra.shift_wide.abs() + ra.shift_narrow.abs())
ra["conf_err"] = 0.5 * (ra.wide_conf_err + ra.narrow_conf_err)
ra["g"] = 0.5 * (ra.shift_wide + ra.shift_narrow)
ra["l"] = 0.5 * (ra.shift_narrow - ra.shift_wide)
COLS = ["shift_wide", "shift_narrow", "unsigned", "conf_err", "home_logloss", "share_wide", "share_narrow"]
cell = ra.groupby(["geometry", "kind", "cond", "seed"])[COLS + ["g", "l"]].mean()   # mean of three agents per seed

print("== Worlds")
print(q[q.geometry.isin(["unequal", "symmetric"])].round(3).to_string(index=False))
p5 = q[q.geometry == "p5"].pivot(index="seed", columns="kind", values="edge_share")
print("P5 offered edge share, built (wear) vs redrawn filtered sampler, by seed:")
print((100 * p5).round(2).to_string(), f"\n  max |difference| {100 * (p5.filtered - p5.treat_wear).abs().max():.2f} points")
print("seconds per seed:", q[q.geometry == "seconds"].built_share.round(0).tolist())

print("\n== P1 / P2: wild agents under the skip rule, by seed (mean of three)")
for geom in ("unequal", "symmetric"):
    x = ra[(ra.geometry == geom) & (ra.kind == "wild") & (ra.cond == "skip")].groupby("seed")[
        ["wide_edge_acc", "narrow_edge_acc", "hw_wide", "hw_narrow"]].mean()
    print(geom); print(x.round(2).to_string())

print("\n== P3: weighted label acceptance share, mean [min, max] over agents")
for (geom, cond), g in ra.groupby(["geometry", "cond"]):
    print(f"  {geom:9s} {cond:7s} wide {g.share_wide.mean():.3f} [{g.share_wide.min():.3f}, {g.share_wide.max():.3f}]  "
          f"narrow {g.share_narrow.mean():.3f} [{g.share_narrow.min():.3f}, {g.share_narrow.max():.3f}]")

print("\n== Cell means (nine agents)")
print(ra.groupby(["geometry", "kind", "cond"])[COLS].mean().round(3).to_string())


def diff(geom, kind, cond, col):
    b, w = cell.loc[(geom, kind, cond)][col], cell.loc[(geom, "wild", cond)][col]
    return b - w  # per seed


print("\n== L10a, unequal world: built minus wild, mean over seeds [per-seed values]")
for kind in ("wear", "eye1"):
    for cond in ("skip", "passive", "s05", "s09"):
        du = diff("unequal", kind, cond, "unsigned")
        cb, cw = cell.loc[("unequal", kind, cond)].conf_err, cell.loc[("unequal", "wild", cond)].conf_err
        hb, hw = cell.loc[("unequal", kind, cond)].home_logloss, cell.loc[("unequal", "wild", cond)].home_logloss
        print(f"  {kind:5s} {cond:7s} pooled unsigned {du.mean():5.2f} [{', '.join(f'{v:.1f}' for v in du)}]  "
              f"conf err {cb.mean():.3f} vs {cw.mean():.3f} (ratio {cb.mean() / cw.mean():.2f})  "
              f"home ll {hb.mean():.3f} vs {hw.mean():.3f}")

print("\n== L10s, unequal world: built minus wild shift toward 45 (wide W, narrow N), G = (W+N)/2, L = (N-W)/2")
for kind in ("wear", "eye1"):
    for cond in ("passive", "s05", "s09", "skip"):
        dw, dn = diff("unequal", kind, cond, "shift_wide"), diff("unequal", kind, cond, "shift_narrow")
        G, L = 0.5 * (dw + dn), 0.5 * (dn - dw)
        sh = cell.loc[("unequal", kind, cond)]
        print(f"  {kind:5s} {cond:7s} W {dw.mean():+5.2f}  N {dn.mean():+5.2f}  G {G.mean():+5.2f} [{', '.join(f'{v:+.1f}' for v in G)}]  "
              f"L {L.mean():+5.2f} [{', '.join(f'{v:+.1f}' for v in L)}]  built label share {sh.share_wide.mean():.2f}/{sh.share_narrow.mean():.2f}")
    s09w, skw = diff("unequal", kind, "s09", "shift_wide").mean(), diff("unequal", kind, "skip", "shift_wide").mean()
    s09n, skn = diff("unequal", kind, "s09", "shift_narrow").mean(), diff("unequal", kind, "skip", "shift_narrow").mean()
    print(f"  {kind}: s = 0.9 minus skip: wide {s09w - skw:+.2f}, narrow {s09n - skn:+.2f}")

print("\n== Raw shifts toward 45 by kind (not differenced), unequal world")
print(ra[ra.geometry == "unequal"].groupby(["cond", "kind"])[["shift_wide", "shift_narrow"]].mean().round(2).unstack("kind").to_string())

print("\n== Symmetric world: built (wear) minus wild, shifts inward-positive on every arc")
for cond in ("skip", "passive", "s05"):
    dw, dn = diff("symmetric", "wear", cond, "shift_wide"), diff("symmetric", "wear", cond, "shift_narrow")
    sb, sw = cell.loc[("symmetric", "wear", cond)], cell.loc[("symmetric", "wild", cond)]
    print(f"  {cond:7s} actions 0/2 {dw.mean():+5.2f}  actions 1/3 {dn.mean():+5.2f}  pooled {(0.5 * (dw + dn)).mean():+5.2f} "
          f"[{', '.join(f'{v:+.1f}' for v in 0.5 * (dw + dn))}]  label share built {sb.share_wide.mean():.2f}/{sb.share_narrow.mean():.2f}, "
          f"wild {sw.share_wide.mean():.2f}/{sw.share_narrow.mean():.2f}")
print("  raw shifts:")
print(ra[ra.geometry == "symmetric"].groupby(["cond", "kind"])[["shift_wide", "shift_narrow"]].mean().round(2).unstack("kind").to_string())

print("\n== Dose: pooled unsigned shift difference against edge share removed (unequal world)")
qe = q[q.geometry == "unequal"].groupby("kind").edge_share.mean()
for cond in ("skip", "passive"):
    for kind in ("dose003", "dose010", "eye1"):
        du = diff("unequal", kind, cond, "unsigned")
        print(f"  {cond:7s} {kind:7s} edge share {100 * qe[kind]:5.1f}% (removed {100 * (qe['wild'] - qe[kind]):5.1f} points)  "
              f"pooled unsigned diff {du.mean():5.2f} [{', '.join(f'{v:.1f}' for v in du)}]  "
              f"W {diff('unequal', kind, cond, 'shift_wide').mean():+.2f}  N {diff('unequal', kind, cond, 'shift_narrow').mean():+.2f}")

print("\n== Power: agent-to-agent spread within cells (unequal world, four main conditions, wild/wear/eye1)")
sub = ra[(ra.geometry == "unequal") & ra.kind.isin(["wild", "wear", "eye1"])]
sds = sub.groupby(["kind", "cond", "seed"])[["shift_wide", "shift_narrow", "g", "l", "unsigned"]].std()
sd = np.sqrt((sds ** 2).mean())
print("  pooled within-cell sd per agent: " + ", ".join(f"{k} {v:.2f}" for k, v in sd.items()))
for k in ("g", "l", "unsigned"):
    print(f"  agents per kind for a built-minus-wild standard error of 0.5 deg on {k}: {int(np.ceil(2 * (sd[k] / 0.5) ** 2))}")

import world5 as W
W.set_geometry(W.UNEQUAL)
nat = W.World(np.random.default_rng(5000), noise=0.01)
for s in (0.5, 0.9):
    w = W.reweight(nat.alpha, s)
    print(f"Weight on acceptances in the natural unequal world at s = {s}: wide {w[0]:.1f}, narrow {w[1]:.1f}")

print("\nErratum (pilot 6): the power estimate above includes the broken s = 0.9 arm. Without it:")
sub2 = sub[sub.cond != "s09"]
sds2 = sub2.groupby(["kind", "cond", "seed"])[["shift_wide", "shift_narrow", "g", "l", "unsigned"]].std()
print("  pooled within-cell sd per agent: " + ", ".join(f"{k} {v:.2f}" for k, v in np.sqrt((sds2 ** 2).mean()).items()))
