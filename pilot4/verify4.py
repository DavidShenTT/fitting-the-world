"""Print every figure used in PILOT4.md straight from the saved CSVs."""
import numpy as np
import pandas as pd

pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
R = lambda f: pd.read_csv(f"results/{f}.csv")
q, e3, ra, en, e9, e8, up = R("q"), R("e3"), R("raised"), R("ens"), R("e9"), R("e8"), R("upd")

print("== Worlds"); print(q[q.world != "seconds"].round(3).to_string(index=False))
print("seconds per seed:", q[q.world == "seconds"].built_share.round(0).tolist())

print("\n== E3 (means over seeds and both builder types; outsiders and same-type maps pooled)")
e3["speedup"] = e3.cost_natural / e3.cost_built
e3["ll_drop"] = e3.ll2k_natural - e3.ll2k_built
e3["h_drop"] = e3.h_natural - e3.h_built
e3["code_drop_bits"] = e3.code_natural - e3.code_built
g = e3.groupby(["world", "arm"])[["cost_natural", "cost_built", "speedup", "ll_drop", "h_drop", "code_natural", "code_built", "code_drop_bits"]].mean()
print(g.round(3).to_string())
for wn in ("eye025", "wear"):
    s = g.loc[wn]
    print(f"{wn}: speed-up ratio same/outsider {s.loc['same_body','speedup']/s.loc['outsider','speedup']:.2f}; "
          f"ll-drop ratio {s.loc['same_body','ll_drop']/s.loc['outsider','ll_drop']:.2f}; "
          f"entropy drop same {s.loc['same_body','h_drop']:.3f}, same type {s.loc['same_type','h_drop']:.3f}, outsider {s.loc['outsider','h_drop']:.3f}; "
          f"code drop (bits) same {s.loc['same_body','code_drop_bits']:.1f}, same type {s.loc['same_type','code_drop_bits']:.1f}, outsider {s.loc['outsider','code_drop_bits']:.1f}")
print("built cost per map arm, eye025:"); print(e3[e3.world == "eye025"].groupby(["builder_body", "arm"])[["cost_built", "speedup", "h_drop", "code_drop_bits"]].mean().round(2).to_string())

s0 = ra[ra.eps_r == 0]
print("\n== Wild agents' starting bias (60k, width 32): learned half-widths and edge accuracy")
print(s0[(s0.kind == "wild") & (s0.width == 32) & (s0.budget == 60000)][["seed", "hw_wide", "hw_narrow", "wide_edge_acc", "narrow_edge_acc"]].round(2).to_string(index=False))

print("\n== E10 single networks (eps_r = 0), means over seeds")
cols = ["acts", "home_logloss", "shift_wide", "shift_narrow", "wide_under_conf_accepted", "narrow_false_conf_refused", "wide_edge_acc", "narrow_edge_acc", "decisive_ratio"]
print(s0.groupby(["kind", "width", "budget"])[cols].mean().round(3).to_string())
wild = s0[s0.kind == "wild"].set_index(["seed", "width", "budget"])
for k in ("wear", "eye1"):
    b = s0[s0.kind == k].set_index(["seed", "width", "budget"])
    dw = b.shift_wide - wild.shift_wide; dn = b.shift_narrow - wild.shift_narrow
    cw = dw.groupby(level=["width", "budget"]).mean(); cn = dn.groupby(level=["width", "budget"]).mean()
    print(f"L10a {k}: built-minus-wild shift toward 45, wide mean {dw.mean():.2f} (cells >= 3: {(cw >= 3).sum()} of 6), "
          f"narrow mean {dn.mean():.2f} (cells >= 3: {(cn >= 3).sum()} of 6); home log loss built {b.home_logloss.mean():.3f} vs wild {wild.home_logloss.mean():.3f}")

print("\n== Ensembles (60k)")
print(en.groupby(["variant", "kind"])[["shift_wide", "shift_narrow", "home_logloss", "spread_edges", "spread_centres", "decisive_ratio"]].mean().round(3).to_string())
for v in ("standard", "diverse", "prior"):
    e = en[en.variant == v].set_index(["seed", "kind"])
    for k in ("wear", "eye1"):
        print(f"L10a-ens {v} {k}: wide {(e.shift_wide.xs(k, level='kind') - e.shift_wide.xs('wild', level='kind')).mean():.2f}, "
              f"narrow {(e.shift_narrow.xs(k, level='kind') - e.shift_narrow.xs('wild', level='kind')).mean():.2f}")

print("\n== Random refusal (width 32, 60k)")
m = ra[(ra.width == 32) & (ra.budget == 60000)].groupby(["kind", "eps_r"])[["home_logloss", "shift_wide", "shift_narrow"]].mean()
print(m.round(3).to_string())

print("\n== Update-count control"); print(up[["seed", "target_acts", "wild_acts", "wild_steps", "wild60k_acts", "shift_wide", "shift_narrow"]].round(2).to_string(index=False))

print("\n== E9")
print(e9.groupby(["kind", "width"])[["decisive_pay_mean", "wide_pay_mean", "narrow_pay_mean", "wide_act_rate_0", "narrow_refusals_per_act_0", "d_nll"]].mean().round(3).to_string())
ew = e9[e9.kind == "wild"].set_index(["seed", "width"])
for k in ("wear", "eye1"):
    b = e9[e9.kind == k].set_index(["seed", "width"])
    rel = 1 - b.decisive_pay_mean.mean() / ew.decisive_pay_mean.mean()
    print(f"L9 {k}: decisive payoff built {b.decisive_pay_mean.mean():.3f} vs wild {ew.decisive_pay_mean.mean():.3f} -> {100*rel:.0f}% lower; "
          f"wide act rate {b.wide_act_rate_0.mean():.3f} vs {ew.wide_act_rate_0.mean():.3f}; "
          f"narrow refusals/act {b.narrow_refusals_per_act_0.mean():.3f} vs {ew.narrow_refusals_per_act_0.mean():.3f}; "
          f"L9b d_nll gap {(b.d_nll - ew.d_nll).mean():.3f}")

print("\n== E8")
e8["fall"] = e8.edge_acc_before - e8.edge_acc_after
print(e8.groupby(["width", "arm"])[["plateau_at", "acts_T2", "edge_seen", "edge_acc_before", "edge_acc_after", "fall", "shift_wide_after", "shift_narrow_after"]].mean().round(3).to_string())
f = e8.pivot_table(index=["seed", "width"], columns="arm", values="fall")
for k in ("treat_wear", "treat_eye1"):
    print(f"{k}: extra fall vs natural by width {(f[k] - f['natural']).groupby(level='width').mean().round(3).to_dict()}; "
          f"vs filtered {(f[k] - f['filtered']).groupby(level='width').mean().round(3).to_dict()}")
a = e8.pivot_table(index=["seed", "width"], columns="arm", values="acts_T2")
print("S7 acts treat_wear/filtered %.3f; edge seen filtered %.3f vs wear world %.3f" % (
    (a.treat_wear / a.filtered).mean(), e8[e8.arm == "filtered"].edge_seen.mean(), e8[e8.arm == "treat_wear"].edge_seen.mean()))

print("\n== Derived readouts (from the same files)")
print("L9b d_nll gap, built minus wild: " + ", ".join(
    f"{k} {(e9[e9.kind == k].set_index(['seed', 'width']).d_nll - ew.d_nll).mean():.4f}" for k in ("wear", "eye1")))
for k in ("treat_wear", "treat_eye1"):
    x = (f[k] - f["natural"]).groupby(level="width").mean()
    print(f"L8b {k}: extra fall width 8 / width 128 = {x.loc[8]:.3f} / {x.loc[128]:.3f} = {x.loc[8] / x.loc[128]:.2f}x")
w60 = s0[(s0.width == 32) & (s0.budget == 60000)].set_index(["seed", "kind"])
u = up.set_index("seed")
for k in ("wear", "eye1"):
    b = w60.xs(k, level="kind")
    print(f"Update-count control, {k} (width 32, 60k) minus wild at matched acted turns: "
          f"wide {(b.shift_wide - u.shift_wide).mean():.2f}, narrow {(b.shift_narrow - u.shift_narrow).mean():.2f}; "
          f"minus wild at 60k steps: wide {(b.shift_wide - w60.xs('wild', level='kind').shift_wide).mean():.2f}, "
          f"narrow {(b.shift_narrow - w60.xs('wild', level='kind').shift_narrow).mean():.2f}")
r = ra[(ra.width == 32) & (ra.budget == 60000)]
wl = r[r.kind == "wild"].set_index("seed")
for k in ("wear", "eye1"):
    b = r[(r.kind == k) & (r.eps_r == 0.01)].set_index("seed")
    print(f"L10b {k} at eps_r 0.01: home log loss {b.home_logloss.mean():.3f} vs wild {wl.home_logloss.mean():.3f} "
          f"(ratio {b.home_logloss.mean() / wl.home_logloss.mean():.2f}); built minus wild wide {(b.shift_wide - wl.shift_wide).mean():.2f}, "
          f"narrow {(b.shift_narrow - wl.shift_narrow).mean():.2f}; agents per kind {len(b)} (binning needs >= 3 bins of 10)")
print("Exploratory, unsigned: mean |shift| over all E10 single networks (eps_r = 0), wide / narrow")
for k in ("wild", "wear", "eye1"):
    b = s0[s0.kind == k]
    print(f"  {k}: {b.shift_wide.abs().mean():.2f} / {b.shift_narrow.abs().mean():.2f}")
cal = s0.groupby("kind")[["wide_under_conf_accepted", "narrow_false_conf_refused"]].mean()
print("L10a calibration split (all E10 cells): underconfidence on accepted wide edges / false confidence on refused narrow edges")
for k in ("wild", "wear", "eye1"):
    print(f"  {k}: {cal.loc[k, 'wide_under_conf_accepted']:.3f} / {cal.loc[k, 'narrow_false_conf_refused']:.3f}")

import re
print("\n== D2b and D3, built minus wild shift toward 45 (wide, narrow), from results/diag_gap.txt and results/diag_labels.txt")
d2 = {}
for line in open("results/diag_gap.txt"):
    mm = re.match(r"\s+passive=(\w+)\s+(\w+)\s*: wide\s+(-?[\d.]+) \[[\d.]+\]\s+narrow\s+(-?[\d.]+)", line)
    if mm:
        d2[(mm.group(1), mm.group(2))] = (float(mm.group(3)), float(mm.group(4)))
for reg in ("False", "True"):
    for k in ("wear", "eye1"):
        b, w = d2[(reg, k)], d2[(reg, "wild")]
        print(f"  D2b {'passive' if reg == 'True' else 'skip   '} {k}: ({b[0] - w[0]:+.1f}, {b[1] - w[1]:+.1f})")
d3 = {}
for line in open("results/diag_labels.txt"):
    p = line.split()
    if p and p[0] in ("skip", "passive"):
        d3[(p[0], p[1])] = [float(x) for x in p[2:]]
for reg in ("skip", "passive"):
    b, w = d3[(reg, "wear")], d3[(reg, "wild")]
    print(f"  D3  {reg:7s} wear: ({b[-2] - w[-2]:+.1f}, {b[-1] - w[-1]:+.1f}); labels that were acceptances, wide/narrow: "
          f"wild {w[1]:.2f}/{w[2]:.2f}, wear {b[1]:.2f}/{b[2]:.2f}; labels inside the window per 1000, wide/narrow: "
          f"wild {w[3]:.0f}/{w[4]:.0f}, wear {b[3]:.0f}/{b[4]:.0f}")
