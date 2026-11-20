"""Print every figure used in PILOT3.md straight from the saved CSVs."""
import numpy as np
import pandas as pd

pd.set_option("display.width", 230); pd.set_option("display.max_columns", 40)
R = lambda f: pd.read_csv(f"results/{f}.csv")
q, e3, ra, en, e9, e8, up = R("q"), R("e3"), R("raised"), R("ens"), R("e9"), R("e8"), R("upd")

print("== Worlds"); print(q[q.world != "seconds"].round(3).to_string(index=False))
print("seconds per seed:", q[q.world == "seconds"].built_share.round(0).tolist())

print("\n== R3: natural learning cost by body type (steps to 0.9)")
nc = e3.drop_duplicates(["seed", "body"])[["seed", "body", "cost_natural"]]
nc["type"] = np.where(nc.body < 2, 1, 2)
print(nc.pivot_table(index="seed", columns="body", values="cost_natural").to_string())
mt = nc.groupby("type").cost_natural.mean(); print("mean by type:", mt.round(0).to_dict(), " ratio %.2f" % (mt.max() / mt.min()))

print("\n== E3 (means over seeds and both builder types)")
e3["speedup"] = e3.cost_natural / e3.cost_built
e3["ll_drop"] = e3.ll2k_natural - e3.ll2k_built
e3["h_drop"] = e3.h_natural - e3.h_built
e3["code_drop"] = e3.code_natural - e3.code_built
g = e3.groupby(["world", "arm"])[["cost_natural", "cost_built", "speedup", "ll_drop", "h_drop", "code_drop", "acts_built"]].mean()
print(g.round(3).to_string())
print("by builder type:"); print(e3.groupby(["world", "builder_body", "arm"])[["speedup", "ll_drop", "h_drop", "code_drop"]].mean().round(3).to_string())
for wn in ("eye025", "wear"):
    s = g.loc[wn]
    print(f"{wn}: speed-up ratio same/outsider {s.loc['same_body','speedup']/s.loc['outsider','speedup']:.2f}; "
          f"ll-drop ratio {s.loc['same_body','ll_drop']/s.loc['outsider','ll_drop']:.2f}; "
          f"entropy drop same {s.loc['same_body','h_drop']:.3f} vs outsider {s.loc['outsider','h_drop']:.3f}; "
          f"code-drop ratio {s.loc['same_body','code_drop']/s.loc['outsider','code_drop']:.2f}")

print("\n== R1 at the chosen setting: wild width-32, 60k (edge accuracy, decisive ratio)")
s0 = ra[ra.eps_r == 0]
print(s0[(s0.kind == "wild") & (s0.width == 32) & (s0.budget == 60000)][["seed", "edge_acc", "decisive_ratio"]].round(3).to_string(index=False))

print("\n== E10 single networks (eps_r = 0), means over seeds")
cols = ["acts", "home_logloss", "competence_home", "edge_acc", "edge_acc_23", "false_conf_refused", "under_conf_accepted", "calib_gap", "decisive_ratio"]
print(s0.groupby(["kind", "width", "budget"])[cols].mean().round(3).to_string())
wild = s0[s0.kind == "wild"].set_index(["seed", "width", "budget"])
for k in ("wear", "eye1"):
    b = s0[s0.kind == k].set_index(["seed", "width", "budget"])
    d = b.false_conf_refused - wild.false_conf_refused
    cell = d.groupby(level=["width", "budget"]).mean()
    print(f"L10a {k}: false-confidence difference (built - wild) mean {d.mean():.3f}; cells >= 0.10: {(cell >= 0.10).sum()} of {len(cell)}; "
          f"range of cell means {cell.min():.3f} to {cell.max():.3f}; edge-acc deficit mean {(wild.edge_acc - b.edge_acc).mean():.3f}; "
          f"accepted-edge underconfidence difference {(b.under_conf_accepted - wild.under_conf_accepted).mean():.3f}")

print("\n== Ensembles (60k)")
print(en.groupby(["variant", "kind"])[["acts", "home_logloss", "edge_acc", "false_conf_refused", "under_conf_accepted", "member_spread", "decisive_ratio"]].mean().round(3).to_string())
for v in ("standard", "diverse", "prior"):
    e = en[en.variant == v].set_index(["seed", "kind"]).false_conf_refused.unstack()
    print(f"L10a-ens {v}: false-confidence difference wear {(e.wear - e.wild).mean():.3f}, eye1 {(e.eye1 - e.wild).mean():.3f}")
sp = en.pivot_table(index="kind", columns="variant", values="member_spread")
print("R6 member spread (built-raised): prior vs standard"); print(sp.round(3).to_string())

print("\n== Q: random refusal (width 32, 60k)")
m = ra[(ra.width == 32) & (ra.budget == 60000)].groupby(["kind", "eps_r"])[["home_logloss", "edge_acc", "mean_conf", "false_conf_refused", "under_conf_accepted", "calib_gap"]].mean()
print(m.round(3).to_string())

print("\n== Update-count check"); print(up.round(3).to_string(index=False))
w32 = s0[(s0.kind == "wild") & (s0.width == 32) & (s0.budget == 60000)].set_index("seed")
print("edge acc: wild at 60k steps %.3f vs wild at matched acts %.3f; built (wear) %.3f" % (
    w32.edge_acc.mean(), up.edge_acc.mean(), s0[(s0.kind == "wear") & (s0.width == 32) & (s0.budget == 60000)].edge_acc.mean()))

print("\n== E9")
print(e9.groupby(["kind", "width"])[["decisive_pay_0", "decisive_pay_500", "decisive_pay_mean", "act_rate_0", "refusals_per_act_0", "home_logloss", "snap_logloss", "d_nll"]].mean().round(3).to_string())
ew = e9[e9.kind == "wild"].set_index(["seed", "width"])
for k in ("wear", "eye1"):
    b = e9[e9.kind == k].set_index(["seed", "width"])
    gap = ew.decisive_pay_mean - b.decisive_pay_mean
    print(f"L9 {k}: decisive-pay gap (wild - built) mean {gap.mean():.3f} [min {gap.min():.3f}, max {gap.max():.3f}]; "
          f"L9b d_nll gap {(b.d_nll - ew.d_nll).mean():.3f}; act rate built {b.act_rate_0.mean():.3f} vs wild {ew.act_rate_0.mean():.3f}; "
          f"refusals/act built {b.refusals_per_act_0.mean():.3f} vs wild {ew.refusals_per_act_0.mean():.3f}")

print("\n== E8")
print(e8.groupby(["width", "arm"])[["plateau_at", "edge_before", "edge_after", "acts_T2", "b_auc", "probe_b", "w1_norm_before", "w1_norm_after"]].mean().round(3).to_string())
e8["fall"] = e8.edge_before - e8.edge_after
f = e8.pivot_table(index=["seed", "width"], columns="arm", values="fall")
for k in ("treat_wear", "treat_eye1"):
    f["extra_" + k] = f[k] - f["natural"]; f["vs_filt_" + k] = f[k] - f["filtered"]
fm = f.groupby(level="width")[[c for c in f.columns if c.startswith(("extra", "vs_"))]].mean()
print(fm.round(3).to_string())
print("L8b wear narrowest/widest: %s" % (f"{fm.loc[8,'extra_treat_wear']/fm.loc[128,'extra_treat_wear']:.2f}" if fm.loc[128, "extra_treat_wear"] != 0 else "n/a"))
a = e8.pivot_table(index=["seed", "width"], columns="arm", values="acts_T2")
print("R5 acted turns, treat_wear / filtered (mean ratio): %.3f; treat_eye1 / filtered: %.3f" % ((a.treat_wear / a.filtered).mean(), (a.treat_eye1 / a.filtered).mean()))
au = e8.pivot_table(index=["seed", "width"], columns="arm", values="b_auc")
print("L8c at width 8: treat_wear AUC / natural AUC %.3f; treat_eye1 / natural %.3f; filtered / natural %.3f" % (
    (au.treat_wear / au.natural).xs(8, level="width").mean(), (au.treat_eye1 / au.natural).xs(8, level="width").mean(),
    (au.filtered / au.natural).xs(8, level="width").mean()))
