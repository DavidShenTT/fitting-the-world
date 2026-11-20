"""Print every figure used in PILOT2.md straight from the saved CSVs."""
import numpy as np, pandas as pd
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
R = lambda f: pd.read_csv(f"results/{f}.csv")
q, e3, fo, ra, e9, e8, en = R("q"), R("e3"), R("forestry"), R("raised"), R("e9"), R("e8"), R("ens")
print("== Q: built and edge shares"); print(q[q["item"] != "seconds"].pivot_table(index=["item", "world"], columns="seed", values="value").round(3).to_string())
print("seconds per seed:", q[q["item"] == "seconds"].value.round(0).tolist())
s = ra[ra.eps_r == 0]
print("\n== Q1 (natural decisive edge turns): heuristic ratio and wild width-32 agents' ratio")
w32 = s[(s.kind == "wild") & (s.width == 32)]
print("heuristic ratio:", w32.heuristic_ratio.round(3).tolist(), " wild-32 decisive ratio:", w32.decisive_ratio.round(3).tolist())
print("\n== Q2/E3: learning cost to competence 0.75 (steps; NaN = not reached by 30k) and log loss at 2k")
print(e3.pivot_table(index=["world", "body"], columns="seed", values="cost").to_string())
print(e3.pivot_table(index=["world", "body"], values="logloss_2k", aggfunc="mean").round(3).to_string())
c = e3.pivot_table(index=["seed", "body"], columns="world", values="cost")
for wn in ("wear", "eye1", "eye025"):
    c["speedup_" + wn] = c["natural"] / c[wn]
print(c.round(2).to_string())
sp = c.groupby(level="body")[[x for x in c.columns if x.startswith("speedup")]].mean()
print("mean speed-up by body:"); print(sp.round(2).to_string())
l = e3.pivot_table(index=["seed", "body"], columns="world", values="logloss_2k")
for wn in ("wear", "eye1", "eye025"):
    l["drop_" + wn] = l["natural"] - l[wn]
print("log-loss drop at 2k by body (mean):"); print(l.groupby(level="body")[[x for x in l.columns if x.startswith("drop")]].mean().round(3).to_string())
print("\n== L3c forestry drop (bits), mean over seeds"); print(fo.pivot_table(index="world", columns="body", values="drop").round(3).to_string())
print(fo.pivot_table(index=["world", "seed"], columns="body", values="drop").round(3).to_string())
print("\n== E10 single networks (eps_r = 0), means over seeds")
g = s.groupby(["kind", "width", "budget"])[["home_logloss", "competence_home", "edge_acc", "calib_gap", "decisive_ratio"]].mean()
print(g.round(3).to_string())
wild = s[s.kind == "wild"].set_index(["seed", "width", "budget"])
for k in ("wear", "eye1", "eye025"):
    b = s[s.kind == k].set_index(["seed", "width", "budget"])
    d = (b.calib_gap - wild.calib_gap); e = (wild.edge_acc - b.edge_acc)
    print(f"L10a {k}: calib-gap difference (built - wild) mean {d.mean():.3f}, cells with mean >= 0.10: "
          f"{(d.groupby(level=['width','budget']).mean() >= 0.10).sum()} of 6; edge-acc deficit mean {e.mean():.3f}")
print("\n== Ensembles (width 32, 30k)"); print(en.groupby("kind")[["edge_acc", "calib_gap", "decisive_ratio"]].mean().round(3).to_string())
sg = s[(s.width == 32) & (s.budget == 30000)].groupby("kind")[["edge_acc", "calib_gap"]].mean()
print("single networks, same cells:"); print(sg.round(3).to_string())
for k in ("wear", "eye1"):
    print(f"L10a-ens {k}: ensemble calib-gap difference (built - wild) {en[en.kind==k].calib_gap.mean() - en[en.kind=='wild'].calib_gap.mean():.3f}")
print("\n== Q3 / L10b: home log loss by eps_r (width 32, 30k)")
m = ra[(ra.width == 32) & (ra.budget == 30000)].groupby(["kind", "eps_r"])[["home_logloss", "edge_acc", "calib_gap"]].mean()
print(m.round(3).to_string())
print("\n== E9"); print(e9.groupby(["kind", "width"])[["decisive_pay_0", "decisive_pay_500", "decisive_pay_mean", "home_logloss", "snap_logloss", "d_nll"]].mean().round(3).to_string())
ew = e9[e9.kind == "wild"].set_index(["seed", "width"])
for k in ("wear", "eye1", "eye025"):
    b = e9[e9.kind == k].set_index(["seed", "width"])
    print(f"L9 {k}: decisive-pay gap (wild - built) mean {(ew.decisive_pay_mean - b.decisive_pay_mean).mean():.3f} "
          f"[min {(ew.decisive_pay_mean - b.decisive_pay_mean).min():.3f}, max {(ew.decisive_pay_mean - b.decisive_pay_mean).max():.3f}]; "
          f"L9b d_nll gap (built - wild) mean {(b.d_nll - ew.d_nll).mean():.3f}")
print("\n== E8"); print(e8.groupby(["width", "arm"])[["plateau_at", "edge_before", "edge_after", "fall", "b_cost", "b_competence_end"]].mean().round(3).to_string())
f = e8.pivot_table(index=["seed", "width"], columns="arm", values="fall")
for k in ("treat_wear", "treat_eye025"):
    f["extra_" + k] = f[k] - f["natural"]; f["vs_filtered_" + k] = f[k] - f["filtered"]
print(f.groupby(level="width")[[x for x in f.columns if "extra" in x or "vs_" in x]].mean().round(3).to_string())
ex = f.groupby(level="width")["extra_treat_wear"].mean(); print("L8b wear narrowest/widest: %.2f" % (ex.loc[8] / ex.loc[128]) if ex.loc[128] != 0 else "L8b: widest extra fall is 0")
bc = e8.pivot_table(index=["seed", "width"], columns="arm", values="b_cost"); print("kind-B cost to 0.75 (steps):"); print(bc.to_string())
