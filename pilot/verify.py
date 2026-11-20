"""Print every figure used in PILOT.md straight from the saved CSVs."""
import numpy as np, pandas as pd
pd.set_option("display.width", 200)
e1 = pd.read_csv("results/e1.csv"); r = pd.read_csv("results/raised.csv"); e8 = pd.read_csv("results/e8.csv"); e9 = pd.read_csv("results/e9.csv")
print("C2/E1 built share by condition (per seed):"); print(e1.pivot(index="seed", columns="condition", values="built_share").round(3).to_string())
print("C4 edge share, built (wear) vs natural:"); print(e1[e1.condition == "wear"][["seed", "edge_share", "natural_edge_share"]].round(3).to_string(index=False))
w32 = r[(r.kind == "wild") & (r.width == 32)]
print("C3 wild width-32 edge accuracy, min/max over seeds and budgets: %.3f / %.3f" % (w32.edge_acc.min(), w32.edge_acc.max()))
print("C5 wall-clock: whole pilot ~31 s on 2 cores (see run log); builder warm-up per world, s:", e1[e1.condition == "wear"].secs.round(2).tolist())
print("\nE10 means over 3 seeds:")
g = r.groupby(["width", "budget", "kind"])[["home_log_loss", "edge_acc", "calib_gap"]].mean().unstack("kind")
g[("calib_gap", "built_minus_wild")] = g[("calib_gap", "built")] - g[("calib_gap", "wild")]
g[("edge_acc", "wild_minus_built")] = g[("edge_acc", "wild")] - g[("edge_acc", "built")]
print(g.round(3).to_string())
d = r.pivot_table(index=["seed", "width", "budget"], columns="kind", values="calib_gap")
print("L10a per-seed calib-gap difference (built - wild): min %.3f, max %.3f, share >= 0.10: %.2f" % ((d.built - d.wild).min(), (d.built - d.wild).max(), ((d.built - d.wild) >= 0.10).mean()))
print("L10b home log loss ranges: built %.3f-%.3f, wild %.3f-%.3f" % (r[r.kind == "built"].home_log_loss.min(), r[r.kind == "built"].home_log_loss.max(), r[r.kind == "wild"].home_log_loss.min(), r[r.kind == "wild"].home_log_loss.max()))
print("\nE9 means:"); print(e9.groupby(["kind", "width"])[["first500_accept", "competence_at_snap", "steps_to_95"]].mean().round(3).to_string())
gap = e9.pivot_table(index=["seed", "width"], columns="kind", values="first500_accept"); print("E9 first-500 gap (wild - built): mean %.3f, min %.3f, max %.3f" % ((gap.wild - gap.built).mean(), (gap.wild - gap.built).min(), (gap.wild - gap.built).max()))
print("\nE8:"); f = e8.pivot_table(index=["seed", "width"], columns="arm", values="fall"); f["extra"] = f.treatment - f.control
print(f.groupby("width").agg(["mean", "min", "max"]).round(3).to_string())
ex = f.groupby("width").extra.mean(); print("L8b narrowest/widest extra fall: %.2f" % (ex.loc[8] / ex.loc[128]))
print("E8 edge accuracy before T2 (mean):", e8[e8.arm == "control"].groupby("width").edge_before.mean().round(3).to_dict())
