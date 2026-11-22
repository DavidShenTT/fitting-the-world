"""Check every reported number in paper/preprint.md against the saved results (PAPER_BRIEF check-in 2).

Each check recomputes a value from a results file, formats it as printed, and confirms that exact text appears in the
preprint (and, where marked, in summary.md). Exit code 1 if any check fails.
    python paper/verify_numbers.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = lambda *p: os.path.join(ROOT, "results", *p)
TEXT = open(os.path.join(HERE, "preprint.md"), encoding="utf-8").read()
SUMMARY = open(os.path.join(HERE, "summary.md"), encoding="utf-8").read()
FAIL, N = [], 0


def f2(v): return f"{v:.2f}"
def f3(v): return f"{v:.3f}"
def pct(v): return f"{100 * v:.0f}%"
def neg(s): return s.replace("-", "−")   # the text uses the Unicode minus sign


def check(desc, printed, computed, where="preprint"):
    """printed: the exact string in the text; computed: the same string built from the results."""
    global N
    N += 1
    doc = TEXT if where == "preprint" else SUMMARY
    ok_val = printed == computed
    ok_txt = printed in doc
    if not (ok_val and ok_txt):
        FAIL.append(f"{desc}: printed '{printed}', computed '{computed}', in {where}: {ok_txt}")


def row(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).to_numpy()
    assert m.sum() == 1, (kw, m.sum())
    return df[m].iloc[0]


# ---------------- E1
e1 = pd.read_csv(R("e1", "summary.csv")).set_index("M")
check("L1 differences", "+0.43, +0.81, +0.93 and +0.96",
      ", ".join("+" + f2(e1.loc[m, "L1_diff"]) for m in (2, 4, 6)) + " and +" + f2(e1.loc[12, "L1_diff"]))
check("kappa 0.25 read-axis range", "0.37–1.02", f"{f2(e1.u1_move_eye025.min())}–{f2(e1.u1_move_eye025.max())}")
check("unread axis", "(0.051)", f"({f3(e1.u2_move_eye025.mean())})")

# ---------------- E2
s2 = pd.read_csv(R("e2", "summary_E2.csv"))
pri, nos, sen = s2.iloc[0], s2.iloc[1], s2.iloc[2]
check("E2 slope M", f"0.91 [0.87, 0.94] on M", f"{f2(pri.slope_M)} [{f2(pri.M_lo)}, {f2(pri.M_hi)}] on M")
check("E2 slope 1/delta", "0.91 [0.88, 0.95] on 1/δ", f"{f2(pri.slope_inv_delta)} [{f2(pri.d_lo)}, {f2(pri.d_hi)}] on 1/δ")
check("E2 slope s_u", "0.98 on s_u", f"{f2(pri.slope_s_u)} on s_u")
check("E2 sensitivity", "the slopes are 0.96 and 0.96", f"the slopes are {f2(sen.slope_M)} and {f2(sen.slope_inv_delta)}")
check("E2 without s_u", "(0.19 and 0.22)", f"({f2(nos.slope_M)} and {f2(nos.slope_inv_delta)})")
check("E2 width slope", neg("slope 0.001 [-0.015, 0.014]"), neg(f"slope {f3(pri.slope_width)} [{f3(pri.w_lo)}, {f3(pri.w_hi)}]"))
rg = json.load(open(R("tuning", "e2", "range.json")))
c = pd.DataFrame(rg["cells"], columns=["M", "s_u", "delta"])
check("E2 eligible cells", "kept the 29 combinations", f"kept the {len(c)} combinations")
cm = np.corrcoef(np.log(c.M), np.log(c.s_u))[0, 1]; cd = np.corrcoef(np.log(1 / c.delta), np.log(c.s_u))[0, 1]
check("E2 selection correlations", neg("each correlate -0.44 with log s_u"), neg(f"each correlate {f2(cm)} with log s_u") if f2(cm) == f2(cd) else "mismatch")
pts = pd.read_csv(R("e2", "points.csv")); p32 = pts[pts.width == 32].copy(); p32["x"] = (p32.M * p32.s_u / p32.delta).round()
cells = p32.groupby(["M", "s_u", "delta", "x"]).built_count_overlap.mean().reset_index()
for x, n in ((600, 5), (800, 3)):
    g = cells[cells.x == x].built_count_overlap
    assert len(g) == n, (x, len(g))
check("E2 same product 600", "47–49 items at 600 (five cells", f"{cells[cells.x == 600].built_count_overlap.min():.0f}–{cells[cells.x == 600].built_count_overlap.max():.0f} items at 600 (five cells")
check("E2 same product 800", "58–60 at 800 (three cells)", f"{cells[cells.x == 800].built_count_overlap.min():.0f}–{cells[cells.x == 800].built_count_overlap.max():.0f} at 800 (three cells)")
fn = pd.read_csv(os.path.join(HERE, "figures", "figure_numbers.csv"))
fv = lambda q: float(fn[fn.quantity.str.strip() == q].value.iloc[0])
check("Fig 2 slope", "slope 0.93 [0.90, 0.97]",
      f"slope {f2(fv('slope of log count on log(M*s_u/delta), width-32 world points'))} [{f2(fv('95% CI low (bootstrap over worlds)'))}, {f2(fv('95% CI high (bootstrap over worlds)'))}]")

# ---------------- E5
dec = pd.read_csv(R("e4", "decay_sanity.csv"))
check("E5 half-times", "1,335–1,450 steps", f"{dec.half_time.min():,.0f}–{dec.half_time.max():,.0f} steps")
check("E5 expected", "ln 2/δ = 1,386", f"ln 2/δ = {np.log(2) / 0.0005:,.0f}")
e45 = pd.read_csv(R("e4", "summary_E4_E5.csv")).set_index("rule")
check("L6", neg("advantage in agreement is -0.002"), neg(f"advantage in agreement is {f3(e45.L6_adv.mean())}") if f3(e45.loc['wear', 'L6_adv']) == f3(e45.loc['eye1', 'L6_adv']) else "mismatch")

# ---------------- E3
e3 = pd.read_csv(R("e3", "summary_E3.csv"))
h = row(e3, rule="eye025", prediction="L3c (primary): entropy drop, bits")
check("E3 entropy drops", "drops by 0.63 bits for the builders' own body, 0.52 for other bodies of the same type, and 0.045",
      f"drops by {f2(h.same_body)} bits for the builders' own body, {f2(h.same_type)} for other bodies of the same type, and {f3(h.outsider)}")
check("E3 key-specific shares", "So 93% of the builders' drop, and 91% of the same type's",
      f"So {pct((h.same_body - h.outsider) / h.same_body)} of the builders' drop, and {pct((h.same_type - h.outsider) / h.same_type)} of the same type's")
cd_ = row(e3, rule="eye025", prediction="L3d: code-length saving, bits")
check("E3 code savings", "falls by 531 bits for the builders' body, 497 for the same type and 416 for outsiders",
      f"falls by {cd_.same_body:.0f} bits for the builders' body, {cd_.same_type:.0f} for the same type and {cd_.outsider:.0f} for outsiders")
check("E3 code key share", "Only 22% of the builders' saving", f"Only {pct((cd_.same_body - cd_.outsider) / cd_.same_body)} of the builders' saving")
check("L3d", "(1.27 [1.20, 1.34]; L3d)", f"({f2(cd_.ratio_same_over_out)} [{f2(cd_.ci_lo)}, {f2(cd_.ci_hi)}]; L3d)")
a = row(e3, rule="eye025", prediction="L3: early learning-curve area gain")
check("L3", "1.56 [1.39, 1.73] times", f"{f2(a.ratio_same_over_out)} [{f2(a.ci_lo)}, {f2(a.ci_hi)}] times")
b = row(e3, rule="eye025", prediction="L3b: home log-loss drop at 2,000 steps")
check("L3b", "1.73 [1.31, 2.25] times", f"{f2(b.ratio_same_over_out)} [{f2(b.ci_lo)}, {f2(b.ci_hi)}] times")
w = row(e3, rule="wear", prediction="L3c (primary): entropy drop, bits")
check("E3 wear control", "(entropy drops 0.27, 0.25 and 0.24 bits)", f"(entropy drops {f2(w.same_body)}, {f2(w.same_type)} and {f2(w.outsider)} bits)")
w3 = pd.read_csv(R("e3", "worlds.csv"))
check("E3 built share", "(the built share is 0.60)", f"(the built share is {f2(w3[w3.rule == 'eye025'].built_share.mean())})")

# ---------------- E8 (figure_numbers holds the width-4 start/end values)
def se(q):
    return [float(x) for x in fn[fn.quantity == q].value.iloc[0].split("/")]
fw = se("e8_fixed treat_wear edge accuracy start/end of T2 (width 4)"); fe = se("e8_fixed treat_eye1 edge accuracy start/end of T2 (width 4)")
fnat = se("e8_fixed natural edge accuracy start/end of T2 (width 4)")
check("E8 fully trained start", "(edge accuracy 0.868)", f"(edge accuracy {f3(fw[0])})")
check("E8 fully trained ends", "ended at 0.885 (wear) and 0.878 (shaping by eye), against 0.897",
      f"ended at {f3(fw[1])} (wear) and {f3(fe[1])} (shaping by eye), against {f3(fnat[1])}")
s8f = pd.read_csv(R("e8_fixed", "summary_E8.csv"))
ef_w, ef_e = row(s8f, rule="wear", width=4), row(s8f, rule="eye1", width=4)
check("D3 extra falls", "0.012 [0.003, 0.019] and 0.019 [0.007, 0.031]",
      f"{f3(ef_w.extra_fall)} [{f3(ef_w.extra_lo)}, {f3(ef_w.extra_hi)}] and {f3(ef_e.extra_fall)} [{f3(ef_e.extra_lo)}, {f3(ef_e.extra_hi)}]")
pw = se("e8 treat_wear edge accuracy start/end of T2 (width 4)"); pe = se("e8 treat_eye1 edge accuracy start/end of T2 (width 4)")
pn = se("e8 natural edge accuracy start/end of T2 (width 4)")
check("E8 partly trained start", "at edge accuracy 0.704", f"at edge accuracy {f3(pw[0])}")
check("E8 partly trained ends", "rose only to 0.762 (wear) and 0.742 (shaping by eye), while the same agents in the natural world rose to 0.856",
      f"rose only to {f3(pw[1])} (wear) and {f3(pe[1])} (shaping by eye), while the same agents in the natural world rose to {f3(pn[1])}")
s8 = pd.read_csv(R("e8", "summary_E8.csv"))
check("L8a primary eye", "(0.114 ≥ 0.10)", f"({f3(row(s8, rule='eye1', width=4).extra_fall)} ≥ 0.10)")
check("L8a primary, appendix", "0.094 / 0.114", f"{f3(row(s8, rule='wear', width=4).extra_fall)} / {f3(row(s8, rule='eye1', width=4).extra_fall)}")
check("L8b", "(ratios 1.11 and 1.30 against 2)", f"(ratios {f2(row(s8, rule='wear', width=4).L8b_ratio)} and {f2(row(s8, rule='eye1', width=4).L8b_ratio)} against 2)")
un_w = fv("untrained newcomers raised in wear world, edge accuracy (width 4, 60k)")
un_e = fv("untrained newcomers raised in eye1 world, edge accuracy (width 4, 60k)")
wl = float(fn[fn.quantity.str.startswith("untrained agents raised wild")].value.iloc[0])
check("E8 untrained", "reach edge accuracy 0.761 (wear) and 0.699 (shaping by eye), against 0.887",
      f"reach edge accuracy {f3(un_w)} (wear) and {f3(un_e)} (shaping by eye), against {f3(wl)}")

# ---------------- E9
e9 = pd.read_csv(R("e10", "summary_E9.csv"))
sw, se_ = row(e9, cond="skip", rule="wear"), row(e9, cond="skip", rule="eye1")
check("L9 gaps", "earn 42% (wear) and 73% (shaping by eye) less", f"earn {pct(sw.rel_gap)} (wear) and {pct(se_.rel_gap)} (shaping by eye) less")
check("E9 payoffs", "(0.204 and 0.097 against 0.353; L9, pass)", f"({f3(sw.pay_built)} and {f3(se_.pay_built)} against {f3(sw.pay_wild)}; L9, pass)")
pw9, pe9 = row(e9, cond="passive", rule="wear"), row(e9, cond="passive", rule="eye1")
check("E9 passive", "the gap is 31% and 53%", f"the gap is {pct(pw9.rel_gap)} and {pct(pe9.rel_gap)}")
check("L9b", "0.056 and 0.095 nats", f"{f3(sw.d_ll_diff)} and {f3(se_.d_ll_diff)} nats")
check("abstract 42-73%", "earn 42–73% less", f"earn {pct(sw.rel_gap)[:-1]}–{pct(se_.rel_gap)} less")
check("summary 42-73%", "they earned 42–73% less", f"they earned {pct(sw.rel_gap)[:-1]}–{pct(se_.rel_gap)} less", where="summary")

# ---------------- E10
L = pd.read_csv(R("e10", "summary_L10a.csv"))
for cond, rule, nm in (("skip", "wear", "wear"), ("skip", "eye1", "shaping by eye"), ("passive", "wear", "wear"), ("passive", "eye1", "shaping by eye")):
    r = row(L, cond=cond, rule=rule)
    lab = "Skip rule" if cond == "skip" else "Passive"
    check(f"Table 2 {cond} {rule}", f"| {lab} | {nm} | {f2(r.conf_ratio)} [{f2(r.conf_lo)}, {f2(r.conf_hi)}] | {f2(r.edge_ll_ratio)} [{f2(r.ll_lo)}, {f2(r.ll_hi)}] |",
          f"| {lab} | {nm} | {f2(r.conf_ratio)} [{f2(r.conf_lo)}, {f2(r.conf_hi)}] | {f2(r.edge_ll_ratio)} [{f2(r.ll_lo)}, {f2(r.ll_hi)}] |")
    assert r.verdict == "pass"
rw, re_ = row(L, cond="skip", rule="wear"), row(L, cond="skip", rule="eye1")
check("home log loss skip", "0.016 (wear) and 0.014 (shaping by eye) against 0.031",
      f"{f3(rw.home_ll_built)} (wear) and {f3(re_.home_ll_built)} (shaping by eye) against {f3(rw.home_ll_wild)}")
cells_ = pd.read_csv(R("e10", "summary_L10a_cells.csv"))
check("L10a cells", "holds in 34 of 40 width × budget cells", f"holds in {(cells_.verdict == 'pass').sum()} of {len(cells_)} width × budget cells")
assert set(cells_[cells_.verdict != "pass"].budget) == {30000}
U = pd.read_csv(R("e10", "summary_update_control.csv")).set_index("rule")
check("update control", "(ratios 2.53 and 1.89 for wear, 3.88 and 2.52 for shaping by eye)",
      f"(ratios {f2(U.loc['wear', 'conf_ratio'])} and {f2(U.loc['wear', 'edge_ll_ratio'])} for wear, {f2(U.loc['eye1', 'conf_ratio'])} and {f2(U.loc['eye1', 'edge_ll_ratio'])} for shaping by eye)")
EN = pd.read_csv(R("e10", "summary_L10a_ens.csv"))
check("ensembles conf range", "(confident-error ratios 2.26–4.18)", f"(confident-error ratios {f2(EN.conf_ratio.min())}–{f2(EN.conf_ratio.max())})")
assert (EN.verdict == "pass").all()
E = pd.read_csv(R("e10", "ens.csv")); p = E[E.variant == "prior"]
rm = p.groupby("kind").spread_edges.mean() / p.groupby("kind").spread_centres.mean()
check("prior spread", "(edge-to-centre spread 3.7–6.2)", f"(edge-to-centre spread {rm.min():.1f}–{rm.max():.1f})")
SH = pd.read_csv(R("e10", "summary_L10shift.csv"))
a1, a2 = row(SH, rule="wear"), row(SH, rule="eye1")
check("L10-shift", "3.69° [3.26, 4.14] (wear) and 5.17° [4.59, 5.72]",
      f"{f2(a1.shift_diff)}° [{f2(a1.lo)}, {f2(a1.hi)}] (wear) and {f2(a2.shift_diff)}° [{f2(a2.lo)}, {f2(a2.hi)}]")
p1, p2 = row(SH, rule="wear (passive, no criterion)"), row(SH, rule="eye1 (passive, no criterion)")
check("shift passive", "(1.41° and 0.88°)", f"({f2(p1.shift_diff)}° and {f2(p2.shift_diff)}°)")
DO = pd.read_csv(R("e10", "summary_L10dose.csv")).sort_values("removed")
check("dose removed", "removing 7%, 22% and 28% of edge items", "removing " + ", ".join(pct(v) for v in DO.removed[:2]) + f" and {pct(DO.removed.iloc[2])} of edge items")
check("dose shift", "the extra shift is 0.18°, 1.53° and 5.17°", f"the extra shift is {f2(DO.shift_diff.iloc[0])}°, {f2(DO.shift_diff.iloc[1])}° and {f2(DO.shift_diff.iloc[2])}°")
check("dose ratio", "the confident-error ratio 1.03, 1.63 and 4.02", f"the confident-error ratio {f2(DO.conf_ratio.iloc[0])}, {f2(DO.conf_ratio.iloc[1])} and {f2(DO.conf_ratio.iloc[2])}")
LM = pd.read_csv(R("e10", "summary_L10m.csv"))
mw = LM[LM.rule == "wear"].sort_values("eps_min"); me = LM[LM.rule == "eye1"].sort_values("eps_min")
check("L10m wear", "from 3.98° to 1.83° to 0.16° under wear", "from " + "° to ".join(f2(v) for v in mw.stretch_diff) + "° under wear")
check("L10m eye", "from 5.44° to 2.89° to 0.93° under shaping by eye", "from " + "° to ".join(f2(v) for v in me.stretch_diff) + "° under shaping by eye")
check("near-side refusals wild", "55% of their refusals", f"{pct(mw.wide_near_share_wild.iloc[0])} of their refusals")
check("near-side refusals built", "24% (wear) and 11% (shaping by eye) of newcomers' refusals",
      f"{pct(mw.wide_near_share_built.iloc[0])} (wear) and {pct(me.wide_near_share_built.iloc[0])} (shaping by eye) of newcomers' refusals")

# ---------------- E4 / L5
check("same-world agreement", "(Cohen's κ 0.99–1.00)", f"(Cohen's κ {f2(e45.k_ov_same.min())}–{f2(e45.k_ov_same.max())})")
check("natural agreement", "already agree at κ = 0.88", f"already agree at κ = {f2(e45.k_ov_nat.mean())}")
check("L5", "(L5, pass: +0.34 and +0.76)", f"(L5, pass: +{f2(e45.loc['wear', 'L5_diff'])} and +{f2(e45.loc['eye1', 'L5_diff'])})")
check("L4 appendix", "+0.115, +0.120 (ceiling 0.123)",
      f"+{f3(e45.loc['wear', 'same_minus_nat'])}, +{f3(e45.loc['eye1', 'same_minus_nat'])} (ceiling {f3(1 - e45.k_ov_nat.mean())})")

# ---------------- D1 (legibility)
dg = pd.read_csv(R("tuning", "leg_diag", "diag.csv")); dg = dg[(dg.lam == 0.4) & (dg.steps == 60000)]
means = [dg[dg.geom == g][c].mean() for g in ("symmetric", "unequal") for c in ("wide_edge_acc", "narrow_edge_acc")]
check("D1 edge accuracy range", "mean edge accuracy 0.84–0.90", f"mean edge accuracy {f2(min(means))}–{f2(max(means))}")

print(f"{N} checks; {len(FAIL)} failed")
for f in FAIL:
    print("FAIL", f)
sys.exit(1 if FAIL else 0)
