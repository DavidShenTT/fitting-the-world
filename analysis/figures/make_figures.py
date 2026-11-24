"""Figures for the preprint (PAPER_BRIEF §6), drawn only from saved results. Writes paper/figures/fig*.png/.svg and
paper/figures/figure_numbers.csv (every number a figure shows that isn't already in REPORT.md).

Styled for the Typst build (paper/typst/style.typ): Inter, drawn at final size (full-width figures 6.2 in, Figs 2 and 6
3.85 in, i.e. 62% of the 158 mm text block), the page's neutrals, SVG with text as paths plus 200-dpi PNG for the HTML.
Fetch the fonts once with `python tools/get_fonts.py`.
    python analysis/figures/make_figures.py
"""
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Wedge  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from common import ROOT, boot_ci  # noqa: E402


def R(*p):
    return os.path.join(ROOT, "results", *p)


OUT = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUT, exist_ok=True)

FONTS = os.path.join(ROOT, "paper", "typst", "fonts")
_otf = sorted(glob.glob(os.path.join(FONTS, "*.otf")))
if not _otf:
    sys.exit(f"no fonts in {FONTS}: run python tools/get_fonts.py")
for f in _otf:
    font_manager.fontManager.addfont(f)

# categorical slots, validated with the dataviz validator (light mode); entity -> colour, fixed across figures
C_WILD, C_WEAR, C_EYE = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1c2230", "#5d6676", "#dfe3ea"            # the page's text, secondary and hairline colours
BLUE_RAMP = ["#104281", "#256abf", "#5598e7", "#9ec5f4"]     # ordinal: darker = more
FULL, NARROW = 6.2, 3.85                                     # final widths in inches (158 mm text block; 62%)
plt.rcParams.update({
    "font.family": "Inter", "font.size": 7.5, "text.color": INK,
    "axes.labelsize": 7.5, "axes.labelcolor": INK, "axes.titlesize": 8, "axes.titleweight": "semibold",
    "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 7,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "legend.fontsize": 7, "legend.title_fontsize": 7, "legend.frameon": False,
    "lines.linewidth": 1.8, "errorbar.capsize": 0,
    "mathtext.fontset": "custom", "mathtext.rm": "Inter", "mathtext.it": "Inter:italic", "mathtext.bf": "Inter:bold",
    "mathtext.sf": "Inter", "mathtext.fallback": "cm",
    "svg.fonttype": "path", "figure.dpi": 100,
})
NUM = []


def note(fig, name, value, source):
    NUM.append(dict(figure=fig, quantity=name, value=value, source=source))


def save(fig, name):
    # no tight cropping: the saved figure is exactly its final size, so text prints at the sizes set above
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200)
    fig.savefig(os.path.join(OUT, name + ".svg"))
    plt.close(fig)


def mean_ci(x):
    lo, hi = boot_ci(x)
    return float(np.mean(x)), lo, hi


# ------------------------------------------------------------------ Fig 1: schematic
def fig1():
    fig = plt.figure(figsize=(FULL, 2.55))
    ax_u, ax_s = fig.add_axes([0.0, 0.15, 0.2, 0.66]), fig.add_axes([0.228, 0.15, 0.2, 0.66])
    G_IN, G_OUT = "#4b5363", "#aab2bf"   # neutral greys: arcs aren't experimental conditions
    for ax, half, title, exps in ((ax_u, (60, 35, 60, 35), "Unequal arcs", "E3, E8–E10"),
                                  (ax_s, (60, 60, 60, 60), "Symmetric arcs", "E1, E2, E4–E6")):
        ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1.55, 1.55); ax.set_ylim(-1.55, 1.55)
        for a, (c, h) in enumerate(zip((0, 90, 180, 270), half)):
            col = G_IN if a % 2 == 0 else G_OUT
            r = 1.0 if a % 2 == 0 else 1.18
            ax.add_patch(Wedge((0, 0), r, c - h, c + h, width=0.12, color=col, lw=0))
            ax.text(1.45 * np.cos(np.deg2rad(c)), 1.45 * np.sin(np.deg2rad(c)), f"{a}", ha="center", va="center",
                    color=MUTED, fontsize=7)
        ax.text(0, 0.1, title.split()[0], ha="center", va="center", fontsize=7.5, fontweight="semibold")
        ax.text(0, -0.12, "arcs", ha="center", va="center", fontsize=7.5, fontweight="semibold")
        ax.text(0, -1.8, exps, ha="center", va="center", fontsize=7, color=MUTED)
    TITLE_Y = 0.92   # one baseline for the A, B and C titles
    title = dict(fontweight="semibold", fontsize=8, va="baseline")
    fig.text(0.005, TITLE_Y, "A  Four actions, each accepted in its arc", **title)
    fig.text(0.005, 0.02, "dark arcs: actions 0, 2;  light arcs: actions 1, 3;  where arcs overlap, both are accepted",
             fontsize=7, color=MUTED, va="bottom")
    # B: natural vs built item distribution (symmetric world, wear; folded onto 0-90 degrees)
    ax = fig.add_axes([0.528, 0.38, 0.2, 0.46])
    h = pd.read_csv(R("e10", "sym_hist.csv")).groupby("bin_lo").share.mean()
    ax.bar(h.index + 2.5, h.values * 100, width=4.2, color=C_WEAR, label="built (wear)")
    ax.axhline(100 / 18, color=C_WILD, lw=1.8, label="natural (uniform)")
    ax.axvspan(30, 60, color=GRID, alpha=0.7, lw=0, zorder=0)
    ax.text(45, h.values.max() * 100 * 0.92, "overlap", ha="center", fontsize=7, color=MUTED)
    ax.set_xlabel("item angle, folded (deg)"); ax.set_ylabel("% of items per 5°"); ax.set_xticks([0, 30, 60, 90])
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.36), ncol=2, handlelength=1.4, columnspacing=1.0)
    fig.text(0.458, TITLE_Y, "B  Built vs natural (symmetric world)", **title)
    note(1, "built (wear, symmetric) % of items in [30,60) deg", float(h.loc[30:55].sum() * 100), "results/e10/sym_hist.csv")
    # C: the cycle
    ax = fig.add_axes([0.8, 0.1, 0.2, 0.74]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    boxes = [(0.5, 0.86, "builders fit the world\n(and must keep it up)"), (0.5, 0.52, "newcomers learn inside\nthe built world"),
             (0.5, 0.18, "upkeep stops:\nthe world snaps back")]
    for x, y, s in boxes:
        ax.text(x, y, s, ha="center", va="center", fontsize=7, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=MUTED, lw=0.6))
    for y0, y1 in ((0.75, 0.63), (0.41, 0.29)):
        ax.add_patch(FancyArrowPatch((0.5, y0), (0.5, y1), arrowstyle="-|>", mutation_scale=8, color=MUTED, lw=0.9))
    fig.text(0.805, TITLE_Y, "C  The cycle", **title)
    save(fig, "fig1_schematic")


# ------------------------------------------------------------------ Fig 2: upkeep law
def fig2():
    p = pd.read_csv(R("e2", "points.csv"))
    p32 = p[p.width == 32].copy()
    p32["x"] = p32.M * p32.s_u / p32.delta
    cells = p32.groupby(["M", "s_u", "delta"]).agg(x=("x", "first"), y=("built_count_overlap", "mean"),
                                                   ysd=("built_count_overlap", "std")).reset_index()
    slope, icpt = np.polyfit(np.log(p32.x), np.log(p32.built_count_overlap), 1)
    rng = np.random.default_rng(0); seeds = p32.seed.unique(); bs = []
    for _ in range(2000):
        q = pd.concat([p32[p32.seed == s] for s in rng.choice(seeds, len(seeds))])
        bs.append(np.polyfit(np.log(q.x), np.log(q.built_count_overlap), 1)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    note(2, "slope of log count on log(M*s_u/delta), width-32 world points", float(slope), "results/e2/points.csv")
    note(2, "  95% CI low (bootstrap over worlds)", float(lo), "results/e2/points.csv")
    note(2, "  95% CI high (bootstrap over worlds)", float(hi), "results/e2/points.csv")
    fig, ax = plt.subplots(figsize=(NARROW, 2.9), layout="constrained")
    levels = sorted(cells.s_u.unique())
    for i, su in enumerate(levels[::-1]):
        g = cells[cells.s_u == su]
        ax.errorbar(g.x, g.y, yerr=g.ysd, fmt="o", ms=4.5, color=BLUE_RAMP[i], mec="white", mew=0.8, elinewidth=0.9,
                    capsize=0, label=f"$s_u$ = {su:g}")
    xs = np.logspace(np.log10(cells.x.min() * 0.8), np.log10(cells.x.max() * 1.2), 50)
    ax.plot(xs, np.exp(icpt) * xs ** slope, color=MUTED, lw=1.1, ls="--")
    ax.text(0.03, 0.97, f"slope {slope:.2f} [{lo:.2f}, {hi:.2f}]\n(joint fit: $M$ 0.91, $1/\\delta$ 0.91, $s_u$ 0.98)",
            transform=ax.transAxes, va="top", fontsize=7, linespacing=1.4)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.minorticks_off()
    ax.set_xticks([300, 600, 1000, 2000, 3000]); ax.set_xticklabels(["300", "600", "1,000", "2,000", "3,000"])
    ax.set_yticks([20, 40, 60, 100, 200]); ax.set_yticklabels(["20", "40", "60", "100", "200"])
    ax.set_xlabel(r"builders × strength ÷ decay  ($M\,s_u/\delta$)"); ax.set_ylabel("built items at steady state")
    ax.legend(loc="lower right", title="shaping strength", handletextpad=0.3, borderaxespad=0.3)
    save(fig, "fig2_upkeep")


# ------------------------------------------------------------------ Fig 3: builders' key
def fig3():
    wm = pd.read_csv(R("e3", "world_means.csv"))
    arms = [("same_body", "builders'\nbody"), ("same_type", "same\ntype"), ("outsider", "outsider")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.55), layout="constrained")
    fig.get_layout_engine().set(wspace=0.12)
    for ax, col, ttl, unit in ((axes[0], "h_drop", "A  Compression at the gate (L3c)", "entropy drop (bits)"),
                               (axes[1], "code_saving", "B  Code-length saving (L3d)", "bits saved over 8,000\nitem–action offers")):
        for i, (arm, nm) in enumerate(arms):
            m, lo, hi = mean_ci(wm[(wm.rule == "eye025") & (wm.arm == arm)][col])
            ax.bar(i, m, width=0.62, color=BLUE_RAMP[i], yerr=[[m - lo], [hi - m]], error_kw=dict(ecolor=INK, lw=0.9))
            ax.text(i + 0.33, m, f"{m:.3f}" if col == "h_drop" else f"{m:.0f}", ha="left", va="center", fontsize=7)
            note(3, f"{col} {arm} (kappa 0.25)", m, "results/e3/world_means.csv")
            mw = wm[(wm.rule == "wear") & (wm.arm == arm)][col].mean()
            ax.plot([i - 0.31, i + 0.31], [mw, mw], color=C_WEAR, lw=1.4, ls=(0, (2, 1.5)))
        out = wm[(wm.rule == "eye025") & (wm.arm == "outsider")][col].mean()
        sb = wm[(wm.rule == "eye025") & (wm.arm == "same_body")][col].mean()
        ax.set_xticks(range(3)); ax.set_xticklabels([a[1] for a in arms]); ax.set_ylabel(unit); ax.grid(axis="x")
        ax.set_xlim(-0.5, 2.9); ax.set_ylim(0, 0.82 if col == "h_drop" else 760)
        ax.set_title(ttl)
        ax.text(0.02, 0.985, f"key-specific: {100 * (sb - out) / sb:.0f}% of the builders' gain", transform=ax.transAxes,
                ha="left", va="top", fontsize=7, color=MUTED)
    axes[0].plot([], [], color=C_WEAR, lw=1.4, ls=(0, (2, 1.5)), label="wear control (no key)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center")
    fig.align_titles()
    save(fig, "fig3_key")


# ------------------------------------------------------------------ Fig 4: newcomers and fragility
def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.75), layout="constrained", gridspec_kw=dict(width_ratios=[1, 1, 1.0]))
    fig.get_layout_engine().set(wspace=0.1)
    raised = pd.read_csv(R("e10", "raised.csv"))
    prim = raised[(raised.cond == "skip") & (raised.arm == "grid") & (raised.width == 4) & (raised.budget == 60000)]
    for ax, rule, col, nm, lab in ((axes[0], "wear", C_WEAR, "wear", "A"), (axes[1], "eye1", C_EYE, "shaping by eye", "B")):
        for d, ls, case in (("e8_fixed", "-", "fully trained"), ("e8", (0, (3, 2)), "partly trained")):
            e = pd.read_csv(R(d, "e8.csv")); e = e[e.width == 4]
            for arm, c in ((f"treat_{rule}", col), ("natural", C_WILD)):
                g = e[e.arm == arm].groupby("seed")[["edge_acc_before", "edge_acc_after"]].mean().mean()
                ax.plot([0, 1], [g.edge_acc_before, g.edge_acc_after], color=c, ls=ls, lw=1.6, marker="o", ms=3.8, mec="white")
                note(4, f"{d} {arm} edge accuracy start/end of T2 (width 4)",
                     f"{g.edge_acc_before:.3f}/{g.edge_acc_after:.3f}", f"results/{d}/e8.csv")
        un = prim[prim.kind == rule].groupby("seed").edge_acc.mean().mean()
        wl = prim[prim.kind == "wild"].groupby("seed").edge_acc.mean().mean()
        ax.scatter([1.2], [un], marker="D", s=22, color=col, edgecolor=INK, lw=0.5, zorder=3)
        ax.scatter([1.2], [wl], marker="D", s=22, color=C_WILD, edgecolor=INK, lw=0.5, zorder=3)
        ax.text(1.29, un, "newcomer", fontsize=7, va="center")
        ax.text(1.29, wl, "wild", fontsize=7, va="center")
        note(4, f"untrained newcomers raised in {rule} world, edge accuracy (width 4, 60k)", float(un), "results/e10/raised.csv")
        note(4, "untrained agents raised wild (not newcomers), edge accuracy (width 4, 60k)", float(wl), "results/e10/raised.csv")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["entry", "+30k\nsteps"]); ax.set_xlim(-0.15, 1.85)
        ax.set_ylim(0.6, 0.95); ax.set_ylabel("edge accuracy" if rule == "wear" else ""); ax.grid(axis="x")
        ax.set_title(f"{lab}  {nm.capitalize()} world")
    axes[0].plot([], [], color=MUTED, ls="-", lw=1.4, label="fully trained (60k)")
    axes[0].plot([], [], color=MUTED, ls=(0, (3, 2)), lw=1.4, label="partly trained")
    axes[0].plot([], [], color=C_WILD, lw=1.4, label="natural control")
    axes[0].scatter([], [], marker="D", s=18, color="white", edgecolor=INK, lw=0.5, label="untrained newcomer (60k)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, handlelength=1.8, columnspacing=1.6)
    e9 = pd.read_csv(R("e10", "e9.csv")); e9 = e9[e9.cond == "skip"]
    ax = axes[2]
    wild = e9[e9.kind == "wild"].groupby("seed").decisive_pay_mean.mean()
    for i, (k, c, nm) in enumerate((("wild", C_WILD, "raised\nwild"), ("wear", C_WEAR, "raised in\nwear world"),
                                    ("eye1", C_EYE, "raised in\neye world"))):
        x = e9[e9.kind == k].groupby("seed").decisive_pay_mean.mean()
        m, lo, hi = mean_ci(x)
        ax.bar(i, m, width=0.62, color=c, yerr=[[m - lo], [hi - m]], error_kw=dict(ecolor=INK, lw=0.9))
        if k != "wild":
            ax.text(i, hi + 0.01, f"−{100 * (wild.mean() - m) / wild.mean():.0f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(3)); ax.set_xticklabels(["wild", "wear", "eye"]); ax.set_xlabel("raised in")
    ax.set_ylim(0, 0.45); ax.set_ylabel("payoff per decisive turn"); ax.grid(axis="x")
    ax.set_title("C  Snap-back (E9)")
    fig.align_titles()
    save(fig, "fig4_newcomers")


# ------------------------------------------------------------------ Fig 5: silent home, confident error
def fig5():
    r = pd.read_csv(R("e10", "raised.csv"))
    r = r[(r.arm == "grid") & (r.width == 32) & (r.budget == 60000)]
    wm = r.groupby(["seed", "kind", "cond"])[["home_logloss", "conf_err"]].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.85), sharey=True, layout="constrained")
    fig.get_layout_engine().set(wspace=0.08)
    for ax, rule, col, nm in ((axes[0], "wear", C_WEAR, "A  Wear"), (axes[1], "eye1", C_EYE, "B  Shaping by eye")):
        for kind, c, kn in (("wild", C_WILD, "wild"), (rule, col, "newcomer")):
            for cond, mk, fill in (("skip", "o", True), ("passive", "^", False)):
                g = wm[(wm.kind == kind) & (wm.cond == cond)]
                ax.scatter(g.home_logloss, g.conf_err, marker=mk, s=22, facecolor=c if fill else "white", edgecolor=c,
                           lw=1.0, label=f"{kn}, {'skip rule' if cond == 'skip' else 'passive'}")
        ax.set_xscale("log"); ax.set_xlabel("home log loss (nats, log scale)")
        ax.set_title(nm)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0,
                  handletextpad=0.3)   # one legend per panel, in that panel's colours
    axes[0].set_ylabel("confident error on unbuilt edges")
    fig.align_titles()
    save(fig, "fig5_silent_home")


# ------------------------------------------------------------------ Fig 6: mechanism
def fig6():
    s = pd.read_csv(R("e10", "summary_L10m.csv"))
    fig, ax = plt.subplots(figsize=(NARROW, 2.75), layout="constrained")
    for rule, col, nm in (("wear", C_WEAR, "wear"), ("eye1", C_EYE, "shaping by eye")):
        g = s[s.rule == rule].sort_values("eps_min")
        ax.fill_between(g.eps_min, g.lo, g.hi, color=col, alpha=0.18, lw=0)
        ax.plot(g.eps_min, g.stretch_diff, color=col, marker="o", ms=4.5, mec="white")
        ax.text(g.eps_min.iloc[-1] * 1.12, g.stretch_diff.iloc[-1], nm, fontsize=7, va="center")
    ax.axhline(0, color=MUTED, lw=0.7)
    ax.set_xscale("log"); ax.set_xticks([0.02, 0.1, 0.3]); ax.set_xticklabels(["0.02", "0.1", "0.3"]); ax.set_xlim(0.015, 1.3)   # room for the "shaping by eye" label
    ax.minorticks_off()
    ax.set_xlabel(r"learner's random-act floor $\varepsilon_{\min}$"); ax.set_ylabel("extra arc stretch, built − wild (deg)")
    ax.text(0.98, 0.98, "builders' policy fixed\n" r"($\varepsilon_{\min}$ 0.02) in every arm", transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=MUTED, linespacing=1.4)
    save(fig, "fig6_mechanism")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, fig6):
        f()
    pd.DataFrame(NUM).to_csv(os.path.join(OUT, "figure_numbers.csv"), index=False)
    print("wrote", sorted(os.listdir(OUT)))
