"""Shared statistics (PREREGISTRATION.md section 2): the world is the unit; agents are averaged within a world first;
95% percentile-bootstrap intervals over worlds (10,000 resamples); paired ratios as geometric means over worlds."""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N_BOOT = 10000
# items behind one agent's confident-error rate: 2 actions x (300 inside + 300 outside) per arc type, 2 arc types
CONF_ITEMS = 2400


def boot_ci(x, stat=np.mean, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    r = np.random.default_rng(seed)
    m = np.array([stat(x[i]) for i in r.integers(0, len(x), (N_BOOT, len(x)))])
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def world_means(d, keys, cols):
    g = d.groupby(keys)
    out = g[cols].mean()
    out["n_agents"] = g.size()
    return out.reset_index()


def paired_ratio(built, wild, col, count_items=None):
    """Geometric mean over worlds of built / wild (world means, joined on seed). For count-based rates (count_items =
    items per agent), a world whose wild rate is zero gets +0.5 counts in numerator and denominator (that world only).
    Returns (geomean, ci_lo, ci_hi, n_worlds, n_zero_wild)."""
    m = built[["seed", col, "n_agents"]].merge(wild[["seed", col, "n_agents"]], on="seed", suffixes=("_b", "_w"))
    b, w = m[f"{col}_b"].to_numpy(float), m[f"{col}_w"].to_numpy(float)
    zero = w <= 0
    if count_items is not None and zero.any():
        nb, nw = m.n_agents_b.to_numpy() * count_items, m.n_agents_w.to_numpy() * count_items
        b = np.where(zero, (b * nb + 0.5) / nb, b)
        w = np.where(zero, 0.5 / nw, w)
    lr = np.log(b / w)
    lo, hi = boot_ci(lr)
    return float(np.exp(lr.mean())), float(np.exp(lo)), float(np.exp(hi)), int(len(lr)), int(zero.sum())


def diff_means(built, wild, col):
    """Mean over worlds of (built - wild), paired by seed, with its bootstrap interval."""
    m = built[["seed", col]].merge(wild[["seed", col]], on="seed", suffixes=("_b", "_w"))
    d = m[f"{col}_b"] - m[f"{col}_w"]
    lo, hi = boot_ci(d)
    return float(d.mean()), lo, hi, int(len(d))


def save(df, path, name):
    os.makedirs(path, exist_ok=True)
    df.to_csv(os.path.join(path, name), index=False)


def show(df, title):
    print(f"\n== {title}")
    with pd.option_context("display.width", 220, "display.max_columns", 40):
        print(df.round(4).to_string(index=False))
