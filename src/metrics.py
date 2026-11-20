# Copied verbatim from D:\ML\AGENTS\Research\Convergence Experiments\src\metrics.py at commit 9c7dd7d
# (BRIEF.md: for E_K, so this project reproduces even if that folder changes). Do not edit; wrap it instead.
"""Distance-chain statistics (R1-R4, G, K), permutation nulls and item bootstrap weights.

Definitions follow BRIEF.md section 3. A Geometry holds everything precomputed for one model on one
item set, so that a permuted statistic only needs index lookups. Row i of a distance matrix may differ
from column i: R1-R3 and G read the upper triangle, while R4 and K read rows.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

DENSITIES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
PRIMARY_RHO = 0.01
BOOTSTRAP_PAIRS = 2_000_000


def preprocess(X, mode="cosine"):
    """Mean-centre, then L2-normalise rows for mode 'cosine'. Mode 'euclidean' only centres."""
    if mode not in ("cosine", "euclidean"):
        raise ValueError(mode)
    X = X.float()
    X = X - X.mean(0, keepdim=True)
    return F.normalize(X, dim=1) if mode == "cosine" else X


def distance_matrix(X, mode="cosine"):
    """Pairwise distances of preprocessed features."""
    D = 1 - X @ X.T if mode == "cosine" else torch.cdist(X, X)
    return D.fill_diagonal_(0)


def rank_average(x):
    """1-based ranks of a 1-D tensor, with ties given their average rank (scipy's 'average')."""
    order = x.argsort()
    s = x[order]
    new = torch.cat([torch.ones(1, dtype=torch.bool, device=x.device), s[1:] != s[:-1]])
    starts = new.nonzero().squeeze(1)
    counts = torch.diff(starts, append=torch.tensor([x.numel()], device=x.device))
    avg = starts.double() + (counts.double() - 1) / 2 + 1
    ranks = torch.empty_like(x, dtype=torch.float64)
    ranks[order] = avg[new.cumsum(0) - 1]
    return ranks


def n_edges(rho, m):
    return max(1, round(rho * m))


def n_neighbours(rho, n):
    return max(1, round(rho * (n - 1)))


def pair_positions(i, j, n):
    """Index into the upper-triangle vector of unordered pairs (i, j), i != j."""
    a, b = torch.minimum(i, j), torch.maximum(i, j)
    return a * n - a * (a + 1) // 2 + b - a - 1


@dataclass
class Geometry:
    """Precomputed distance structure of one representation."""
    n: int
    m: int
    iu: torch.Tensor
    ju: torch.Tensor
    d_mean: float
    d_c: torch.Tensor  # centred upper-triangle distances
    d_ss: float
    r_c: torch.Tensor  # centred global ranks of the upper triangle
    r_ss: float
    order: torch.Tensor  # pair indices from closest to farthest, first m * max(rho) entries
    pair_rank: torch.Tensor  # ordinal global rank of each pair
    nbr: torch.Tensor  # N x kmax neighbour indices, closest first
    row_rank: torch.Tensor  # N x N ordinal rank of j among i's neighbours
    row_c: torch.Tensor  # N x N centred average ranks, zero on the diagonal
    row_ss: torch.Tensor  # per-row sum of squares of row_c


def geometry(D, densities=DENSITIES):
    """Build a Geometry from an N x N distance matrix."""
    n = D.shape[0]
    device = D.device
    m = n * (n - 1) // 2
    iu, ju = torch.triu_indices(n, n, 1, device=device)
    d = D[iu, ju].double()
    d_mean = d.mean().item()
    d_c = d - d_mean
    r = rank_average(d)
    r_c = r - r.mean()
    full_order = d.argsort(stable=True)
    pair_rank = torch.empty(m, dtype=torch.int32, device=device)
    pair_rank[full_order] = torch.arange(m, dtype=torch.int32, device=device)
    order = full_order[:n_edges(max(densities), m)].int()
    del d, r, full_order

    Dn = D.clone().fill_diagonal_(float("inf"))
    row_order = Dn.argsort(dim=1, stable=True)
    row_rank = torch.empty((n, n), dtype=torch.int32, device=device)
    row_rank.scatter_(1, row_order, torch.arange(n, dtype=torch.int32, device=device).expand(n, n))
    row_c = (row_rank.float() + 1 - n / 2)
    sorted_vals = Dn.gather(1, row_order)[:, :-1]
    tied = (sorted_vals[:, 1:] == sorted_vals[:, :-1]).any(1).nonzero().squeeze(1)
    keep = torch.ones(n, dtype=torch.bool, device=device)
    for i in tied.tolist():
        keep[:] = True
        keep[i] = False
        row_c[i, keep] = (rank_average(Dn[i, keep]) - n / 2).float()
    row_c.fill_diagonal_(0)
    row_ss = row_c.double().pow(2).sum(1)
    nbr = row_order[:, :n_neighbours(max(densities), n)].int()
    return Geometry(n, m, iu, ju, d_mean, d_c, d_c.pow(2).sum().item(), r_c, r_c.pow(2).sum().item(),
                    order, pair_rank, nbr, row_rank, row_c, row_ss)


def statistics(gx, gy, perm=None, densities=DENSITIES):
    """All rungs for one pair; perm re-labels Y's items, so that row i of Y becomes row perm[i].

    Returns scalars R1-R4 and G_rho / K_rho, plus per-item tensors under 'items'.
    """
    n, m = gx.n, gx.m
    if perm is None:
        d_y, r_y = gy.d_c, gy.r_c
        rows = (gx.row_c.double() * gy.row_c.double()).sum(1)
        edge_rank = gy.pair_rank[gx.order.long()]
        nbr_rank = gy.row_rank.gather(1, gx.nbr.long())
    else:
        pos = pair_positions(perm[gx.iu], perm[gx.ju], n)
        d_y, r_y = gy.d_c[pos], gy.r_c[pos]
        rows = (gx.row_c.double() * gy.row_c[perm[:, None], perm].double()).sum(1)
        o = gx.order.long()
        edge_rank = gy.pair_rank[pair_positions(perm[gx.iu[o]], perm[gx.ju[o]], n)]
        nbr_rank = gy.row_rank[perm[:, None], perm[gx.nbr.long()]]

    cov = (gx.d_c * d_y).sum().item()
    out = {
        "R1": 2 * cov / (gx.d_ss + gy.d_ss + m * (gx.d_mean - gy.d_mean) ** 2),
        "R2": cov / (gx.d_ss * gy.d_ss) ** 0.5,
        "R3": (gx.r_c * r_y).sum().item() / (gx.r_ss * gy.r_ss) ** 0.5,
    }
    y_ss = gy.row_ss if perm is None else gy.row_ss[perm]
    r4_items = rows / (gx.row_ss * y_ss).sqrt()
    out["R4"] = r4_items.mean().item()
    items = {"R4": r4_items}
    for rho in densities:
        e, k = n_edges(rho, m), n_neighbours(rho, n)
        out[f"G_{rho}"] = (edge_rank[:e] < e).sum().item() / e
        k_items = (nbr_rank[:, :k] < k).double().mean(1)
        out[f"K_{rho}"] = k_items.mean().item()
        items[f"K_{rho}"] = k_items
    out["items"] = items
    return out


def knn_overlap(gx, gy, k, perm=None):
    """K at an explicit neighbour count, within the width kept in the Geometry."""
    nbr = gx.nbr[:, :k].long()
    if perm is None:
        r = gy.row_rank.gather(1, nbr)
    else:
        r = gy.row_rank[perm[:, None], perm[nbr]]
    return (r < k).double().mean().item()


def stratified_permutation(labels, gen):
    """Random permutation that only moves items within their own stratum. labels is a CPU tensor."""
    perm = torch.arange(labels.numel())
    for s in labels.unique():
        idx = (labels == s).nonzero().squeeze(1)
        perm[idx] = idx[torch.randperm(idx.numel(), generator=gen)]
    return perm


def geometry_to(g, device):
    """Copy a Geometry to a device."""
    moved = {f: getattr(g, f).to(device) if torch.is_tensor(getattr(g, f)) else getattr(g, f)
             for f in g.__dataclass_fields__}
    return Geometry(**moved)


def analytic_null_means(n, densities=DENSITIES):
    """Exact permutation-null means: 0 for R1-R4, the graph density for G and K."""
    m = n * (n - 1) // 2
    out = {"R1": 0.0, "R2": 0.0, "R3": 0.0, "R4": 0.0}
    for rho in densities:
        out[f"G_{rho}"] = n_edges(rho, m) / m
        out[f"K_{rho}"] = n_neighbours(rho, n) / (n - 1)
    return out


def permutation_null(gx, gy, n_perm=100, seed=0, densities=DENSITIES):
    """Mean and sd of every statistic over n_perm random re-labellings of Y's items."""
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(n_perm):
        s = statistics(gx, gy, torch.randperm(gx.n, generator=gen).to(gx.iu.device), densities)
        s.pop("items")
        rows.append(s)
    keys = rows[0].keys()
    t = torch.tensor([[r[k] for k in keys] for r in rows], dtype=torch.float64)
    return {k: (t[:, i].mean().item(), t[:, i].std().item()) for i, k in enumerate(keys)}


def excess(s, m):
    """Normalised excess E = (s - m) / (1 - m)."""
    return (s - m) / (1 - m)


def item_weights(n, b, seed=0):
    """B x N multinomial resampling counts, shared by every pair in an analysis."""
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randint(n, (b, n), generator=gen)
    w = torch.zeros(b, n, dtype=torch.float32)
    w.scatter_add_(1, idx, torch.ones(b, n, dtype=torch.float32))
    return w


def bootstrap_pairs(m, seed=0, size=BOOTSTRAP_PAIRS, device="cpu"):
    """Fixed subsample of pair indices for the R1-R3 bootstrap; all pairs when m <= size."""
    if m <= size:
        return torch.arange(m, device=device)
    gen = torch.Generator().manual_seed(seed)
    return torch.randperm(m, generator=gen)[:size].to(device)


def _weighted_corr(x, y, w):
    wsum = w.sum()
    mx, my = (w * x).sum() / wsum, (w * y).sum() / wsum
    dx, dy = x - mx, y - my
    sxx, syy, sxy = (w * dx * dx).sum(), (w * dy * dy).sum(), (w * dx * dy).sum()
    return sxy, sxx, syy, wsum, mx, my


def weighted_statistics(gx, gy, w, subset, rho=PRIMARY_RHO):
    """One bootstrap replicate, with neighbour sets and graphs held at their full-sample values.

    Item weights w act on pairs as w_i * w_j over i < j, so a resampled item is never paired with its
    own copy. This avoids the zero-distance duplicates that plain item resampling would create.
    """
    n, m = gx.n, gx.m
    i, j = gx.iu[subset], gx.ju[subset]
    wp = (w[i] * w[j]).double()
    sxy, sxx, syy, wsum, mx, my = _weighted_corr(gx.d_c[subset] + gx.d_mean, gy.d_c[subset] + gy.d_mean, wp)
    out = {"R1": (2 * sxy / (sxx + syy + wsum * (mx - my) ** 2)).item(), "R2": (sxy / (sxx * syy).sqrt()).item()}
    sxy, sxx, syy, *_ = _weighted_corr(gx.r_c[subset], gy.r_c[subset], wp)
    out["R3"] = (sxy / (sxx * syy).sqrt()).item()

    r4 = (gx.row_c.double() * gy.row_c.double()).sum(1) / (gx.row_ss * gy.row_ss).sqrt()
    wd = w.double()
    out["R4"] = ((wd * r4).sum() / wd.sum()).item()

    e, k = n_edges(rho, m), n_neighbours(rho, n)
    o = gx.order[:e].long()
    we = (w[gx.iu[o]] * w[gx.ju[o]]).double()
    shared = gy.pair_rank[o] < e
    out[f"G_{rho}"] = ((we * shared).sum() / we.sum()).item()
    k_items = (gy.row_rank.gather(1, gx.nbr[:, :k].long()) < k).double().mean(1)
    out[f"K_{rho}"] = ((wd * k_items).sum() / wd.sum()).item()
    return out


def hierarchical_bootstrap(values, pair_groups, n_boot=2000, seed=0):
    """Resample models, then items, for a table of pair statistics.

    values: B x P array, one row per item-weight replicate and one column per pair.
    pair_groups: list of P (group_a, a, group_b, b) tuples naming each pair's models. A model is resampled
        within its own group. Weights are products of multiplicities over pairs in the table, and a
        model's pair with its own copy is not in the table, so such pairs are excluded by construction.
    Returns dicts of bootstrap means and of the proportion of pairs above zero, as numpy arrays.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    groups = {}
    for ga, a, gb, b in pair_groups:
        groups.setdefault(ga, set()).add(a)
        groups.setdefault(gb, set()).add(b)
    groups = {g: sorted(s) for g, s in groups.items()}
    index = {g: {m: i for i, m in enumerate(ms)} for g, ms in groups.items()}
    pa = [(ga, index[ga][a]) for ga, a, _, _ in pair_groups]
    pb = [(gb, index[gb][b]) for _, _, gb, b in pair_groups]
    means, props = [], []
    for _ in range(n_boot):
        row = values[rng.integers(values.shape[0])]
        counts = {g: rng.multinomial(len(ms), [1 / len(ms)] * len(ms)) for g, ms in groups.items()}
        w = np.array([counts[ga][i] * counts[gb][j] for (ga, i), (gb, j) in zip(pa, pb)], dtype=float)
        if w.sum() == 0:
            continue
        means.append((w * row).sum() / w.sum())
        props.append((w * (row > 0)).sum() / w.sum())
    return np.array(means), np.array(props)


def hubness(g, rho=PRIMARY_RHO):
    """Skewness of k-occurrence counts at density rho."""
    k = n_neighbours(rho, g.n)
    c = torch.bincount(g.nbr[:, :k].flatten().long(), minlength=g.n).double()
    return (((c - c.mean()) / c.std(unbiased=False)) ** 3).mean().item()
