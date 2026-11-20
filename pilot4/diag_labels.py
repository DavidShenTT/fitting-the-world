"""D3 (seed 4000): the balance of labels each agent learns from, under the skip rule and under passive feedback.

Runs one step at a time so each label's item angle is read right after the step that produced it.
Writes nothing but stdout; redirect to results/diag_labels.txt.
"""
import numpy as np, world4 as W

LAM, NOISE, STEPS, N_AG = 0.4, 0.01, 60000, 3


def fold(alpha):
    a = alpha % 180.0
    return np.where(a > 90, 180 - a, a)   # 0 = a wide centre, 90 = a narrow centre


def raise_one(world, B, fr, kw, seed, sbase, passive):
    ag = W.make_agents(np.random.default_rng(seed), 1, 32, 0)
    offer = np.empty(STEPS, dtype=np.int64); act = np.empty(STEPS, dtype=np.int64); ang = np.empty(STEPS)
    for t in range(STEPS):
        r = W.step(world, B, ag, fr, 1, seed=sbase + t, passive=passive, **kw)
        offer[t], act[t] = r["offer"][0], r["act"][0]
        ang[t] = world.alpha[r["item"][0]]
    y = W.accepted_matrix(ang)[np.arange(STEPS), offer]
    upd = np.ones(STEPS, bool) if passive else act.astype(bool)
    wide = np.isin(offer, W.WIDE); win = (fold(ang) >= 35) & (fold(ang) <= 65)
    out = {"updates": upd.sum()}
    for name, sel in (("wide", wide), ("narrow", ~wide)):
        s = upd & sel
        out[f"pos_{name}"] = y[s].mean()
        out[f"win_per_1000_{name}"] = 1000 * (s & win).sum() / upd.sum()
        out[f"pos_win_{name}"] = y[s & win].mean() if (s & win).sum() > 0 else np.nan
    bs = W.boundary_shift(B, ag, 0, np.random.default_rng(6), noise=NOISE)
    out["shift_wide"], out["shift_narrow"] = bs["shift_wide"], bs["shift_narrow"]
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(4000); nat = W.World(rng, noise=NOISE); B = W.make_bodies(rng, [1, 1, 1, 1, 2, 2, 2, 2], LAM)
    kw = dict(rule=W.WEAR, eta=0.15)
    wear = nat.copy(); bl = W.make_agents(np.random.default_rng(2), 6, 32, body=0)
    W.step(wear, B, bl, None, 30000, seed=70, learner_shapes=True, **kw)
    print("D3: share of acceptances among the labels each agent learned from (60k steps, width 32, mean of 3 agents)")
    print("window = 35-65 degrees from a wide centre, where the built worlds are empty (D2a)")
    keys = ["updates", "pos_wide", "pos_narrow", "win_per_1000_wide", "win_per_1000_narrow", "pos_win_wide", "pos_win_narrow",
            "shift_wide", "shift_narrow"]
    print(f"{'regime':8s} {'kind':5s} " + " ".join(f"{k:>18s}" for k in keys))
    for passive in (False, True):
        for kind in ("wild", "wear"):
            rows = []
            for k in range(N_AG):
                if kind == "wild":
                    rows.append(raise_one(nat.copy(), B, None, {}, 300 + k, 70000 * (6 * passive + k), passive))
                else:
                    rows.append(raise_one(wear.copy(), B, W.copy_agents(bl), kw, 400 + k, 70000 * (6 * passive + 3 + k), passive))
            m = {key: np.nanmean([r[key] for r in rows]) for key in keys}
            print(f"{'passive' if passive else 'skip':8s} {kind:5s} " + " ".join(f"{m[k]:18.3f}" for k in keys), flush=True)
