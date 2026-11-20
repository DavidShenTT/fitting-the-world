"""Diagnostic (seed 3000): wild vs built-raised (wear) agents' mean predicted acceptance of the boundary action,
by signed distance from the arc edge (negative = inside the arc, where the action is accepted). Not a prediction test."""
import numpy as np, world3 as W, pandas as pd
from pilot3 import build, agent, live, Seeder, WEAR, NOISE, LAM
if __name__ == "__main__":
    sd = Seeder(9); rng = np.random.default_rng(3000); nat = W.World(rng, noise=NOISE); B = W.make_bodies(rng, [1, 1, 2, 2], LAM)
    wb, fr = build(nat, B, WEAR, 0, sd)
    t = W.natural_test_set(np.random.default_rng(5), n=20000)
    th = t["alpha"]; e = W.is_edge(th, 30.0)
    ba = W.boundary_action(th[e]); acc = W.accepted_matrix(th[e])[np.arange(e.sum()), ba]
    edges = np.array([[(c - 60) % 360, (c + 60) % 360] for c in W.CENTRES])
    dist = np.abs(W.wrapdiff(th[e][:, None], edges[ba])).min(axis=1) * np.where(acc, -1, 1)
    rows = []
    for kind in ("wild", "built"):
        ag = agent(77, 32)
        if kind == "wild":
            w = nat.copy(); live(w, B, ag, None, None, 60000, sd)
        else:
            w = wb.copy(); live(w, B, ag, fr, WEAR, 60000, sd)
        p = W.predict(ag, 0, W.observe(B, 0, th[e], t["u"][e], t["kind"][e], np.random.default_rng(1), NOISE))[np.arange(e.sum()), ba]
        for lo in range(-15, 30, 5):
            msk = (dist >= lo) & (dist < lo + 5)
            rows.append(dict(kind=kind, dist=f"{lo}..{lo+5}", mean_p=round(float(p[msk].mean()), 3)))
    print(pd.DataFrame(rows).pivot(index="dist", columns="kind", values="mean_p").reindex([f"{lo}..{lo+5}" for lo in range(-15, 30, 5)]).to_string())
