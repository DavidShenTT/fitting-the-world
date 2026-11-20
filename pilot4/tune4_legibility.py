"""Tuning step 1 (seed 4000, wild agents only): legibility under unequal arcs (S1) and an unbiased boundary (S2)."""
import numpy as np, world4 as W
for lam, noise in ((0.4, 0.01), (0.4, 0.005), (0.25, 0.01)):
    rng = np.random.default_rng(4000); nat = W.World(rng, noise=noise); B = W.make_bodies(rng, [1, 1, 1, 1, 2, 2, 2, 2], lam)
    test = W.natural_test_set(np.random.default_rng(4077))
    rows = []
    for body in (0, 4):
        w = nat.copy(); ag = W.make_agents(np.random.default_rng(11 + body), 1, 32, body)
        tr = []
        for c in range(3):
            W.step(w, B, ag, None, 20000, seed=500 + body * 10 + c)
            es = W.edge_scores(B, ag, 0, test, np.random.default_rng(5), noise=noise)
            bs = W.boundary_shift(B, ag, 0, np.random.default_rng(6), noise=noise)
            tr.append((round(es["wide_edge_acc"], 3), round(es["narrow_edge_acc"], 3), round(es["wide_decisive_ratio"], 3),
                       round(es["narrow_decisive_ratio"], 3), round(bs["hw_wide"], 1), round(bs["hw_narrow"], 1)))
        rows.append((int(B["types"][body]), tr))
    print(f"lambda={lam} noise={noise}: (type, [(edge wide, edge narrow, dec wide, dec narrow, hw wide, hw narrow) at 20k/40k/60k]): {rows}", flush=True)
