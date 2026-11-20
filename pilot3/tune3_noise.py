"""D1 (seed 3000, wild agents only): legibility over lambda x observation noise x training length."""
import numpy as np, world3 as W
from tune3_lambda import cost, STEP1_EXPLORE
for lam in (0.4, 0.25):
    for noise in (0.05, 0.02, 0.01):
        rng = np.random.default_rng(3000); nat = W.World(rng, noise=noise); B = W.make_bodies(rng, [1, 1, 2, 2], lam)
        test = W.natural_test_set(np.random.default_rng(3077))
        flat = nat.copy(); flat.u[:] = 0.0; flat.home_u[:] = 0.0
        rows = []
        for body in (0, 2):
            w = nat.copy(); ag = W.make_agents(np.random.default_rng(11 + body), 1, 32, body)
            tr = []
            for c in range(3):
                W.step(w, B, ag, None, 20000, seed=500 + body * 10 + c, explore=STEP1_EXPLORE)
                es = W.edge_scores(B, ag, 0, test, np.random.default_rng(5), noise=noise)
                tr.append((round(es["edge_acc"], 3), round(es["decisive_ratio"], 3)))
            ca, _ = cost(nat.copy(), B, body, 21 + body); cf, _ = cost(flat.copy(), B, body, 31 + body)
            rows.append((int(B["types"][body]), tr, ca, cf))
        print(f"lambda={lam} noise={noise}: (type, [(edge, decisive) at 20k/40k/60k], cost active, cost flat): {rows}", flush=True)
