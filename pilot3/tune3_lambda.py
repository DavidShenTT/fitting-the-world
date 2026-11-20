"""Tuning step 1 (seed 3000, wild agents only): legibility lambda for 2-D structured bodies."""
import numpy as np, world3 as W

STEP1_EXPLORE = dict(eps_min=0.02, eps_m=0.3, m=0.2)  # the default when steps 1 and D1 ran

def cost(w, B, body, seed, cap=20000):
    ag = W.make_agents(np.random.default_rng(seed), 1, 32, body)
    for s in range(0, cap + 1, 250):
        if s > 0 and W.competence(w, B, ag, 0, np.random.default_rng(seed + s)) >= 0.9:
            return s, ag
        if s < cap:
            W.step(w, B, ag, None, 250, seed=seed * 100 + s, explore=STEP1_EXPLORE)
    return None, ag

if __name__ == "__main__":
  for lam in (1.0, 0.6, 0.4, 0.25):
        rng = np.random.default_rng(3000); nat = W.World(rng); B = W.make_bodies(rng, [1, 1, 2, 2], lam)
        test = W.natural_test_set(np.random.default_rng(3077))
        flat = nat.copy(); flat.u[:] = 0.0; flat.home_u[:] = 0.0
        rows = []
        for body in range(4):
            w = nat.copy(); ag = W.make_agents(np.random.default_rng(11 + body), 1, 32, body)
            W.step(w, B, ag, None, 20000, seed=400 + body, explore=STEP1_EXPLORE)
            es = W.edge_scores(B, ag, 0, test, np.random.default_rng(5))
            c_active, _ = cost(nat.copy(), B, body, 21 + body)
            c_flat, _ = cost(flat.copy(), B, body, 31 + body)
            rows.append((B["types"][body], round(es["edge_acc"], 3), round(es["decisive_ratio"], 3), c_active, c_flat))
        print(f"lambda={lam}: (type, edge acc, decisive ratio, cost active, cost flat): {rows}", flush=True)
