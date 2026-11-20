"""D3 tuning on seed 2000, competence only."""
import numpy as np, world2 as W
rng = np.random.default_rng(2000); base = W.World(rng); B = W.make_bodies(rng, 3)
for eps in (0.05, 0.2):
    for bias in (0.0, 1.5):
        for lr in (0.05, 0.1):
            w = base.copy(); ag = W.make_agents(np.random.default_rng(1), 1, 32); ag["b2"][:] = bias
            hit = None; tr = []
            for c in range(30):
                W.step(w, B, ag, None, 500, seed=10 + c, lr=lr, eps=eps)
                cp = W.competence(w, B, ag, 0, np.random.default_rng(99 + c)); tr.append(round(cp, 3))
                if hit is None and cp >= 0.9: hit = (c + 1) * 500
            print(f"eps={eps} bias={bias} lr={lr}: 0.9 first at {hit}; every 2.5k: {tr[4::5]}")
