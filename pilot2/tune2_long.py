"""D3: competence over 40,000 steps (seed 2000, lone width-32 agent, natural world)."""
import numpy as np, world2 as W
rng = np.random.default_rng(2000); base = W.World(rng); B = W.make_bodies(rng, 3)
for eps, lr in ((0.2, 0.05), (0.2, 0.02), (0.1, 0.05)):
    w = base.copy(); ag = W.make_agents(np.random.default_rng(1), 1, 32)
    tr = []
    for c in range(80):
        W.step(w, B, ag, None, 500, seed=10 + c, lr=lr, eps=eps)
        if c % 8 == 7: tr.append(round(float(np.mean([W.competence(w, B, ag, 0, np.random.default_rng(99 + c + 1000*j)) for j in range(3)])), 3))
    print(f"eps={eps} lr={lr}: competence every 4k steps to 40k: {tr}")
