"""Tuning step 2 (seed 3000, competence only): exploration settings at lambda = 0.4, noise = 0.01."""
import itertools, numpy as np, world3 as W
rng = np.random.default_rng(3000); nat = W.World(rng, noise=0.01); B = W.make_bodies(rng, [1, 1, 2, 2], 0.4)
for m, em, lr in itertools.product((0.1, 0.2), (0.3, 0.6), (0.05, 0.1)):
    ex = dict(eps_min=0.02, eps_m=em, m=m); costs = []
    for body in range(4):
        w = nat.copy(); ag = W.make_agents(np.random.default_rng(41 + body), 1, 32, body); c = None
        for s in range(250, 20001, 250):
            W.step(w, B, ag, None, 250, seed=6000 + body * 1000 + s // 250, lr=lr, explore=ex)
            if W.competence(w, B, ag, 0, np.random.default_rng(s)) >= 0.9:
                c = s; break
        costs.append(c if c is not None else 99999)
    print(f"m={m} eps_m={em} lr={lr}: steps to 0.9 by body {costs}, median {np.median(costs):.0f}", flush=True)
