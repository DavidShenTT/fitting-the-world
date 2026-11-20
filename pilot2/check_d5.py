"""D5 legibility check (seed 2000, wild width-32 agents only): appearance inputs scaled by lambda."""
import numpy as np, world2 as W
for lam in (1.0, 0.5, 0.25):
    rng = np.random.default_rng(2000); nat = W.World(rng); B = W.make_bodies(rng, 3)
    B["Win"][:, :, 2:4] *= lam
    test = W.natural_test_set(np.random.default_rng(2077))
    res = []
    for body in (0, 1, 2):
        ag = W.make_agents(np.random.default_rng(7 + body), 1, 32, body); w = nat.copy()
        W.step(w, B, ag, None, 30000, seed=300 + body, eps=0.2, lr=0.05)
        r = np.random.default_rng(5); es = W.edge_scores(B, ag, 0, test, r)
        res.append((round(W.competence(w, B, ag, 0, r), 3), round(es["edge_acc"], 3), round(es["decisive_ratio"], 3)))
    print(f"lambda={lam}: (competence, edge accuracy, decisive ratio) for bodies 0,1,2: {res}")
