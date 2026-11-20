"""C1 tuning on seed 1000 only: learning rate for a lone width-32 agent in the natural world."""
import time, numpy as np, world as W
for lr in (0.02, 0.05, 0.1, 0.2):
    rng = np.random.default_rng(1000); w = W.World(rng); ag = W.make_agents(rng, 1, 32)
    t0 = time.time(); hit = None; trace = []
    for c in range(40):
        W.step(w, ag, None, 500, eta=0.0, lr=lr, seed=1000 + c)
        g = W.greedy_acceptance(w, ag, 0, np.random.default_rng(5000 + c)); trace.append(round(g, 3))
        if hit is None and g >= 0.95: hit = (c + 1) * 500
    print(f"lr={lr}: first 95% at {hit} steps; greedy acc every 2k: {trace[3::4]}; {time.time()-t0:.1f}s")
