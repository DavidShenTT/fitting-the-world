"""Seed 2000 only: learning rate for competence >= 0.9 (lone width-32 agent, natural world), and the
building regime sweep (built share only) for check Q4."""
import time, numpy as np, world2 as W
rng = np.random.default_rng(2000); base = W.World(rng); B = W.make_bodies(rng, 3)
for lr in (0.02, 0.05, 0.1):
    w = base.copy(); ag = W.make_agents(np.random.default_rng(1), 1, 32)
    hit = None; tr = []
    for c in range(30):
        W.step(w, B, ag, None, 500, seed=10 + c, lr=lr)
        cp = W.competence(w, B, ag, 0, np.random.default_rng(99 + c)); tr.append(round(cp, 3))
        if hit is None and cp >= 0.9: hit = (c + 1) * 500
    print(f"lr={lr}: competence 0.9 first at {hit}; every 2.5k: {tr[4::5]}")
t0 = time.time()
rows = []
for rule, settings in ((W.WEAR, [dict(eta=e) for e in (0.005, 0.02, 0.08)]),
                       (W.EYE, [dict(s_u=s, kappa=k) for k in (0.25, 1.0) for s in (0.01, 0.03, 0.1)])):
    for kw in settings:
        w = base.copy(); bl = W.make_agents(np.random.default_rng(2), 6, 32)
        W.step(w, B, bl, None, 30000, seed=50, rule=rule, learner_shapes=True, **kw)
        print(rule, kw, "built share %.3f, edge share %.3f (natural %.3f)" % (w.built_share(), w.edge_share(), base.edge_share()))
print("sweep %.0fs" % (time.time() - t0))
