"""Q4 re-sweep with eps = 0.2 (seed 2000): built share and edge share after 30,000 steps of 6 learning builders."""
import numpy as np, world2 as W
rng = np.random.default_rng(2000); base = W.World(rng); B = W.make_bodies(rng, 3)
for rule, settings in ((W.WEAR, [dict(eta=e) for e in (0.02, 0.08, 0.15)]),
                       (W.EYE, [dict(s_u=s, kappa=0.25) for s in (0.03, 0.1, 0.3)] + [dict(s_u=s, kappa=1.0) for s in (0.03, 0.1, 0.3)])):
    for kw in settings:
        w = base.copy(); bl = W.make_agents(np.random.default_rng(2), 6, 32)
        W.step(w, B, bl, None, 30000, seed=50, rule=rule, learner_shapes=True, eps=0.2, **kw)
        print(rule, kw, "built share %.3f, edge share %.3f (natural %.3f)" % (w.built_share(), w.edge_share(), base.edge_share()), flush=True)
