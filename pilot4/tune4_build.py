"""Tuning step 2 (seed 4000): wear strength for a 5-10% edge share; shaping by eye as in pilot 3."""
import numpy as np, world4 as W
rng = np.random.default_rng(4000); nat = W.World(rng, noise=0.01); B = W.make_bodies(rng, [1, 1, 1, 1, 2, 2, 2, 2], 0.4)
for rule, kw in [(W.WEAR, dict(eta=0.15)), (W.WEAR, dict(eta=0.25)), (W.WEAR, dict(eta=0.4)),
                 (W.EYE, dict(s_u=0.3, kappa=1.0)), (W.EYE, dict(s_u=0.3, kappa=0.25))]:
    w = nat.copy(); bl = W.make_agents(np.random.default_rng(2), 6, 32, body=0)
    W.step(w, B, bl, None, 30000, seed=70, rule=rule, learner_shapes=True, **kw)
    print(rule, kw, "built %.3f edge %.3f (natural %.3f) | u1 moved %.3f, u2 moved %.3f"
          % (w.built_share(), w.edge_share(), nat.edge_share(), w.u_move(0), w.u_move(1)), flush=True)
