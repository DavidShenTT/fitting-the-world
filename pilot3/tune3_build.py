"""Tuning step 3 (seed 3000): building strengths; built share, edge share and appearance movement only."""
import numpy as np, world3 as W
rng = np.random.default_rng(3000); nat = W.World(rng, noise=0.01); B = W.make_bodies(rng, [1, 2], 0.4)
for rule, kw in [(W.WEAR, dict(eta=0.05)), (W.WEAR, dict(eta=0.15)),
                 (W.EYE, dict(s_u=0.1, kappa=1.0)), (W.EYE, dict(s_u=0.3, kappa=1.0)),
                 (W.EYE, dict(s_u=0.1, kappa=0.25)), (W.EYE, dict(s_u=0.3, kappa=0.25))]:
    w = nat.copy(); bl = W.make_agents(np.random.default_rng(2), 6, 32, body=0); grads = []
    for c in range(6):
        W.step(w, B, bl, None, 5000, seed=70 + c, rule=rule, learner_shapes=True, **kw)
        grads.append(round(W.appearance_gradient(B, bl, w, np.random.default_rng(c)), 4))
    print(rule, kw, "built %.3f edge %.3f (natural %.3f) | u1 moved %.3f, u2 moved %.3f | grad %s"
          % (w.built_share(), w.edge_share(), nat.edge_share(), w.u_move(0), w.u_move(1), grads), flush=True)
