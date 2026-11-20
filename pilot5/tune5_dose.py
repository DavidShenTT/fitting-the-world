"""Pilot 5 tuning step 1 and check P6 (seed 5000 only; built share and edge share only).

Step 1: shaping by eye at kappa = 1 with s_u in {0.3, 0.1, 0.03, 0.01}, in the unequal world.
P6: edge share left by wear (eta = 0.15) in the symmetric world against the unequal world.
"""
import numpy as np
import world5 as W

LAM, NOISE, LR, M, BW, WARM = 0.4, 0.01, 0.05, 6, 32, 30000


def build(nat, B, kw, sd):
    w = nat.copy()
    bl = W.make_agents(np.random.default_rng(sd), M, BW, 0)
    W.step(w, B, bl, None, WARM, seed=sd + 1, learner_shapes=True, lr=LR, **kw)
    return w


if __name__ == "__main__":
    for geom, name in ((W.UNEQUAL, "unequal"), (W.SYMMETRIC, "symmetric")):
        W.set_geometry(geom)
        rng = np.random.default_rng(5000); nat = W.World(rng, noise=NOISE); B = W.make_bodies(rng, [1, 1, 1, 1, 2, 2, 2, 2], LAM)
        print(f"{name}: natural edge share {nat.edge_share():.3f}")
        rules = [("wear 0.15", dict(rule=W.WEAR, eta=0.15))]
        if name == "unequal":
            rules += [(f"eye1 s_u {s}", dict(rule=W.EYE, s_u=s, kappa=1.0)) for s in (0.3, 0.1, 0.03, 0.01)]
        for k, (label, kw) in enumerate(rules):
            w = build(nat, B, kw, 900 + 10 * k)
            print(f"  {label:14s} built share {w.built_share():.3f}  edge share {w.edge_share():.3f}", flush=True)
