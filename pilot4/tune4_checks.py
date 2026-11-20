"""Design checks S4-S6 (seed 4000). S4: second-task bottleneck in wild agents. S5: randomized-prior manipulation check.
S6: does the built-minus-wild boundary shift survive passive feedback (no skip)?"""
import numpy as np, world4 as W
LAM, NOISE = 0.4, 0.01
rng = np.random.default_rng(4000); nat = W.World(rng, noise=NOISE); B = W.make_bodies(rng, [1, 1, 1, 1, 2, 2, 2, 2], LAM)
test = W.natural_test_set(np.random.default_rng(4077))
wb = nat.copy(); bl = W.make_agents(np.random.default_rng(2), 6, 32, body=0)
W.step(wb, B, bl, None, 30000, seed=70, rule=W.WEAR, eta=0.15, learner_shapes=True)
fr = W.copy_agents(bl); WEAR = dict(rule=W.WEAR, eta=0.15)

print("S4: second-task competence after 30k steps (wild, kind-B at 30% of offers)")
for body in (0, 4):
    res = {}
    for width in (8, 32, 128):
        w = nat.copy(); w.add_kind_b(np.random.default_rng(8), 300); ag = W.make_agents(np.random.default_rng(21 + width + body), 1, width, body)
        W.step(w, B, ag, None, 30000, seed=900 + width + body, pB=0.3)
        res[width] = round(float(np.mean([W.competence(w, B, ag, 0, np.random.default_rng(j), kind=1) for j in range(3)])), 3)
    print(f"  body {body}: {res}", flush=True)

print("S5: ensemble member spread in the wear world (60k), edges vs centres")
for name, beta in (("standard", 0.0), ("prior b=1", 1.0), ("prior b=3", 3.0), ("prior b=10", 10.0)):
    ens = W.make_ensemble(np.random.default_rng(31), 5, [32] * 5, body=0, beta=beta)
    w = wb.copy(); W.step_ens(w, B, ens, fr, 60000, seed=950, **WEAR)
    es = W.edge_scores_ens(B, ens, test, np.random.default_rng(5), noise=NOISE)
    print(f"  {name}: spread edges {es['spread_edges']:.3f}, centres {es['spread_centres']:.3f}, ratio {es['spread_edges']/max(es['spread_centres'],1e-9):.2f}", flush=True)

print("S6: boundary shift (toward 45) after 60k steps, skip rule vs passive feedback")
for passive in (False, True):
    out = {}
    for kind in ("wild", "built"):
        ag = W.make_agents(np.random.default_rng(41), 1, 32, 0)
        if kind == "wild":
            w = nat.copy(); W.step(w, B, ag, None, 60000, seed=970, passive=passive)
        else:
            w = wb.copy(); W.step(w, B, ag, fr, 60000, seed=971, passive=passive, **WEAR)
        bs = W.boundary_shift(B, ag, 0, np.random.default_rng(6), noise=NOISE)
        out[kind] = (round(bs["shift_wide"], 1), round(bs["shift_narrow"], 1))
    d = (round(out["built"][0] - out["wild"][0], 1), round(out["built"][1] - out["wild"][1], 1))
    print(f"  passive={passive}: wild (wide, narrow) {out['wild']}, built {out['built']}, built minus wild {d}", flush=True)
