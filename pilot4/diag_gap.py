"""D2 (seed 4000): (a) where the built worlds are empty; (b) S6 with five agents per condition."""
import numpy as np, world4 as W
LAM, NOISE = 0.4, 0.01
if __name__ == "__main__":
    rng = np.random.default_rng(4000); nat = W.World(rng, noise=NOISE); B = W.make_bodies(rng, [1, 1, 1, 1, 2, 2, 2, 2], LAM)
    worlds = {}
    for name, kw in (("wear", dict(rule=W.WEAR, eta=0.15)), ("eye1", dict(rule=W.EYE, s_u=0.3, kappa=1.0))):
        w = nat.copy(); bl = W.make_agents(np.random.default_rng(2), 6, 32, body=0)
        W.step(w, B, bl, None, 30000, seed=70, learner_shapes=True, **kw)
        worlds[name] = (w, W.copy_agents(bl), kw)
    print("(a) share of items per 5-degree bin of the angle within each quarter (0 = a wide centre, 90 = a narrow centre)")
    bins = np.arange(0, 95, 5)
    for name, w in (("natural", nat),) + tuple((k, v[0]) for k, v in worlds.items()):
        a = w.alpha[w.kind == 0] % 180.0          # fold: 0-90 runs from a wide centre to a narrow centre
        a = np.where(a > 90, 180 - a, a)
        h, _ = np.histogram(a, bins=bins); h = h / h.sum()
        print(f"  {name:8s}", " ".join(f"{x:.3f}" for x in h))
    print("  bins:   ", " ".join(f"{b:>5d}" for b in bins[:-1]))
    print("(b) mean boundary shift toward 45 over five agents (wide, narrow) [sd]")
    for passive in (False, True):
        for kind in ("wild", "wear", "eye1"):
            sh = []
            for k in range(5):
                ag = W.make_agents(np.random.default_rng(100 + k), 1, 32, 0)
                if kind == "wild":
                    w = nat.copy(); W.step(w, B, ag, None, 60000, seed=980 + k, passive=passive)
                else:
                    w0, fr, kw = worlds[kind]; w = w0.copy(); W.step(w, B, ag, fr, 60000, seed=990 + k, passive=passive, **kw)
                bs = W.boundary_shift(B, ag, 0, np.random.default_rng(6), noise=NOISE)
                sh.append((bs["shift_wide"], bs["shift_narrow"]))
            sh = np.array(sh)
            print(f"  passive={passive!s:5s} {kind:5s}: wide {sh[:,0].mean():5.1f} [{sh[:,0].std():.1f}]  narrow {sh[:,1].mean():5.1f} [{sh[:,1].std():.1f}]", flush=True)
