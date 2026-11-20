"""Check Neon proposed (round 3): were pilots 1-2's 'overconfidence' readouts false refusal, not false acceptance?
Pilots 1-2 scored an unsigned gap, max(p, 1-p) - accuracy, which counts a confident 'refused' on an accepted item as
overconfidence. Agents are re-raised with each pilot's own code and settings (width 32, full budget, wear world),
seeds as in each pilot, so they are not the identical agents of the original runs."""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))

def split(p_b, truth):
    correct = (p_b > 0.5) == truth
    return dict(unsigned_gap=float(np.maximum(p_b, 1 - p_b).mean() - correct.mean()),
                false_conf_refused=float(p_b[~truth].mean()), under_conf_accepted=float(1 - p_b[truth].mean()))

def pilot1():
    sys.path.insert(0, os.path.join(HERE, "..", "pilot")); import world as W1; import pilot as P1; sys.path.pop(0)
    rows = []
    for seed in (1000, 1001, 1002):
        sd = P1.Seeder(seed + 50)
        nat, built, builders, _, _ = P1.warm_builders(seed, W1.WEAR, P1.ETA, sd)
        frozen = W1.copy_agents(builders)
        test = W1.natural_test_set(np.random.default_rng(seed + 77))
        th = test["theta"]; e = W1.is_edge(th); ba = W1.boundary_action(th[e]); truth = W1.accepted_matrix(th[e])[np.arange(e.sum()), ba]
        for kind in ("wild", "built"):
            ag, _ = P1.raise_agent(kind, 32, 15000, nat, built, frozen, seed, sd)
            o = W1.observe(nat.body, th[e], test["nuis"][e], np.random.default_rng(1), nat.noise)
            rows.append(dict(pilot=1, seed=seed, kind=kind, **split(W1.predict(ag, 0, o)[np.arange(e.sum()), ba], truth)))
    return rows

def pilot2():
    sys.path.insert(0, os.path.join(HERE, "..", "pilot2")); import world2 as W2; import pilot2 as P2; sys.path.pop(0)
    rows = []
    for seed in (2000, 2001, 2002):
        sd = P2.Seeder(seed + 50); rng = np.random.default_rng(seed); nat = W2.World(rng); B = W2.make_bodies(rng, 3)
        built, fr = P2.build(nat, B, P2.RULES["wear"], sd)
        test = W2.natural_test_set(np.random.default_rng(seed + 77))
        th = test["alpha"]; e = W2.is_edge(th); ba = W2.boundary_action(th[e]); truth = W2.accepted_matrix(th[e])[np.arange(e.sum()), ba]
        for kind in ("wild", "built"):
            ag = P2.new_agent(seed * 97 + 32 + 30000, 32)
            if kind == "wild":
                w = nat.copy(); P2.live(w, B, ag, None, None, 30000, sd)
            else:
                w = built.copy(); P2.live(w, B, ag, fr, P2.RULES["wear"], 30000, sd)
            o = W2.observe(B, 0, th[e], test["u"][e], test["kind"][e], np.random.default_rng(1), 0.05)
            rows.append(dict(pilot=2, seed=seed, kind=kind, **split(W2.predict(ag, 0, o)[np.arange(e.sum()), ba], truth)))
    return rows

if __name__ == "__main__":
    import pandas as pd
    d = pd.DataFrame(pilot1() + pilot2())
    os.makedirs("results", exist_ok=True); d.to_csv("results/recheck_p12.csv", index=False)
    print(d.groupby(["pilot", "kind"])[["unsigned_gap", "false_conf_refused", "under_conf_accepted"]].mean().round(3).to_string())
