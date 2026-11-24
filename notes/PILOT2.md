# Pilot 2 report: the version 3 design (20 Nov 2026)

**What this is.** A small pilot of version 3 of the brief, after Neon's review of pilot 1. Three seeds (2000–2002), 600 items, widths 8, 32 and 128, wear and shaping by eye at κ = 1 and 0.25. Version 3's thresholds and this pilot's checks (`PILOT2_SPEC.md`) were committed to git before any pilot 2 code. Every change along the way is in `PILOT2_LOG.md`, logged before the runs it affected. **None of this is evidence for or against a prediction.**

**In one paragraph.** Neon's skip-with-penalty fix works as intended against the old shortcut: the nearest-centre rule now earns nothing on decisive turns. But it exposed a deeper flaw: no agent reads the arc edges well enough for edge knowledge to matter, wild-raised or built-raised. So the behavioural tests (fragility payoffs, forgetting) sit at a floor. The tests that read the agents' predictions still point the predicted way, and the ensemble control strengthens rather than weakens E10. E3's "key" test doesn't yet separate builders from outsiders, because random bodies all see every coordinate. Each problem has a proposed fix below, to settle with Neon before version 4.

## Design checks

| Check | Result | Numbers |
| --- | --- | --- |
| Q1: nearest-centre heuristic ≤ 0.8 of optimal on decisive edge turns | Pass | 0.0 on every seed: a boundary action is never the nearest centre, so the heuristic always skips |
| Q1: wild-raised width-32 agents ≥ 0.6 of optimal there | **Fail** | −0.11 to 0.18 at λ = 1. Agents are rarely confident enough near an edge to act |
| Q2: different-body newborns reach the competence criterion in the natural world | Pass, at a lowered criterion | 2,000–4,000 steps to competence 0.75 (D4: 0.9 wasn't reachable in time) |
| Q3: some random-refusal rate matches home surprise within ±20% | Pass | Wild 0.126. Wear at ε_r = 0.05: 0.114. Shaping by eye at κ = 1 with ε_r = 0.10: 0.133 |
| Q4: unsaturated building exists under every rule | Pass | Built share 0.63 (wear), 0.58 (eye, κ = 1), 0.47 (eye, κ = 0.25) on the sweep |
| Q5: the second task is learnable, not instant | **Fail** | Steps to criterion range from 1,000 to 17,000 or never; too noisy to read |
| Q6: cost | Fine | About 160 s per seed on 2 cores |

## Design findings, with proposed fixes for version 4

1. **The edges are too hard to read (D5), the main finding.** Wild-raised agents' edge accuracy is 0.64–0.68, where chance is 0.5; pilot 1 showed the same at 0.71. When nobody knows the edges well, built-raised agents can't be much worse, and E8 has little to forget.
   - Check on wild agents only: scaling every body's appearance inputs by λ = 0.25 raises edge accuracy to 0.73–0.88 and the decisive payoff to 0.39–0.66 of optimal.
   - **Proposed:** make the angle legible (λ, observation noise or width), chosen on wild agents only, until edge accuracy is at least 0.85 and the decisive ratio at least 0.6.
2. **The skip rule slows learning and building (D3, D4).** An unsure agent skips, and learns only from exploratory acts. ε = 0.2 helps, but competence still levels off near 0.85 after 40,000 steps. **Proposed:** keep ε = 0.2, and set the competence criterion and budgets after fix 1, since legibility will change both.
3. **E3's learning-cost measure hit a floor and a confound.**
   - Every built-world cost sits at the first check (2,000 steps).
   - The builders' body happened to be the hardest body in the natural world: 5,000–9,500 steps against 2,000–4,000 for the outsiders. So same-body "speed-ups" (3.8× against 1.3× and 1.6×) mostly reflect that baseline.
   - **Proposed:** check competence every 250 steps from step 0, and rotate which body builds across worlds so that body difficulty averages out.
4. **Random bodies have no distinct "key".** Every random body reads every coordinate, so when shaping by eye moves an item's appearance, outsiders see the move too. The forestry drop is about the same for builders and outsiders: 0.30 against 0.31 and 0.22 bits at κ = 0.25; 0.62 against 0.64 and 0.41 at κ = 1. Under wear it is near zero for all three, because crisping the angle barely changes the 2-D projection. **Proposed:** structured bodies, Neon's subspace mismatch: builders read the angle and one appearance axis, outsiders the angle and the other. Also measure entropy under each body's learned carving (for example, clusters of a trained newborn's hidden layer), not under fixed PCA bins.
5. **Shaping by eye at κ = 0.25 builds weakly.** Built share is 0.35–0.44, and edge share 0.30–0.35 against 0.44–0.48 in the natural world. So E8–E10 under that rule can only show small effects. **Proposed:** run E8–E10 under wear and κ = 1, and keep κ = 0.25 for E3, where moving appearance is the point.
6. **Decisive payoff is at a floor; the log-loss jump works.** With unreadable edges, everybody skips decisive turns, so payoffs are near zero for both kinds of agent. **Proposed:** keep both measures; fix 1 should lift the payoff measure off the floor.
7. **The second task's criterion is noisy.** **Proposed:** score the area under its learning curve, not steps to a threshold.
8. **Version 3's claim about random refusal was wrong here.** The brief said noisy feedback should make networks less overconfident and so bias L10b against the prediction. In pilot 2 the calibration gap grew with ε_r instead: under wear, 0.209 / 0.204 / 0.242 / 0.257 at ε_r = 0 / 0.05 / 0.1 / 0.2.

## Readouts against version 3's thresholds

Three seeds, λ = 1 throughout, so all are affected by finding 1. Reported for scale, not as verdicts.

| Prediction | Threshold | Readout |
| --- | --- | --- |
| L3: same-body speed-up vs outsiders' (eye, κ = 0.25) | ≥ 1.5× | 3.8× against 1.3× and 1.6×. Not interpretable (finding 3); wear shows the same pattern, where the brief expects none |
| L3b: log-loss drop at 2,000 steps (eye, κ = 0.25) | ≥ 1.5× | 0.100 against 0.041 and 0.044. Same confound |
| L3c: forestry drop, builders vs outsiders | ≥ 0.5 bits; outsiders ≤ half | 0.30 against 0.31 and 0.22. Against (finding 4) |
| L8a: extra fall, treatment vs natural control (wear) | ≥ 0.10 | 0.036 / 0.022 / 0.021 at widths 8 / 32 / 128. Mostly the control improving; little was known to forget |
| L8a′: treatment vs filtered control (wear) | expected within 0.03 | −0.002 / 0.014 / 0.010: as expected |
| L8b: narrowest vs widest extra fall | ≥ 2× | 1.7×, on very small values |
| L8c: second task, treatment vs natural control | ≤ 0.8× the steps | Not readable (Q5) |
| L9: decisive-payoff gap after snap-back | ≥ 0.3 | 0.011 (wear), 0.054 (eye κ = 1), 0.047 (κ = 0.25). At a floor (finding 6) |
| L9b: log-loss jump at the snap, built minus wild | ≥ 0.1 nats | 0.109 (wear), 0.196 (κ = 1), 0.125 (κ = 0.25): clears it under all three rules |
| L10a: calibration gap, built minus wild (single networks) | ≥ 0.10 | Mean 0.056 (wear), 0.084 (κ = 1), 0.046 (κ = 0.25). 1, 1 and 0 of 6 width × budget cells clear it |
| L10a-ens: the same with ensembles | ≥ 0.10 means "not specific to single networks" | 0.137 (wear), 0.143 (κ = 1). Wild ensembles become well calibrated (gap 0.036); built ensembles stay overconfident (0.17–0.18) |
| L10b: edge accuracy at matched home surprise | ≥ 0.10 lower | Means only (three seeds can't fill bins): 0.077 lower under wear at ε_r = 0.05; 0.110 lower at κ = 1 with ε_r = 0.10 |

**Erratum (added after pilot 3).** The calibration gap here is unsigned. Re-raised agents (`pilot4/recheck_p12.py`) show that the built-raised agents' larger gap came from false *refusal* on accepted edge items, not from false acceptance. The ensemble result reads the same way.

**The pattern behind the readouts.** Everything that reads an agent's predictions points the way the naive-realism mechanism predicts, even with unreadable edges:
- the log-loss jump at the snap;
- calibration, especially for ensembles;
- edge accuracy at matched home surprise.

Everything that needs edge knowledge to change behaviour is at a floor: decisive payoffs, forgetting and the second task. The ensemble result is the most informative. Averaging five networks cured the wild-raised agents' overconfidence, but not the built-raised agents', whose networks all extrapolate the same way. On three seeds this is a reason to fix the design, not a result.

## Not covered

E2's slopes, E4–E7, E_K, structured bodies, the accidental co-option idea and λ < 1 for anything but the D5 check.

## Files

- `PILOT2_SPEC.md`: checks, written before code. `PILOT2_LOG.md`: D3–D5 and the test set-up fixes.
- `pilot2/world2.py`: the version 3 world (numba). `pilot2/test_world2.py`: ten sanity tests. `pilot2/tune2.py`, `tune2b.py`, `tune2_long.py` and `tune2_resweep.py`: the tuning and regime sweeps. `pilot2/check_d5.py`: the legibility check. `pilot2/pilot2.py`: the runner. `pilot2/verify2.py`: prints every figure above from `pilot2/results/*.csv`.
- To regenerate: `cd pilot2 && python3 test_world2.py && python3 pilot2.py && python3 verify2.py`.
