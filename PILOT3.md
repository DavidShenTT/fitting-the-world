# Pilot 3 report: the version 4 design (20 Nov 2026)

**What this is.** A small pilot of version 4 of the brief, after Neon's second review. Three seeds (3000–3002), 600 items, structured bodies (two maps of each type), widths 8, 32 and 128, wear and shaping by eye at κ = 1 and 0.25, ensembles of 5. Version 4's thresholds and this pilot's checks (`PILOT3_SPEC.md`) were committed to git before any pilot 3 code. Every tuning step and change is in `PILOT3_LOG.md`, logged before it ran. **None of this is evidence for or against a prediction.**

**In one paragraph.** The redesign works as a design: edges are now legible, agents reach competence fast, and E3's entropy measure cleanly separates readers of the shaped axis from outsiders. But the central E10 prediction came out reversed. Built-raised agents are not confident and wrong at unbuilt edges. They are cautious: they predict that the arcs end earlier than they do, as narrow as the built world's clusters, so they skip more and miss payoffs. Pilots 1 and 2's "overconfidence" appears to have come from illegible edges and noisy exploration. That changes what the naive-realism prediction should say, which is the main question for the next review.

## Design checks

| Check | Result | Numbers |
| --- | --- | --- |
| R1: legibility (edge accuracy ≥ 0.85, decisive ratio ≥ 0.6, appearance slows learning ≥ 2×) | **Partial** | Reached only at λ = 0.4 with observation noise 0.01, and only after 40–60k steps (D1, D2). At 60k: edge accuracy 0.84–0.88 and decisive ratio 0.56–0.67 over seeds. Slow-down 1.5–1.6×, below 2× |
| R2: competence 0.9 within 20k steps | Pass | Median 3,375 steps at the chosen exploration (m = 0.2, ε_m = 0.6), against a plateau near 0.85 in pilot 2 |
| R3: balanced difficulty between body types | Pass | Mean natural cost 2,708 (type 1) vs 3,458 (type 2) steps, 1.28×. Individual maps range from 1,500 to 6,750 |
| R4: shaping moves only what the builder reads | Pass in substance | At κ = 0.25 the builders' axis moved 0.63–0.78 from home; the unread axis 0.046–0.051, which is the natural drift band. The 0.05 threshold was set at the band itself |
| R5: acted turns, treatment vs filtered within 10% | Borderline | Treatment had 0.90 (wear) and 0.89 (κ = 1) of the filtered arm's acted turns |
| R6: randomized-prior members disagree more at edges | Weak | Member spread on built-raised agents: 0.104 vs 0.080 under wear, 0.085 vs 0.083 under κ = 1. At β = 1 the priors change little |
| R7: cost | Fine | About 245 s per seed on 2 cores |

## The main finding: built-raised agents narrow the categories

Signed distance from the arc edge against predicted acceptance of the boundary action, for one wild and one built-raised (wear) agent on seed 3000 (`pilot3/diag_edge_curve.py`):

| Distance from the arc edge | Wild | Built-raised |
| --- | --- | --- |
| 10–15° inside | 0.987 | 0.886 |
| 5–10° inside | 0.913 | 0.712 |
| 0–5° inside | 0.700 | 0.480 |
| 0–5° outside | 0.426 | 0.300 |
| 5–10° outside | 0.193 | 0.167 |

The built-raised agent's acceptance curve sits several degrees inside the true edge. The built world pulled items toward the centres, so the outer parts of each arc were rarely seen accepted, and the agent learned arcs as narrow as the clusters it saw. It isn't overconfident: on refused edges its false confidence is about the same as a wild agent's. It is underconfident where acting would have paid. In the theory's terms the agent still takes the built world's carving for the world's, but that shows up as timid blindness, not reckless error. Where the gate gave no verdicts, the network's own extrapolation decided, and it decided "narrow".

## Readouts against version 4's thresholds

Three seeds. Reported for scale, not as verdicts.

| Prediction | Threshold | Readout |
| --- | --- | --- |
| L3: speed-up, same body vs outsider (eye, κ = 0.25) | ≥ 1.5× | 3.29 vs 1.92: 1.71×. Same-type newborns 1.71, no better than outsiders. Under wear 3.53 vs 4.32 (0.82×, within 20% as expected) |
| L3b: log-loss drop at 2,000 steps | ≥ 1.5× | 0.022 vs 0.007 (3.3×); same type 0.011. Small absolute values |
| L3c: forestry entropy drop | ≥ 0.5 bits for builders; outsiders ≤ half | Builders' body 0.48, same type 0.52, outsider 0.035. Under wear 0.23–0.27 for all three |
| L3d: code-length reduction | ≥ 1.5× | 0.007, 0.013 and −0.000 nats per step: too small to read at 2,000 steps |
| L8a: extra fall in edge accuracy vs natural control | ≥ 0.10 | Wear 0.046 / 0.021 / 0.034; κ = 1 0.092 / 0.026 / 0.042 (widths 8 / 32 / 128). Mostly the natural control improving |
| L8a′: treatment vs filtered control | expected within 0.03 | κ = 1: 0.001 / −0.047 / 0.000. Wear: −0.044 / −0.052 / −0.008. The filtered control forgets *more* than wear, because the wear world still has 21% edge items and the filtered arm has none |
| L8b: narrowest vs widest extra fall | ≥ 2× | 1.36× |
| L8c: second-task curve area, treatment vs natural (width 8) | ≥ 10% larger | 1.4% (wear), 1.5% (κ = 1). Linear probe 0.97–0.995 in every arm: the second task is too easy for capacity to bind |
| L9: decisive-payoff gap after snap-back | ≥ 0.3 | 0.075 (wear), 0.138 (κ = 1) |
| L9 mechanism | none | Built-raised agents act less: act rate 0.28 and 0.25 against 0.42 for wild agents. Refusals per act are 0.049 (wear) and 0.122 (κ = 1) against 0.096. Mostly cautious skipping, not confident error |
| L9b: log-loss jump at the snap | ≥ 0.1 nats | 0.029 (wear), 0.064 (κ = 1) |
| L10a: false confidence on refused edges, built minus wild | ≥ 0.10 | −0.040 (wear), +0.012 (κ = 1); no width × budget cell clears it. Underconfidence on accepted edges is +0.16 and +0.19. **Reversed relative to pilots 1–2** |
| L10a-ens: the same, for ensembles | (readings pre-stated) | Standard −0.127 / −0.066, diverse −0.112 / −0.067, randomized prior −0.118 / −0.066 (wear / κ = 1). All three kinds show the same caution |
| L10b: edge accuracy at matched home surprise | ≥ 0.10 lower | Wear already matches at ε_r = 0 (home log loss 0.031 vs 0.038): deficit 0.016. κ = 1 needs ε_r between 0 and 0.05; the grid was too coarse to match |
| Update-count check | none | A wild agent given the same number of acted turns as a built-raised agent (about 17k, reached at 48k steps) has edge accuracy 0.813, against 0.843 for the built-raised agent under wear and 0.771 under κ = 1. The small wear deficit disappears at equal updates |

## Design findings, with proposed fixes for version 5

1. **What L10 should predict.** "Confident and wrong" (false confidence on refused edges) failed in every arm, including all three ensembles. What did appear is a shift of the learned boundary toward the built world's clusters. **Proposed:** restate L10 as a signed **boundary shift**: where the agent's predicted acceptance crosses 0.5, compared with the true edge, in degrees. The predicted direction is fixed by what building removed (here inward), and home log loss should give no warning. Keep false confidence and underconfidence as secondary readouts. This is a change of prediction, not just of measure, so it needs Neon's view and David's decision.
2. **E9's threshold is too strict for this payoff scale.** Wild agents earn about 0.3 per decisive turn, so a 0.3 gap requires built-raised agents to earn nothing. **Proposed:** a gap of at least 30% of the wild payoff, with the mechanism (act rate, refusals per act) reported.
3. **Compare at equal updates.** The wear deficits in E10 vanish when wild agents get equal acted turns. **Proposed:** equal acted turns as the primary comparison in E8–E10, equal steps as secondary.
4. **Match the filtered control to the built world's edge exposure.** Remove edge items at the rate the built world does, not entirely. Otherwise E8's control is a stronger shift than wear.
5. **L3d needs a better window.** Code length over the first 2,000 steps is dominated by early-learning noise. **Proposed:** online log loss on a frozen built state over 10,000 steps, or on held-out items at fixed checkpoints.
6. **E3 has an interesting split worth keeping.** Entropy says the key is the axis read (same-type newborns see the compression as clearly as same-body ones: 0.52 vs 0.48 bits). Learning speed says the key is the exact map (same-type 1.71, same body 3.29). This could be noise across only two maps per type. **Proposed:** at least four maps per type, and report built-world cost directly alongside the ratio.
7. **The second task is too easy.** It's at ceiling for every width. **Proposed:** a harder second task (narrower arcs, or a nonlinear rule on appearance), or drop L8c.
8. **Smaller fixes:**
   - ε_r grid in steps of 0.01 for κ = 1;
   - match acted turns in E8's treatment and filtered arms;
   - β = 3 for the randomized-prior arm if that contrast matters;
   - drop R1's slow-down criterion, since E3 separated without it.

## Not covered

E2's slopes, E4–E7, E_K and the spillover arm.

## Files

- `PILOT3_SPEC.md`: checks, written before code. `PILOT3_LOG.md`: tuning steps 1–3, D1–D2 and the diagnostic.
- `pilot3/world3.py`: the version 4 world (numba), with ensembles. `pilot3/test_world3.py`: eleven sanity tests.
- `pilot3/tune3_lambda.py`, `tune3_noise.py`, `tune3_explore.py` and `tune3_build.py`: tuning. `pilot3/pilot3.py`: the runner. `pilot3/verify3.py`: prints every figure above from `pilot3/results/*.csv`. `pilot3/diag_edge_curve.py`: the edge-curve table.
- To regenerate: `cd pilot3 && python3 test_world3.py && python3 pilot3.py && python3 verify3.py && python3 diag_edge_curve.py`.
