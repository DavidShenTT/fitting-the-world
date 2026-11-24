# Pilot 4 report: the version 5 design (20 Nov 2026)

**What this is.** A small pilot of version 5 of the brief, after Neon's third review. Three seeds (4000–4002), 600 items, unequal arcs (two wide at ±60°, two narrow at ±35°), structured bodies (four maps of each type), widths 8, 32 and 128, wear and shaping by eye at κ = 1 and 0.25, ensembles of 6. Version 5's thresholds and this pilot's checks (`PILOT4_SPEC.md`) were committed to git before any pilot 4 code. Every tuning step, diagnostic and change is in `PILOT4_LOG.md`, logged before it ran. **None of this is evidence for or against a prediction.**

**In one paragraph.** Unequal arcs did their job: they separate a shift toward the midpoint from generic caution or boldness, and they show that neither Neon's 45° mechanism nor any fixed-direction account fits. Under the brief's skip rule, built-raised agents stretch narrow arcs outward by about 6°, as predicted, but they stretch wide arcs too, where L10a predicted shrinking. Under passive feedback (every offer labelled) both arcs shrink instead. In this world the direction tracks how the agent's labels arrive: under the skip rule 84–95% of them are acceptances, under passive feedback 16–35% (D3). What holds in every condition is that built-raised boundaries sit further from the truth than wild agents', inside the window the built world emptied, and that home log loss gives no warning. The built world has no boundary there to carry over. The learner fills the gap with its own default, and nothing at home tells it that it did.

## Design checks

| Check | Result | Numbers |
| --- | --- | --- |
| S1: legibility (edge accuracy ≥ 0.85, decisive ratio ≥ 0.6, wide and narrow) | Pass on the tuning seed; borderline in the full pilot | Tuning seed, both maps: edge accuracy 0.87–0.94, decisive ratio 0.69–0.79. Full pilot, wild width 32 at 60k: wide edge accuracy 0.84 / 0.79 / 0.93 and narrow 0.83 / 0.82 / 0.89 by seed; decisive ratio 0.595 |
| S2: unbiased start (half-widths within ±3°) | Pass on the tuning seed; one seed outside | Tuning seed 62.0°/35.5° and 60.0°/36.2°. Full pilot: wide 58.9–61.9°, narrow 34.0–38.6° (seed 4001's narrow at 38.6°) |
| S3: wear leaves 5–10% edge items | **Fail** by the letter (D1) | Even the weakest candidate η = 0.15 leaves 1.0%, against 1.3% for κ = 1 and 27% natural. Kept η = 0.15 so wear removes edges about as thoroughly as κ = 1. Full pilot: 0.3–1.7% |
| S4: the second task binds | **Fail** | No width learns the 8-sector rule in 30k steps (competence −0.01 to 0.11). L8c dropped, as pre-stated |
| S5: randomized priors disagree where there's no data | Pass at β = 1 by the rule | Edge-to-centre spread ratio 2.32 (standard ensembles 1.96). Only β = 10 clearly differs (3.60) |
| S6: shift not an artefact of skipping | **Fail** | The direction depends on the feedback regime (next section) |
| S7: matched exposure (edge share within 2 points, acts within 10%) | Acts pass; exposure just outside | Acted turns, treatment / filtered 0.993. Edge items seen: filtered 3.4% against the wear world's 1.2% |
| S8: cost | Fine | About 240 s per seed; 485 s for the whole pilot on 2 cores |

## The main finding: the direction of the error follows the learner, not the world

Built minus wild boundary shift toward the 45° midpoint, in degrees. Positive means a wide arc shrinking or a narrow arc stretching; L10a predicted both positive.

| Condition | Skip rule (the brief's design) | Passive feedback |
| --- | --- | --- |
| Wear, D2b (seed 4000, 5 agents) | wide −3.0, narrow +3.9 | wide +3.4, narrow −0.7 |
| κ = 1, D2b | wide −5.6, narrow +3.9 | wide +3.5, narrow −2.5 |
| Wear, D3 (seed 4000, 3 other agents) | wide −2.8, narrow +4.1 | wide +1.6, narrow −0.7 |
| Wear, full pilot (3 seeds × 6 cells) | wide −0.7, narrow +6.1 | not run |
| κ = 1, full pilot | wide −4.2, narrow +6.0 | not run |

Under the skip rule both arcs stretch; under passive feedback both shrink. Neither regime gives the opposite-sign pattern of Neon's 45° account. Nor does either match interpolation to the middle of the emptied window, which D2a puts at about 52.5° from a wide centre (wide shrinking about 7.5°, narrow stretching about 2.5°).

**Why the direction flips (D3, consistent but not tested).** An agent under the skip rule learns from the acts it chose, and it acts where it expects acceptance, so most of its labels are acceptances: 92% / 84% (wide / narrow) for wild agents, 95% / 88% for built-raised ones. Under passive feedback it learns from every offer, and a random action accepts only a minority of items: 35% / 18% and 34% / 16%. The built world also leaves almost nothing inside the window: per 1,000 labels, built-raised agents get 31 (wide) and 5 (narrow) there under the skip rule, against 243 and 120 for wild agents. Where there are no labels, the network's answer drifts toward the balance of the labels it has: acceptance under the skip rule, refusal under passive feedback. Wild agents show the same lean, smaller, because edge items keep correcting it (wild narrow shift +2.3° under the skip rule in D2b). Building removes the correction, so the lean shows through.

**What this account doesn't explain.** Pilot 3 also used the skip rule, yet its diagnostic agent (wear, all arcs ±60°) put the boundary 2–3° *inside* the edge. Pilot 3's world differed: wear kept 21% edge items, and neighbouring arcs overlapped by 30°, so the edges lay deep inside another action's arc. One agent on one seed is thin evidence either way, but it means the label balance can't be the whole story. Version 6 should rerun pilot 3's symmetric world with the version 5 shift measure, under both regimes.

**What holds in every condition.**
- **Built-raised boundaries sit further from the truth.** Mean unsigned shift over all E10 single networks (exploratory): wild 1.5° (wide) and 2.1° (narrow); wear 2.3° and 7.4°; κ = 1 5.7° and 7.3°.
- **Home surprise gives no warning.** Built-raised agents' home log loss is 0.029 against wild agents' 0.049.
- **The calibration split goes the predicted way even where the boundary doesn't.** Underconfidence on accepted wide edges is 0.23 (wear) and 0.21 (κ = 1) against 0.17 (wild); false confidence on refused narrow edges is 0.61 and 0.58 against 0.29. The built-raised wide curves seem flatter rather than shifted inward.

**In the theory's terms.** The built world has no boundary inside the emptied window. It has a gap. The agent doesn't carry over the built partition; it fills the gap with its own default, set by how it learned, and takes the result for the world's boundary. That is the briefing's "the verdicts never arrived, so the prior decides", with the prior here being the label balance and the network's extrapolation. Naive realism, on this reading, is mistaking one's own fill-in for a feature of the world, with nothing at home to flag it as a fill-in. It predicts larger errors where building erased the cases. It predicts their sign only from the learner.

## Readouts against version 5's thresholds

Three seeds, skip rule. Reported for scale, not as verdicts.

| Prediction | Threshold | Readout |
| --- | --- | --- |
| L3: speed-up, same body vs outsider (κ = 0.25) | ≥ 1.5× | 2.60 vs 3.05: **0.85×**. Same-type newborns 3.74. Built-world costs vary by map (outsiders 2,083 vs 3,563 steps for the two builder maps). Under wear 0.70×; expected within 20% |
| L3b: log-loss drop at 2,000 steps | ≥ 1.5× | 0.086 vs 0.028: 3.03×; same type 0.036. Under wear 1.42× |
| L3c: forestry entropy drop | ≥ 0.5 bits for builders; outsiders ≤ half | Builders' body 0.524, same type 0.476, outsider 0.052. Under wear 0.25–0.28 for all three |
| L3d: passive code-length reduction over 8,000 pairs | ≥ 1.5× | 550.7 vs 397.2 bits: **1.39×**; same type 472.3. Under wear about 585 for all three |
| L8a: extra fall in edge accuracy vs natural control | ≥ 0.10 | Wear 0.109 / 0.084 / 0.058; κ = 1 0.117 / 0.062 / 0.053 (widths 8 / 32 / 128). Mostly the natural control improving (+0.06 to +0.15) |
| L8a′: treatment vs filtered control | expected within 0.03 | Wear 0.031 / 0.012 / 0.015; κ = 1 0.039 / −0.010 / 0.010. Close, though the filtered arm saw about three times the edge items (S7) |
| L8b: narrowest vs widest extra fall | ≥ 2× | Wear 1.89×, κ = 1 2.22× |
| L8c | conditional | Dropped (S4) |
| L9: decisive payoff after the snap | ≥ 30% below wild | Wear 0.202 vs 0.327: **38% lower**. κ = 1 0.072: **78% lower** |
| L9 mechanism | none | Narrow half holds in both worlds: refusals per act 0.272 (wear) and 0.328 (κ = 1) against 0.108. Wide half holds only under wear: act rate 0.409 against 0.461; under κ = 1 it is higher, 0.557, as wide arcs stretch there |
| L9b: log-loss jump at the snap | ≥ 0.1 nats | Wear 0.052; κ = 1 0.1004 |
| L10a: shifts toward the midpoint | ≥ 3° wide (inward) and ≥ 3° narrow (outward) | Wear: wide −0.71 (1 of 6 cells ≥ 3), narrow +6.13 (5 of 6). κ = 1: wide −4.24 (0 of 6), narrow +5.98 (6 of 6). **Fails on wide.** By the letter of "against", wear's wide shift is under 1° and κ = 1's shifts both stretch |
| L10a: home surprise | built no higher than wild | 0.029 vs 0.049: holds |
| L10a-ens | readings pre-stated | Standard: wear (−2.06, +5.54), κ = 1 (−3.67, +7.70). Diverse (−1.01, +9.15), (−3.13, +9.10). Randomized prior (−1.77, +5.33), (−3.27, +8.42). None of the three readings applies: every kind keeps the narrow shift, none shrinks wide arcs. Wild ensembles stretch both arcs too (wide −2.4 to −3.2, narrow +2.3 to +3.9) |
| L10b: matched home surprise | both shifts ≥ 3° in every bin | ε_r = 0.01 matches (home log loss 0.89× and 0.85× wild's). Built minus wild: wear (+0.71, +4.51), κ = 1 (−0.11, +5.51). Binning needs at least 30 agents per kind; the pilot has 3 |
| Update-count control | none | Wild agents given equal acted turns (56–58k steps): built minus matched wild is wear (−2.40, +5.54), κ = 1 (−4.41, +4.43). Equal updates don't remove the narrow shift |

## Design findings, with proposals for version 6

1. **What L10 should predict (a theory-level decision for David).** Every fixed-direction reading has broken. Pilots 1–2 looked like overconfidence, which was false refusal. Pilot 3 looked like caution, but its arcs were all equal, so caution and a shift toward the clusters couldn't be told apart. Pilot 4's shift toward the midpoint holds only for narrow arcs, and only under the skip rule. What has held every time is a larger error inside the emptied window and no warning from home surprise. **Proposed:**
   - **L10a (unsigned):** built-raised agents' boundaries deviate further from the truth than wild agents' raised under the same feedback, by at least 3° on each arc type, with home log loss no higher.
   - **L10s (sign from the learner):** the sign follows the label balance, not the geometry. Built-raised arcs stretch relative to wild where labels are mostly acceptances (skip rule) and shrink where they are mostly refusals (passive feedback).
   - **A test of L10s:** a third arm that changes the balance within one regime: passive feedback with acceptances weighted up to half of each action's labels. Predicted in advance: the shift moves from shrinking toward stretching. This is the first prediction that could separate "the learner's default decides" from "the world's geometry decides".
   - The brief's worry about unsigned measures was that generic caution or boldness would pass them. The wild agents in the same regime now control for that: whatever lean a regime has shows in them too. The prediction is only about what building adds.
2. **Run E10 in both feedback regimes.** It costs one extra arm per world and turns the pilot's surprise into a factor.
3. **Wear can't reach 5–10% edge items at the tested strengths.** Under unequal arcs both building rules empty the window almost completely. **Proposed:** a dose arm: wear at a lower η (for example 0.05) chosen to leave 5–10%, alongside η = 0.15, to test whether the error grows with how completely the cases were erased. That was Neon's own point about pilot 3's smaller effect.
4. **E3's speed-up doesn't separate readers; entropy does.** Steps-to-criterion depends heavily on the map (outsiders took 2,083 steps in one builder map's world and 3,563 in the other's). Entropy separates cleanly (0.52 and 0.48 vs 0.05 bits) and code length nearly does (1.39×). **Proposed:** report speed-up as area under the learning curve over a fixed window instead of steps to criterion, keep L3c as the primary reader-relative measure, and keep L3d. A primary change; David decides.
5. **Match E8's filtered control on edge items seen, not on home angles.** The filtered arm saw 3.4% edge items against the wear world's 1.2%, because drift carries natural items into the edges. Sample the filtered arm so that the share of edge items actually offered matches.
6. **Drop L8c.** No width learns the 8-sector rule in 30k steps, so there's no capacity bottleneck to free. A task that binds capacity without being unlearnable would need its own pilot, and nothing else depends on it.
7. **Randomized priors at β = 10, or drop the arm.** At β = 1 they're barely distinguishable from standard ensembles. Ensembles of every kind behaved like single networks.
8. **L9's mechanism per regime.** The wide half (fewer acts) failed under κ = 1 because wide arcs stretched there. If L10s is adopted, E9's mechanism should be stated per regime too.
9. **A symmetric-arc check arm** (pilot 3's world, version 5's shift measure, both regimes), for the open question above.
10. **Smaller fixes:** at least 30 agents per kind for L10b's bins; S1 and S2 checked on every pilot seed, not just the tuning seed.

## Not covered

E2's slopes, E4–E7, E_K and the spillover arm.

## Files

- `PILOT4_SPEC.md`: checks, written before code. `PILOT4_LOG.md`: the pilot 1–2 recheck, tuning steps, D1–D3 and the reproducibility rerun.
- `pilot4/world4.py`: the version 5 world (numba) with unequal arcs, passive feedback and the matched sampler. `pilot4/test_world4.py`: nine sanity tests.
- `pilot4/recheck_p12.py`: split calibration for pilots 1–2. `pilot4/tune4_legibility.py`, `tune4_build.py` and `tune4_checks.py`: tuning and checks S4–S6. `pilot4/diag_gap.py` (D2) and `pilot4/diag_labels.py` (D3): diagnostics.
- `pilot4/pilot4.py`: the runner. `pilot4/verify4.py`: prints every figure above from `pilot4/results/`.
- To regenerate: `cd pilot4 && python3 test_world4.py && python3 pilot4.py && python3 diag_gap.py > results/diag_gap.txt && python3 tune4_checks.py > results/tune4_checks.txt && python3 diag_labels.py > results/diag_labels.txt && python3 verify4.py`.
