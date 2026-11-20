# Pilot 6 report: the version 7 check (20 Nov 2026)

**What this is.** The small check Neon asked for before freezing version 7. Six seeds (6000–6005), unequal-arc world, wild, wear and shaping-by-eye (κ = 1) agents, single networks of width 32 raised for 60,000 steps, twelve agents per cell (two per seed). Five feedback conditions:
- the skip rule at ε_min = 0.02, 0.1 and 0.3;
- passive feedback;
- biased offers at s = 0.9.

Item histograms were also taken for the symmetric world. Version 7's thresholds and this pilot's checks (`PILOT6_SPEC.md`) were committed before any pilot 6 code. Nothing was tuned. **None of this is evidence for or against a prediction.**

**In one paragraph.**
- **The surprise Neon allowed for came up.** Biased offers move wild agents' boundaries outward by 4–5°, because the agent sees noisy observations and offer density near an edge then acts like a reweighting. So built-against-wild comparisons under biased offers are contaminated, and L10a fails there for wear.
- **L10a held in the other two main conditions.** Under the skip rule and under passive feedback, built-raised agents' confident error is 2.2–3.6 times wild agents', their unbuilt-edge log loss 1.8–2.6 times, and their home log loss lower.
- **L10-shift held** under the skip rule (3.1° and 4.6°).
- **L10m's ε_min prediction held:** the stretch falls from 3.6° to 1.8° to 0.2° under wear (5.1°, 2.5°, 0.7° under κ = 1) as random acts restore far-side refusals.
- **Across all five conditions,** built-raised agents' raw stretch follows the scarcity of far-side refusals exactly (Spearman 1.00 in both worlds).
- **Wild agents escape under the skip rule** because the rule makes them test at their own margin: 54% of their refusals come from just beyond the edge, against 16% under biased offers. Building empties that margin.

## Design checks

| Check | Result | Numbers |
| --- | --- | --- |
| Q1: biased offers hit their share | Pass | Overall 0.897–0.902 per agent; 0.928 wide and 0.851 narrow (predicted about 0.93 and 0.87) |
| Q2: biased offers barely move wild boundaries (≤ 1.5°) | **Fail** (D1) | Wild half-widths +4.59° (wide) and +3.98° (narrow) against passive feedback; every seed above 2°. Wild confident error 0.148 against 0.051, home log loss 0.052 against 0.019 |
| Q3: refusal sources differ as intended | Partly | Acceptances per far-side refusal, wild agents on wide actions: skip 27:1, biased 15:1, passive 0.6:1. ε_min lowers the skip rule's ratio to 5.8:1 and 2.0:1. Neon's arithmetic (50:1 and 47:1) gets the order right but the sizes about half; exploration near the margin adds refusals |
| Q4: symmetric histogram | Neon's density account fits | Symmetric wear world, % of items per 5° bin: 0.65 in [50°, 60°) against 7.87 in [60°, 90°); the whole overlap [30°, 60°) is nearly empty. So each boundary keeps items, and refusals, just outside it and almost none just inside |
| Q5: variance split | Worlds matter as much as agents | Agent-to-agent sd 0.53° (pooled shift) and 0.50° (stretch); world-to-world 0.71° and 0.61°. Pilot 5's data: 0.58° / 0.43° and 0.59° / 0.46°. **Pilot 5's earlier 1.7° estimate included its broken arm; corrected** |
| Q6: cost | Fine | 495 s for the whole pilot on 2 cores |

**D1, the design flaw.** The outcome given the true angle is unchanged under biased offers, but the agent sees noisy observations. Near an edge, an item just inside is offered the action about 27 times as often as one just outside. The agent's posterior given what it sees therefore shifts outward, as with a reweighting. The spec flagged this possibility before the run. **Proposed fix:** bias offers only for items more than 20° from every arc edge, and offer uniformly near edges. That starves far-side refusals without biasing the edge posterior.

## What pilot 6 shows about the mechanism

Raw mean stretch (how much wider than true the arcs are, averaged over arc types; not differenced from wild), against acceptances per far-side refusal on wide actions:

| Condition | Far-side ratio (wear) | Wear stretch | κ = 1 stretch | Wild stretch |
| --- | --- | --- | --- | --- |
| Passive | 0.5:1 | −1.38° | −1.26° | −0.20° |
| Skip, ε_min 0.3 | 1.8:1 | +0.15° | +0.66° | −0.08° |
| Skip, ε_min 0.1 | 5.4:1 | +2.05° | +2.77° | +0.24° |
| Biased offers | 14.8:1 | +3.02° | +5.10° | **+4.09°** |
| Skip, ε_min 0.02 | 25.2:1 | +4.27° | +5.77° | +0.68° |

- **Built-raised agents stretch as far-side refusals get scarcer,** in perfect rank order in both worlds (Spearman 1.00, exploratory). Under the skip rule and under biased offers alike, they stretch when the refusals that would pin the boundary from the far side are missing. This is Neon's refusal-starvation reading, and the ε_min sweep confirms it.
- **Wild agents break the pattern only under biased offers.** Under the skip rule they barely stretch (+0.68°), because the rule makes them test where they're unsure: 54% (wide) and 49% (narrow) of their refusals come from within 20° beyond the edge. Under biased offers the near share is 16% and 14%, and they stretch +4.09°.
- **So the built-against-wild gap under the skip rule has two parts.** Starved far-side refusals push every learner's arcs outward. Margin-targeted testing corrects that, but only where there are items at the margin to test, and building removes them. Neon's self-selection reading is the second half of this: the agent's own choice of where to test is what protects it, and a built world disables that protection.
- **The symmetric world fits the same picture from the other side.** There each boundary keeps refusals just outside it and loses acceptances just inside it (Q4), so every arc moves inward under every condition.

These readings are post hoc and labelled as such. The ε_min result was predicted.

## Readouts against version 7's thresholds

| Prediction | Threshold | Readout |
| --- | --- | --- |
| L10a, skip rule | confident error ≥ 1.5×; edge log loss ≥ 1.5×; home log loss no higher | Wear: 0.131 vs 0.057 (2.29×; seeds 1.14–4.48), 0.480 vs 0.260 (1.84×), 0.015 vs 0.031. κ = 1: 3.55× (seeds 2.37–5.87), 2.46×, 0.014 vs 0.031. **Met** |
| L10a, passive | same | Wear 2.16×, 1.91×, 0.001 vs 0.019. κ = 1 3.05×, 2.62×, 0.001 vs 0.019. **Met** |
| L10a, biased offers | same | Wear 0.88×, 1.03×. κ = 1 1.46×, 1.60×. Home log loss 0.004 vs 0.052. **Not met; the wild comparator moved (Q2)** |
| L10-shift, skip rule | pooled shift ≥ 3° | Wear 3.06° (seeds 2.0–4.4), κ = 1 4.57° (2.8–5.7). **Met**, wear narrowly |
| L10-dose, confident error (pilot 5 data) | rises with erasure | 1.32×, 1.94×, 4.86× with 6, 21 and 28.5 points of edge items removed. **Rises** |
| L10m: stretch falls as ε_min rises | 0.02 > 0.1 > 0.3 | Wear +3.58 > +1.81 > +0.22; κ = 1 +5.09 > +2.53 > +0.74. **Met** |
| L10m: biased offers' position, passive (0) to skip (1) | pre-stated readings | Wear 0.02 ("self-selection"), κ = 1 0.34 ("between"). Contaminated by Q2: wild agents moved, and built agents' raw stretch under biased offers is close to the skip rule's |

## Proposals: what's left before freezing

1. **Biased offers: fix or drop.**
   - **Fix:** bias offers only beyond 20° from every edge, then check Q2 again. That's a ten-minute pilot.
   - **Drop:** remove biased offers from L10a's main conditions, so L10a holds under the skip rule and passive feedback, both met here. L10m's preregistered prediction is then the ε_min sweep, with the fixed biased offers as an exploratory arm.

   I'd lean to dropping them and freezing. Both main conditions and the ε_min prediction are in hand, and the mechanism no longer depends on biased offers.
2. **Report raw stretch beside built minus wild in L10m,** with the near-side and far-side refusal shares, since a condition can move the comparator.
3. **Variance.** At least 8 seeds × 3 agents per kind per condition is ample: a built-minus-wild standard error of 0.38° for the pooled shift and 0.34° for the stretch (end of verify6.py). Per-world confident-error ratios vary widely (1.1–4.5 under wear), so the criterion should be judged on the mean over worlds, as written.
4. **Decisions for David before preregistration:**
   - L10's hierarchy (confident error and silent home surprise primary; the shift secondary, under the skip rule);
   - L8a scoped to the narrowest width;
   - what to do with biased offers;
   - handing the brief to CC to write PLAN.md and PREREGISTRATION.md.

## Files

- `PILOT6_SPEC.md`: checks, written before code. `PILOT6_LOG.md`: tests, D1, the correction to pilot 5, the reproducibility rerun.
- `pilot6/world6.py`: the version 7 world (biased offers, angle records, refusal sources, unbuilt-edge log loss, histograms). `pilot6/test_world6.py`: sixteen tests, including an exact match with pilot 5's world when offers are unbiased.
- `pilot6/pilot6.py`: the runner. `pilot6/verify6.py`: prints every figure above from `pilot6/results/` and pilot 5's saved data.
- To regenerate: `cd pilot6 && python3 test_world6.py && python3 pilot6.py && python3 verify6.py`.
