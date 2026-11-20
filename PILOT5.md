# Pilot 5 report: the version 6 design (20 Nov 2026)

**What this is.** A small pilot of version 6's new E10 design, after Neon's fourth review. Three seeds (5000–5002), single networks of width 32 raised for 60,000 steps, nine agents per cell (three per seed).
- **Worlds:** the unequal-arc world (natural, wear, shaping by eye at κ = 1, and two weaker upkeep levels as a dose), and a symmetric-arc world with all arcs ±60° (natural and wear).
- **Feedback conditions:** the skip rule; passive feedback; and passive feedback with acceptances reweighted to a share s = 0.5 or 0.9.

Version 6's thresholds and this pilot's checks (`PILOT5_SPEC.md`) were committed to git before any pilot 5 code. Every tuning step and finding is in `PILOT5_LOG.md`. **None of this is evidence for or against a prediction.**

**In one paragraph.** The skip-rule result replicated a third time. Built-raised agents' boundaries sit 3.4–4.7° further from the truth than wild agents', their confident-error rate is 3–5 times higher, and home log loss gives no warning. Neon's two-pull account did not hold up, though the test had two flaws.
- **At s = 0.5 the label-balance pull didn't vanish:** it stayed close to its unweighted-passive value.
- **In the symmetric world the label balance flipped (0.26 against 0.93) and the shift didn't:** all arcs moved 2–3° inward under every condition.
- **The flaws:** reweighting at s = 0.9 used weights of 17–40 and broke the wild learners. At any s, reweighting moves the 0.5 crossing of agents that have edge data, so it contaminates the wild comparison.

What the data point to, untested, is the skip rule itself rather than the balance of labels. An agent that learns only from its own acts rarely collects the refusals that would pin its boundary from the far side, and building removes the edge items that would force them. Under passive feedback the displacement shrinks to 1.2–1.7°, while confident error stays 2–4 times wild agents'.

## Design checks

| Check | Result | Numbers |
| --- | --- | --- |
| P1: legibility on every seed (unequal world) | Pass | Wild edge accuracy 0.89–0.93; half-widths 60.2–61.1° and 35.3–36.3° |
| P2: the symmetric world is legible | Borderline | Edge accuracy 0.84–0.90 (one seed's narrow actions at 0.84); half-widths 59.7–61.7° |
| P3: reweighting hits its target share | Pass on the share; see D1 | s = 0.5: 0.493–0.509; s = 0.9: 0.896–0.905, per agent and action type |
| P4: a usable dose | Pass | Tuning seed: 0.7%, 8.3% and 22.5% edge items at s_u = 0.3, 0.1 and 0.03. Full pilot: 0.0–0.2%, 6.2–9.2% and 21.7–23.0% |
| P5: filtered sampler matched on what is offered | Pass | Redrawn every 50 steps: within 0.40 points of the wear world (1.03–1.60% offered against 1.27–1.96%) |
| P6: why wear erases so much (diagnosis) | Geometry | Wear at η = 0.15 leaves 23.7% edge items in the symmetric world against 1.2% in the unequal world. 30° overlaps hold items in a tug of war between two actions; 5° overlaps don't |
| P7: cost | Fine | 569 s for the whole pilot on 2 cores |

**Two design flaws, found in the results (D1, D2 in the log).**
- **D1:** the weight on acceptances is 17 (wide) and 40 (narrow) at s = 0.9, against 1.9 and 4.5 at s = 0.5. Steps that large destabilise SGD. Wild agents at s = 0.9 have home log loss 0.760, against 0.016–0.031 for wild agents in the other unequal-world conditions, so every s = 0.9 comparison is uninterpretable.
- **D2:** upweighting acceptances by w moves any well-trained agent's 0.5 crossing to where the true acceptance probability is 1/(1 + w). Wild agents have edge data, so their boundaries move with the weights: the wild narrow shift is +1.64° at s = 0.5 against −0.05° under unweighted passive feedback. Built minus wild then carries this artefact.

## The main finding: the skip rule, not the balance of labels, seems to set the sign

Built minus wild shift toward the 45° midpoint, unequal world, nine agents per cell. G = (W + N)/2 and L = (N − W)/2, as defined in L10s. Label shares are built-raised agents' (wide / narrow).

| Condition | Label share | Wear: W, N | Wear: G, L | κ = 1: W, N | κ = 1: G, L |
| --- | --- | --- | --- | --- | --- |
| Passive | 0.34 / 0.16 | +2.80, −0.67 | +1.06, −1.73 | +0.85, −2.21 | −0.68, −1.53 |
| Passive, s = 0.5 | 0.50 / 0.50 | +3.17, −1.05 | +1.06, −2.11 | +0.12, −2.01 | −0.94, −1.06 |
| Passive, s = 0.9 | 0.90 / 0.90 | broken (D1) | | broken (D1) | |
| Skip rule | 0.95 / 0.88 | −1.96, +5.40 | +1.72, +3.68 | −4.48, +5.47 | +0.49, +4.98 |

- **L flips between passive feedback and the skip rule,** as in pilot 4: about −1.5 to −1.7 against +3.7 to +5.0.
- **Balancing the labels didn't move L toward zero.** Going from shares of 0.34 / 0.16 to 0.50 / 0.50, L went from −1.73 to −2.11 (wear) and from −1.53 to −1.06 (κ = 1). Part of this is D2. The built-raised agents' own shifts at s = 0.5, before subtracting wild, are +2.6° (wide) and +0.6° (narrow) under wear, and −0.4° and −0.4° under κ = 1. That is the predicted pattern only weakly, under wear, and not at all under κ = 1.
- **G is small and its sign isn't stable:** +1.1 to +1.7 under wear, −0.9 to +0.5 under κ = 1.
- **The symmetric world contradicts the label-balance account most directly.** Built minus wild, every arc inward: +3.19° under the skip rule, +3.08° under passive feedback and +2.07° at s = 0.5. The label share runs from 0.26 (passive) to 0.93 (skip), and the shift doesn't follow it. Caveat: the symmetric wear world keeps 23.7% edge items, so it isn't an empty gap.

**A reading the data suggest, not tested.**
- **Under the skip rule,** an agent collects verdicts only where it acts, and it acts where it already expects acceptance. So once early learning is over, it rarely collects refusals of a wide action on the neighbouring narrow cluster. Under passive feedback those refusals arrive anyway, and they pin the boundary from the far side.
- **Reweighting** changes the balance but keeps those far-side refusals, which would explain why it didn't reproduce the skip-rule pattern at s = 0.5.
- **Building** removes the edge items that would otherwise force refusals near the boundary.
- **In the theory's terms,** the strong effect appears where the gate is the theory's own: verdicts arrive only for actions taken, and the agent chooses its actions from its model. Its model then shields itself from refutation, and a built world removes the cases that would have broken through.

## Readouts against version 6's thresholds

Three seeds. Reported for scale, not as verdicts.

| Prediction | Threshold | Readout |
| --- | --- | --- |
| L10a, skip rule | pooled shift ≥ 3°; confident error ≥ 1.5×; home log loss no higher | Wear 3.44° (seeds 3.2, 2.9, 4.3), κ = 1 4.73°. Confident error 0.126 and 0.181 against 0.037 (3.4× and 4.9×). Home log loss 0.016 and 0.014 against 0.031. **All three met** |
| L10a, passive | same | Pooled shift 1.65° and 1.22°: **under 3°**. Confident error 2.2× and 3.6×; home log loss 0.001 and 0.000 against 0.016 |
| L10a, s = 0.5 | same | Pooled shift 0.40° and 0.04°: **under 1°, "against" by the letter**. Confident error 0.90× and 1.76× |
| L10a, s = 0.9 | same | Uninterpretable (D1) |
| L10s: G > 0 in every condition | G > 0 | Wear: +1.06, +1.06, +1.72. κ = 1: −0.68, −0.94, +0.49. **Fails under κ = 1** |
| L10s: L rises with s across passive conditions | rising | Wear −1.73 → −2.11 (falls); κ = 1 −1.53 → −1.06 (rises); s = 0.9 unreadable. **Not met** |
| L10s: L < 0 passive, > 0 skip | signs | Met in both worlds |
| L10s: s = 0.5 decisive case | \|L\| ≤ 1 and both shifts positive | Wear L −2.11, shifts +3.17 and −1.05: both arcs shrink by ≥ 1°, **"against" by the letter**. κ = 1 L −1.06, shifts +0.12 and −2.01: not met |
| L10s: s = 0.9 matches skip | within 1.5° | Uninterpretable (D1) |
| L10-dose | pooled shift rises with erasure | Skip rule: 0.27°, 1.59°, 4.73° with 6, 21 and 28.5 points of edge items removed: **rises**. Passive: 0.66°, 1.69°, 1.22°: not monotone |
| Symmetric check | exploratory | Inward 2–3° under every condition; no dependence on the label share (above) |
| Agents needed | none | **Erratum (pilot 6):** the first estimate here, an agent-to-agent sd of about 1.7°, included the broken s = 0.9 arm. Without it the sd is about 0.6° for the pooled shift and 0.9° for G (printed at the end of verify5.py). Pilot 6 adds a world-to-world sd of about 0.4–0.7° |

## Design findings, with proposals for version 7

1. **Replace reweighting with biased offers.** Under passive feedback, draw each offered action so that it is accepted with probability s, instead of reweighting the update. That changes the label balance without changing any agent's conditional acceptance probability, so wild boundaries stay put (fixes D2). Every step keeps weight 1 (fixes D1).
2. **Separate the balance of labels from the agent's own selection.** Compare the skip rule with biased-offer passive feedback at the skip rule's own share, about 0.9. If they match, balance is enough. If they don't, it matters which verdicts the agent selects: refusals only near its own margin and none on the far side. A second probe: the skip rule with a higher random-act floor ε_min (0.02, 0.1, 0.3), which collects far-side refusals while keeping self-selection elsewhere.
3. **L10a under the skip rule (a theory-level decision for David).** The large displacement appears only when the agent learns from its own acts, which is the theory's gate. Proposed: the ≥ 3° pooled shift is required under the skip rule. The confidence and home-surprise conditions are required under every condition; they held under passive feedback too (2.2–3.6×). This change follows pilot data and should be read that way.
4. **L10s as stated isn't supported here.** G was small and unstable in sign, L didn't vanish when labels were balanced, and the symmetric world showed no dependence on the balance. Proposed: drop the two-pull decomposition as a preregistered prediction and keep G and L as descriptive readouts. Preregister instead the contrast in item 2, with its direction stated after Neon's review.
5. **L10-dose under the skip rule,** where it rose cleanly; report passive feedback alongside.
6. **Symmetric arm stays exploratory.** Wear keeps 23.7% edge items there, so it's a partial gap; say so beside every symmetric readout.
7. **At least 24 agents per kind per condition** for the E10 contrasts. (Erratum: this rested on the inflated 1.7° sd above; see pilot 6 for the corrected variance split.)

## Not covered

E2's slopes, E3, E4–E7, E8 (apart from P5), E9, ensembles, L10b and E_K.

## Files

- `PILOT5_SPEC.md`: checks, written before code. `PILOT5_LOG.md`: tests, tuning step 1, P6, D1–D2 and the reproducibility rerun.
- `pilot5/world5.py`: the version 6 world (numba): settable arcs, reweighted passive feedback, label records, confident error. `pilot5/test_world5.py`: thirteen sanity tests, including an exact match with pilot 4's world when nothing is reweighted.
- `pilot5/tune5_dose.py`: tuning step 1 and P6. `pilot5/pilot5.py`: the runner. `pilot5/verify5.py`: prints every figure above from `pilot5/results/`.
- To regenerate: `cd pilot5 && python3 test_world5.py && python3 tune5_dose.py && python3 pilot5.py && python3 verify5.py`.
