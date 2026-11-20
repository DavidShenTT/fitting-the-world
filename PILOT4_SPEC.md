# Pilot 4 spec (written 20 Nov 2026, before any pilot 4 code)

**What pilot 4 is for.** It checks version 5's redesign at small scale before CC spends time on it. Above all it checks whether unequal arcs produce boundary shifts of opposite sign. Like the earlier pilots, it is not evidence for or against any prediction. Seeds 4000 and up (pilots 1–3 used 1000–3002, real runs 0–999).

**Scale.** As pilot 3: 600 items, 16-dimensional observations, γ = 1.5, λ = 0.4, observation noise 0.01, one offer per turn with a skip, margin exploration (m = 0.2, ε_m = 0.6, ε_min = 0.02), learning rate 0.05. Builders M = 6, width 32, from random initialisation. Raised agents: widths 8, 32 and 128; budgets 30k and 60k. Four maps per body type. Three seeds.

**Tuning allowed, in this order, on seed 4000 only:**
1. **Legibility on wild agents only.** Keep λ = 0.4 and noise 0.01 if S1 passes; otherwise try noise 0.005, then λ = 0.25.
2. **Wear strength, on built share and edge share only,** so the built world keeps 5–10% edge items. Candidates η in {0.15, 0.25, 0.4}.
3. **β for randomized priors, by the manipulation check (S5) only.**
4. **ε_r on home log loss only,** in steps of 0.01.

Nothing is tuned on any boundary shift of built-raised agents, calibration, decisive payoff, speed-up, entropy or code length.

## Design checks, each pass or fail

- **S1, legibility:** wild width-32 agents after 60k steps reach edge accuracy ≥ 0.85 and a decisive ratio ≥ 0.6 on wide and on narrow actions.
- **S2, an unbiased start:** wild agents' learned half-widths are within ±3° of the true ones (60° and 35°).
- **S3, tight wear:** some η gives the wear world 5–10% edge items.
- **S4, the second task binds:** on wild agents trained on the main ring plus kind-B items (30% of offers) for 30k steps, width-8 second-task competence is at least 0.15 below width-128's. If not, L8c is dropped from the pilot.
- **S5, randomized priors disagree where there is no data:** for built-raised ensembles, member spread on unbuilt edge items is at least twice the spread at the centres, at some β in {1, 3, 10}. If not, the arm is dropped from the pilot.
- **S6, the shift isn't an artefact of skipping:** on one seed, wild and built-raised (wear) agents trained with passive feedback (every turn labelled, no skip) show built-minus-wild shifts in the same directions as agents under the skip rule.
- **S7, matched exposure:** the filtered control's edge share is within 2 points of the treatment world's, and their acted turns within 10%.
- **S8, cost:** wall-clock.

## Readouts reported but not judged

L3, L3b, L3c and L3d (passive), with same-type maps; L8a, L8a′, L8b, and L8c if S4 passes; L9 (relative) with the mechanism split by action type, and L9b; L10a (both shifts and the home-surprise condition), L10a-ens, and L10b at matched ε_r; the update-count control. Each against its version 5 threshold, labelled as a pilot readout. A design flaw found along the way is logged with a proposed fix; no threshold changes because of this pilot.

## Not in pilot 4

E2's slopes, E4–E7, E_K and the spillover arm.
