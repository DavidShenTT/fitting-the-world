# Pilot 3 spec (written 20 Nov 2026, before any pilot 3 code)

**What pilot 3 is for.** It checks version 4's redesign at small scale, before CC spends time on it. Like the earlier pilots, it is not evidence for or against any prediction. Seeds 3000 and up (pilot 1 used 1000–1002, pilot 2 2000–2002, real runs 0–999).

**Scale.** 600 items, 16-dimensional observations, γ = 1.5, observation noise 0.05. One offer per turn with a skip (+1 / 0 / −2). Builders: M = 6, width 32, from random initialisation. Raised agents: widths 8, 32 and 128. Three seeds. Ensembles of 5.

**Tuning allowed, in this order, on seed 3000 only:**
1. **Legibility λ, on wild agents only.** Candidates 0.25, 0.4, 0.6 and 1.0, for type-1 and type-2 bodies (two maps each). Pick the largest λ that meets R1.
2. **Exploration (m, ε_m, learning rate), on competence only.** Candidates m in {0.1, 0.2}, ε_m in {0.3, 0.6}, learning rate in {0.05, 0.1}.
3. **Building strengths, on built share and appearance movement only:** wear η, and shaping-by-eye s_u at κ = 1 and κ = 0.25.
4. **ε_r, on home log loss only.**

Nothing is tuned on edge accuracy of built-raised agents, calibration, decisive payoff, speed-up, entropy or code length.

## Design checks, each pass or fail

- **R1, legibility:** at the chosen λ, wild width-32 agents of both types reach edge accuracy ≥ 0.85 and a decisive ratio ≥ 0.6, and appearance slows learning by at least 2× (steps to competence 0.9 with appearance active vs held constant).
- **R2, exploration:** at the chosen settings, a lone width-32 wild agent reaches competence 0.9 within 20,000 steps.
- **R3, balanced difficulty:** type-1 and type-2 newborns' natural-world learning costs are within 1.5× of each other on average.
- **R4, shaping moves only what the builder reads:** in a world built by type-1 builders under shaping by eye, u2 stays in its drift band (mean |u2 − home| under 0.05), while u1 moves at least 0.3 on average at κ = 0.25. Builders' |∂p/∂u1| is above zero throughout the warm-up.
- **R5, update counts:** acted turns are logged everywhere, and E8's treatment and filtered arms are within 10% of each other.
- **R6, ensembles behave as designed:** randomized-prior members disagree more on natural edge items than standard members do, for built-raised agents (spread of member predictions).
- **R7, cost:** wall-clock, to size the real run.

## Readouts reported but not judged

L1 (κ = 1 and κ = 0.25 forms); L3, L3b, L3c and L3d, and where same-type newborns fall; L8a, L8a′, L8b and L8c with probe and weight norms; L9 with act rate and refusals per act, and L9b; L10a (refused and accepted edges), L10a-ens for all three ensembles, and L10b at the matched ε_r; the update-count check. Each against its version 4 threshold, labelled as a pilot readout. A design flaw found along the way is logged with a proposed fix; no threshold changes because of this pilot.

## Not in pilot 3

E2's slopes, E4–E7, E_K and the spillover arm.
