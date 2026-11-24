# Pilot 2 spec (written 20 Nov 2026, before any pilot 2 code)

**What pilot 2 is for.** It checks whether version 3's redesign fixes what pilot 1 and Neon's review found, before CC spends time on it. Like pilot 1, it is not evidence for or against any prediction: small worlds, few seeds. Seeds 2000 and up (real runs use 0–999; pilot 1 used 1000–1002).

**Scale.** 600 items, 16-dimensional observations, γ = 1.5, observation noise 0.05. Builders: M = 6, width 32, frozen after warm-up. Raised agents: widths 8, 32, 128. Three seeds. One offer per turn with a skip (payoffs +1 / 0 / −2), act threshold 2/3, ε = 0.05.

**Tuning allowed.** Learning rate on seed 2000 for competence ≥ 0.9; shaping strengths and δ chosen from the E2 regime sweep (check Q4) so that the built share is unsaturated; ε_r chosen by home log loss only (check Q3). Nothing is tuned on edge accuracy, calibration, decisive payoff, speed-up or entropy.

## Design checks, each pass or fail

- **Q1, edge knowledge now matters:** on natural decisive edge turns, the nearest-centre heuristic (act only if the offered action's centre is nearest) gets at most 0.8 of the optimal payoff, and wild-raised width-32 agents get at least 0.6 of optimal. If the heuristic does as well as the agents, Bug 1 isn't fixed.
- **Q2, E3 isn't trivial:** different-body newborns reach competence 0.9 in the natural world within the budget. If they can't, any E3 difference is just that they can't learn.
- **Q3, matching is possible:** some ε_r in {0, 0.05, 0.1, 0.2} puts built-raised width-32 agents' mean home log loss within ±20% of wild-raised width-32 agents'.
- **Q4, unsaturated building exists:** under wear and under shaping by eye (κ = 0.25 and 1), some setting on a small grid gives a built share between 0.2 and 0.8.
- **Q5, the second task is learnable but not instant:** in the natural control, width-8 agents need more than 1,000 steps to reach competence 0.9 on kind-B items, and reach it within T2.
- **Q6, cost:** wall-clock, to size the real run.

## Readouts reported but not judged

L3, L3b and L3c (both κ, and wear); L8a, L8a′, L8b and L8c; L9 and L9b; L10a, L10a-ens and L10b at the matched ε_r. Each against its version 3 threshold, labelled as a pilot readout. A design flaw found along the way is reported and logged, with a proposed fix; no threshold changes because of this pilot.

## Not in pilot 2

E4–E7, E_K, E2's slopes (only its regime sweep), the accidental co-option idea.
