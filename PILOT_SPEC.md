# Pilot spec (written 20 Nov 2026, before any pilot code)

**What the pilot is for.** It checks the design of E1, E8, E9 and E10 in `BRIEF.md` at small scale, before CC spends a day on it. It is not evidence for or against any prediction: it uses tiny worlds, few seeds and wear only. Its seeds are 1000 and up; real runs use 0–999.

**Scale.** 600 items, 16-dimensional observations, one body (same map for everyone), γ = 1.5, observation noise 0.05. M = 6 builders of width 32. Widths for raised agents: 8, 32, 128. Three seeds per condition. Wear only (η = 0.2), drift δ = 0.0005; shaping by eye is implemented and smoke-tested only for E1.

**Tuning allowed.** Learning rate and exploration may be set on seed 1000 so that a lone width-32 agent reaches 95% greedy acceptance in the natural world. Nothing else is tuned, and nothing is tuned on any outcome measure.

## Design checks, each pass or fail

- **C1, competence:** a lone width-32 agent reaches 95% greedy acceptance (held-out check of 400 items every 500 steps) within 20,000 steps, on all three seeds.
- **C2, building emerges:** after 30,000 steps of 6 builders under wear, the built share is at least 0.3 above the η = 0 control on all three seeds. (E1's criterion, used here only to see whether the world can build at all.)
- **C3, no ceiling or floor:** wild-raised width-32 agents' natural edge accuracy is between 0.55 and 0.95. Above 0.95, E8 and E10 can't show a fall; the edge window would need narrowing.
- **C4, built worlds remove edge cases:** the share of edge items in the built world is at most half that in the natural world. If not, built-raised agents see the same hard cases and E8–E10 test nothing.
- **C5, cost:** wall-clock per condition, to size the real run.

## Readouts reported but not judged

The directions of L8a, L8b, L9, L10a and L10b, with the numbers, labelled as pilot readouts. If one comes out against its prediction, that's reported plainly, and it doesn't change the brief's thresholds. If a readout reveals a design flaw (for example, a measure that can't move), the flaw is reported and a fix proposed for PLAN.md.

## Not in the pilot

E2–E7, L8c (the second item type), shaping by eye beyond the E1 smoke test, different bodies, E_K.
