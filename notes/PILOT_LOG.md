# Pilot log and deviations

Every change below is logged before the runs it affects.

## 20 Nov

- **Drift noise set to 0.05°/step** (it was 0.2 in the first draft of `world.py`, not in the spec). At 0.2 the natural band had a standard deviation of about 6°, so items wandered across zone boundaries with nobody acting; the sanity test "η = 0 items stay in their noise band" failed (max 19°). At 0.05 the band is about 1.6° and the test passes. No outcome measure had been computed.

- **C1 tuning (seed 1000).** A lone width-32 agent reaches 95% greedy acceptance after 500–1,500 steps at every learning rate tried (0.02–0.2), and greedy acceptance then sits at 0.99–1.00.

- **D1, design flaw found: the greedy policy never needs the arc edges.** With 60° arcs every 90°, the action whose centre is nearest an item is always accepted, with at least 15° to spare. So acceptance, and the planned "edge accuracy" (greedy action on edge items), are at ceiling by construction. C3 would fail, and E8–E10 could not show a fall. This also bears on version 1's E3 and E4, which use learning cost to 95% acceptance. Two fixes, both adopted for the rest of the pilot and proposed for PLAN.md:
  1. **Two offers per turn.** Each turn an item affords two of the four actions, drawn at random; the agent picks one of the two (ε-greedy between them). Near a boundary the agent must then know whether the offered action is still accepted. Builders act the same way.
  2. **Edge measures read the agent's model, not just its choice.** For each natural edge item, take the *boundary action*: the action whose arc edge lies within 10°. **Edge accuracy** = share of edge items where the agent's prediction for that action, thresholded at 0.5, matches the world. **Calibration gap** = mean confidence in that prediction (max(p, 1 − p)) minus edge accuracy; positive means overconfident.
  - **Competence** becomes: of the turns where at least one offered action is accepted, the share on which the greedy pick is accepted, checked on 400 held-out offers every 500 steps.
  - The brief's thresholds are unchanged. C3 now applies to the new edge accuracy.

- **C1 after D1 (seed 1000).** Competence still reaches 95% within 500 steps at every learning rate, and sits at 0.99–1.00 after. Learning rate fixed at 0.05, ε at 0.05.

- **D2, E8's pretraining length.** The brief says agents first reach 95% acceptance, then T2 starts. Competence arrives within about 500 steps, long before the agent knows where the arcs end, so control agents would keep improving at the edges during T2 and the treatment–control gap would mostly measure that. For the pilot, E8 pretrains for a fixed 15,000 natural steps (the same budget as raising), then runs T2 = 15,000 steps. Proposed for PLAN.md: pretrain until edge accuracy has levelled off, with a criterion fixed in advance.
