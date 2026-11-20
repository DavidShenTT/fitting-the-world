# Pilot 6 spec (written 20 Nov 2026, before any pilot 6 code)

**What pilot 6 is for.** Neon judged version 7 ready to freeze unless the biased-offer check throws a surprise. Pilot 6 is that check, kept small. It asks:
- whether biased offers hit their acceptance share, and how far they move wild boundaries;
- what L10a's primary measures look like across the three main conditions, including the new unbuilt-edge log loss;
- what L10m's contrast and the ε_min sweep show;
- whether the refusal sources differ as Neon's arithmetic says (about 50:1 acceptances to far-side refusals under the skip rule, about 47:1 under biased offers);
- whether the symmetric world's item histogram shows the density dip Neon proposed;
- how variance splits between seeds and agents.

It is not evidence for or against any prediction. Seeds 6000–6005 (pilots 1–5 used 1000–5002, real runs 0–999). E10 only.

**Scale.** As pilot 5: 600 items, λ = 0.4, observation noise 0.01, learning rate 0.05, margin exploration m = 0.2 and ε_m = 0.6, single networks of width 32 raised for 60,000 steps. Builders: six type-1 agents raised for 30,000 steps, then kept acting. Six seeds × two agents per cell (twelve per cell), following Neon's point that variance across worlds may dominate.

**Worlds per seed.** Unequal arcs: natural, wear (η = 0.15) and shaping by eye (κ = 1, s_u = 0.3). For the histogram only: symmetric natural and symmetric wear (no agents raised there).

**Conditions, in the unequal world, for wild, wear and κ = 1 agents:**
- the skip rule at ε_min = 0.02, 0.1 and 0.3;
- passive feedback;
- biased offers at s = 0.9.

**Biased offers.** For a learner's turn, with probability s the offered action is drawn uniformly from the actions accepting the item, otherwise uniformly from those refusing it. Frozen builders are unchanged.

**No tuning.** Every setting is fixed above.

## Design checks

- **Q1, biased offers hit their share:** each agent's overall acceptance share of labels is within ±0.02 of 0.9. The share per action type is reported; the arcs' widths predict about 0.93 (wide) and 0.87 (narrow).
- **Q2, biased offers barely move wild boundaries:** wild agents' learned half-widths under biased offers are within 1.5° of their values under passive feedback, on each arc type. Stated before running: the conditional outcome is unchanged, but the agent sees noisy observations, so offer density could still act like a weak reweighting near the edge. A larger move means the biased-offer comparison carries a smaller version of pilot 5's D2.
- **Q3, the refusal sources differ as intended:** the ratio of acceptances to far-side refusals is reported per condition, against Neon's arithmetic (about 50:1 skip, 47:1 biased, much lower passive). The ε_min sweep should lower the skip rule's ratio.
- **Q4, the symmetric histogram:** item-angle histograms in 5° bins over a quarter turn (folded) for the symmetric natural and wear worlds and the unequal ones. Neon's prediction: in the symmetric wear world, density drops in [50°, 60°] relative to the mass in [60°, 90°]. Reported, not pass or fail.
- **Q5, variance:** between-seed and within-seed sd of the pooled shift, the mean stretch and the confident-error rate, from this pilot and from pilot 5's saved data.
- **Q6, cost:** wall-clock.

## Readouts reported but not judged

- **L10a under the three main conditions:** confident-error ratio, unbuilt-edge log-loss ratio and home log loss, built against wild.
- **L10-shift under the skip rule.**
- **L10m:** mean stretch under each condition, biased offers' position between passive feedback and the skip rule, and whether the mean stretch falls across ε_min.
- **L10-dose's new confident-error condition,** read from pilot 5's saved dose arms.

Each is reported against version 7's threshold and labelled as a pilot readout. Neon's two pre-stated readings for biased offers are reported as they fall. Design flaws found along the way are logged with proposed fixes; no threshold changes because of this pilot.

## Not in pilot 6

Everything outside E10, ensembles, L10a-matched, the dose arms (read from pilot 5) and agents in the symmetric world.
