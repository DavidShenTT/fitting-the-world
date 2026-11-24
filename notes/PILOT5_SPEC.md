# Pilot 5 spec (written 20 Nov 2026, before any pilot 5 code)

**What pilot 5 is for.** It checks version 6's new E10 design at small scale before CC spends time on it:
- whether reweighted passive feedback hits its target acceptance share;
- whether the two pulls of L10s can be read at pilot scale, and how many agents real runs need;
- whether the symmetric-arc world is legible, and what it shows;
- whether a lower upkeep gives a usable dose of erasure;
- whether the redrawn filtered sampler matches what the built world offers.

Like the earlier pilots, it is not evidence for or against any prediction. Seeds 5000 and up (pilots 1–4 used 1000–4002, real runs 0–999). E10 only, plus the sampler check.

**Scale.** As pilot 4: 600 items, 16-dimensional observations, γ = 1.5, λ = 0.4, observation noise 0.01, one offer per turn with a skip, margin exploration (m = 0.2, ε_m = 0.6, ε_min = 0.02), learning rate 0.05. Builders: six type-1 agents of width 32, raised from random initialisation for 30,000 steps, then kept acting. Raised agents: single networks of width 32, 60,000 steps, three agents per seed per cell (nine per cell over three seeds). Three seeds.

**Worlds per seed.**
- **Unequal arcs (version 5's world):** natural; wear (η = 0.15); shaping by eye (κ = 1, s_u = 0.3).
- **Symmetric arcs (all ±60°):** natural; wear (η = 0.15).
- **Dose:** shaping by eye at κ = 1 with two lower s_u values, chosen in tuning.

**Feedback conditions.**
- **Unequal worlds:** skip rule; passive; passive reweighted to s = 0.5; passive reweighted to s = 0.9.
- **Symmetric worlds:** skip rule; passive; s = 0.5.
- **Dose worlds:** skip rule; passive.
- **The reweighting:** at each redraw of the sampler (every 500 steps), for each action type, q is the share of the learner's offerable items that the action type accepts. An acceptance's learning step is multiplied by w = s(1 − q) / (q(1 − s)). This applies to every update in the arm, acted or not.

**Tuning allowed, on seed 5000 only:**
1. **The dose levels, on built share and edge share only.** Shaping by eye at κ = 1 with s_u in {0.1, 0.03, 0.01}. Choose two levels that leave edge shares spread between pilot 4's s_u = 0.3 (about 1%) and natural (about 27%), aiming for roughly 5–10% and 15–20%. If no level lands there, take the two that spread most and say so.

Nothing is tuned on any boundary shift, label share, confident error or home log loss.

## Design checks, each pass or fail

- **P1, legibility on every seed:** wild agents under the skip rule in the unequal world reach edge accuracy ≥ 0.85 on wide and on narrow actions, and learned half-widths within ±3° of 60° and 35° (mean of three agents per seed).
- **P2, the symmetric world is legible:** the same, with all half-widths 60°. If this fails, the symmetric readouts are reported as uninterpretable.
- **P3, the reweighting hits its target:** the weighted acceptance share of each action type's labels is within ±0.03 of s, for wild and built-raised agents.
- **P4, a usable dose:** the two chosen levels give edge shares that differ from each other and from s_u = 0.3 by at least 3 points.
- **P5, the matched filtered sampler:** a wild agent learning in the natural world for 30,000 steps, with the filtered sampler redrawn every 50 steps at the wear world's offered edge share, is offered edge items within 0.5 points of what a built-raised agent in the wear world is offered.
- **P6, why wear erases so much (a diagnosis, not pass or fail):** the edge share left by wear at η = 0.15 in the symmetric world against the unequal world. Stated before running: if the cause is that 30°-wide overlaps hold items in a tug of war between two actions while 5° overlaps don't, the symmetric world keeps many more edge items (pilot 3 kept 21%).
- **P7, cost:** wall-clock.

## Readouts reported but not judged

- **L10a under each feedback condition:** the pooled unsigned shift difference, the confident-error ratio and home log loss, built minus wild.
- **L10s:** G and L for each condition, whether L rises with s across the passive conditions, the s = 0.5 case, and s = 0.9 against the skip rule.
- **The symmetric check:** shifts beside label shares.
- **The dose readout:** pooled shift difference against edge share removed.

Each is reported against its version 6 threshold and labelled as a pilot readout. Also reported: the agent-to-agent spread of G and L, to estimate how many agents real runs need. A design flaw found along the way is logged with a proposed fix; no threshold changes because of this pilot.

## Not in pilot 5

E2's slopes, E3, E4–E7, E8 (apart from P5), E9, ensembles, L10b and E_K.
