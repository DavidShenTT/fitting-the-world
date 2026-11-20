# PREREGISTRATION.md

Committed 20 Nov 2026, before any code for the real runs and before any real seed (0–999) has been run. It applies BRIEF.md v7.2, PLAN.md and REVIEW_1.md (approved by David 20 Nov). Where they differ, REVIEW_1 wins, then this file, then PLAN.md, then BRIEF.md. The git commit time of this file is its timestamp.

**Framing (REVIEW_1 C).** Manifestation and its lifecycle (E1–E9) are the subject; naive realism (E10) is a secondary consequence, reported in full.

**Addendum rule.** Section 11 is left for the frozen settings: the tuned values (legibility λ, exploration, ε_r, E2's sweep range) and the hash of the frozen code. It is filled after tuning on seeds 7000–7099 and the smoke test, and committed **before the first real seed**. Tuning uses only the measures BRIEF.md §6 allows (wild-agent legibility, competence, home log loss; built share for E2's range, as BRIEF.md E2 prescribes). Nothing else in this file may change after that without an entry in DEVIATIONS.md logged before the run it affects.

## 1. Changes from BRIEF.md, stated as such

1. **Geometry per experiment (REVIEW_1 A).** E1, E2, E4, E5 and E6 run in the **symmetric world** (all arcs ±60°; overlaps of 30° between neighbours, a third of the ring). E3, E8, E9 and E10 run in the **unequal world** (±60° wide, ±35° narrow). E7, if approved, uses the unequal world (section 9).
2. **Built share in the symmetric-world experiments (REVIEW_1 A)** follows version 1: of the kind-A items whose home lies in an overlap zone (accepted by two actions), the share currently sitting outside every overlap zone. The unequal-world experiments keep BRIEF.md v7.2's boundary-distance built share where they use one.
3. **L10-dose levels (REVIEW_1 B8):** shaping by eye at κ = 1 with s_u = 0.03, 0.1 and 0.3 (pilot 5's levels, roughly 22%, 8% and 1% edge items left), not levels inside E2's unsaturated range.
4. **E4's definitions (REVIEW_1 B10)** come from version 1 (`BRIEF_v2_OLD.md`): behavioural agreement is chance-corrected (Cohen's κ), and E_K is computed on the hidden (penultimate) layer with the Convergence code.
5. **Builders' policy is held fixed across feedback conditions.** In pilot 6 the frozen builders inherited the learner's ε_min (so builders acted 30% at random in the ε_min = 0.3 arm), and in the ensemble runner they had no margin exploration. Here frozen builders always use the base policy (act above 2/3, ε_m = 0.6 within m = 0.2, ε_min = 0.02), in every condition and for every learner type. Only the learner's feedback changes between conditions.
6. **E5's L6** keeps version 1's two extra bullets as sanity checks, not predictions (BRIEF v7.2 makes the halving time a sanity check).

## 2. Common set-up

- **Items:** 600 kind-A items per world; home angle uniform on the ring, home appearance (u1, u2) standard normal. Drift toward home at δ = 0.0005 per step in every coordinate, angle noise 0.05°/step, appearance noise 0.002/step.
- **Bodies:** 8 fixed random maps per world, 4 of type 1 (read cos α, sin α, λ·u1, kind) and 4 of type 2 (λ·u2); 32 tanh hidden units, γ = 1.5, 16-dimensional observation, observation noise 0.01. λ = 0.4 unless the legibility check (section 3) fails in a geometry.
- **Agents:** MLP observation → one tanh hidden layer → 4 sigmoid outputs; neutral init (zero output bias); plain SGD on binary cross-entropy of the offered action, lr 0.05. Policy: act if p > 2/3; within ±0.2 of 2/3 act with probability 0.6; floor ε_min = 0.02. Evaluation is greedy and noise-free in the policy (observation noise kept at 0.01 where a measure says so).
- **Offers and payoffs:** each turn an item is drawn uniformly and offers one action uniformly; accepted +1, skip 0, refused −2. Learners learn only from acts, except under passive feedback (they also learn the offered action's verdict when they skip).
- **Shaping rules:** wear η = 0.15 (angle moves 15% of the way to the action's centre after an accepted use); shaping by eye with step s_u per coordinate along the sign of the agent's own logit gradient (angle κ·s_u·20°), s_u = 0.3, κ = 1 or 0.25; no-shaping control. Only accepted uses shape. Newborns never shape; builders do.
- **Builders:** M = 6 agents sharing one map (map 0, or map 4 where a design rotates builder type), width 32, from random init, learning and shaping for a 30,000-step warm-up; then frozen copies keep acting (no learning) while newborns learn alongside them.
- **Natural test set:** 3,000 fresh items at fresh home values, never shaped, scored without learning; decisive sets of 300 inside + 300 outside per action within 10° of its own edge.
- **Seeds:** real 0–999, the first 10 (0–9) per condition unless stated; tuning and smoke test 7000–7099.
- **Statistics:** the world (seed) is the unit; agents within a world and cell are averaged first. Criteria are judged on the point estimate as written; 95% percentile-bootstrap intervals over worlds (10,000 resamples) are reported beside them. "At least k times wild" is the geometric mean over worlds of the paired per-world ratio (built ÷ wild, same seed and condition); a world with a zero wild rate gets +0.5 counts in numerator and denominator in that world only, and the number of such worlds is reported.

## 3. Stage 2 procedures before real seeds

- **Legibility, wild agents only, both geometries (seeds 7000–7004):** width 32, 60k steps, skip rule. Every seed must give edge accuracy ≥ 0.85 and decisive ratio ≥ 0.6 for actions 0/2 and 1/3 separately, and learned half-widths within ±3° of the true ones. If a geometry fails on any seed, lower λ for that geometry in steps of 0.05 and repeat; each step logged. Then the narrowest width meeting the targets in the unequal world (candidates 4, 8, 16) sets E8's ladder: that width ×1, ×2, ×4, ×8, ×16.
- **Exploration:** a lone wild agent reaches competence ≥ 0.9 within 60k steps on every tuning seed, in both geometries. Retune m, ε_m on competence only if not.
- **ε_r for L10a-matched (unequal world, 2 agents × 5 tuning seeds):** for each built rule and main condition, the ε_r in {0, 0.01, …, 0.20} whose mean built home log loss is closest to the wild mean, required to be within ±20%; if none is, the arm is reported as unmatched.
- **E2's sweep range** (section 5).
- **Smoke test:** seed 7010 through E1–E6 and E8–E10 end to end with the frozen code and full-size settings; every logged column filled and finite where it should be. E7 is outside the smoke test unless approved.
- **Sanity checks (verify code, report no finding):** (a) E1 under wear clusters items in the symmetric world (built share ≥ 0.3 above control); (b) under wear, the built share responds to pull/drift balance in the predicted direction across δ; (c) after abandonment the mean displacement |α − home| of overlap-home items halves in ln 2 / δ ≈ 1,386 steps (exponential fit within 10%); (d) with builders frozen but still acting, the built share stays within 10% over 5/δ steps. A failed sanity check stops the run for debugging.

## 4. E1: does building emerge under shaping by eye? (symmetric world)

M ∈ {2, 4, 6, 12} builders from random init, shaping by eye at κ = 1 and κ = 0.25 (s_u = 0.3), plus the no-shaping control at each M; 60,000 steps; 10 real seeds. Built share, the builders' read- and unread-axis movement and the appearance gradient are logged every 1,000 steps.
- **Steady state:** the first 5,000-step window whose mean built share differs from the previous window's by < 0.02; the steady-state value is the mean over the last 10,000 steps. Runs that never settle are reported as such and still scored on the last 10,000 steps.
- **L1 (κ = 1):** for every M, mean over worlds of (built share − the same-M control's) ≥ 0.3. **Against:** ≤ 0.1 for any M.
- **L1 (κ = 0.25):** for every M, the builders' read appearance axis (u1, builders on map 0) sits ≥ 0.3 from home on average over kind-A items at steady state. **Against:** < 0.1 for any M.
- **Logged:** builders' appearance gradient over the warm-up; unread-axis movement (expected at drift level); edge share.

## 5. E2: the upkeep law (symmetric world, shaping by eye κ = 1)

- **Built items:** the count of overlap-home items currently outside every overlap zone, at steady state (version 1, REVIEW_1 A).
- **Sweep and range (confirmation 2).** Pilot 1 saw 0.99–1.00 built share in the symmetric world at its settings, so the sweep starts well below them and the range is chosen to exclude saturation. On tuning seeds 7020–7022: M ∈ {1, 2, 4, 8, 16} × s_u ∈ {0.01, 0.03, 0.1, 0.3} × δ ∈ {0.0005, 0.001, 0.002, 0.004}, builders width 32. A cell enters the test range only if its tuning-seed mean built share lies in [0.1, 0.9] **and no tuning seed in it exceeds 0.95**. The test range is then the largest full factorial block (at least 3 levels of M and 3 of δ at one or more s_u values) inside those cells. If no such block exists, the grid is extended downward in s_u (0.003) and upward in δ (0.008) by the same rule, logged, before any real E2 seed. Any real-seed cell whose world-mean built share comes out above 0.95 is reported and flagged as saturated in the slope fit's sensitivity analysis (the fit is also run without it).
- **Real run:** the test range on seeds 0–9, plus builder network widths 16, 32 and 64 at the range's middle cell (every level of M and δ at width 32; width varied at the central M, δ).
- **L2:** OLS of log(count) on log M, log(1/δ) and log(width), world-level points: slopes on M and on 1/δ in [0.75, 1.25]; slope on width within ±0.25. **Against:** any slope outside its range. Bootstrap intervals over worlds reported.
- **Exploratory:** the lowest M, and the lowest builder competence (warm-up cut short at 2k, 5k, 10k steps), at which the built share stays within 0.05 of the control.

## 6. E3: who benefits? (unequal world)

Builders on one map build for 30,000 steps under shaping by eye at κ = 0.25, and under wear as the control. In even seeds the builders use map 0 (type 1), in odd seeds map 4 (type 2). Newborns learn in the built world with the frozen builders still acting, and in the natural world, paired by map and initial weights: same body (the builders' map), same type (the other three maps of that type), outsiders (the four maps of the other type); 3 newborns per map per world; seeds 0–9.
- **Measures:** early learning-curve area (mean competence over steps 0–5,000, checked every 250 steps, built minus natural for the same map and init); steps to competence 0.9 (checked every 250 steps, cap 20,000); home log loss at 2,000 steps; passive code length over 8,000 pairs of the frozen state; forestry entropy (k-means K = 32 fitted on each body's noise-free observations of the natural pool).
- **L3c (primary):** entropy drop (natural − built state) ≥ 0.5 bits under the builders' body, and the outsiders' drop ≤ half the builders'. **Against:** outsiders' drop ≥ 80% of the builders'.
- **L3:** same-body gain in early learning-curve area ≥ 1.5 × outsiders' (ratio of world-level means of gains). Under wear the two gains are within 20% of each other. **Against:** under κ = 0.25 the gains within 20%.
- **L3b:** home log-loss drop (natural-raised minus built-raised, same map, after 2,000 steps) ≥ 1.5 × as large for same body as for outsiders.
- **L3d:** code-length saving (natural − built, same newborn map and init) ≥ 1.5 × as large for same body as for outsiders.
- **Exploratory:** same-type newborns on every measure; steps-to-criterion speed-up.

## 7. E4–E6: routes, decay and bodies (symmetric world)

**Worlds.** Per seed: the natural world, and two built worlds A and B from the same home values but different builder seeds (6 builders on map 0, 30,000-step warm-up), under wear and under shaping by eye at κ = 1. Newborns (width 32, map 0 unless stated, skip rule) learn for 60,000 steps alongside the frozen builders.

**Pairs.** Per seed and rule, 5 pairs of each kind: same-world (both raised in A), different-world (one in A, one in B), natural (both in the natural world). Seeds 0–9.

**Readout of the chosen action (confirmation 3).** The agents act on one offer at a time, so their act/skip choices never reveal which action they would pick. The action-agreement readout uses the network's four outputs instead: an agent's **chosen action** on an item is the argmax of its four predicted acceptances, i.e. the action its greedy policy rates best if all four were offered. Skip is not a category: on every item at least one action is accepted, so the question is which, not whether. Each agent reads each item in the state it has in **the agent's own world** at the end of raising (so different-world partners see the same item, identified by its shared home, in their two worlds' states), through the agent's own body, without observation noise. Items: the world's 600 pool items, split by home into overlap-home items (home accepted by two actions; about a third) and unambiguous-home items (home accepted by one action and more than 10° from every overlap zone). Agreement of a pair is Cohen's κ over those items: (observed share of same chosen action − chance) / (1 − chance), with chance = Σ_a f₁(a) f₂(a) from each agent's own action frequencies on those items. As a secondary readout, the same κ on the natural test set's overlap and unambiguous items (both agents see the same fresh items).

**Representational agreement.** E_K: the Convergence K statistic (`src/metrics.py`, cosine distance, density ρ = 0.01, i.e. 6 nearest neighbours among 600 items) on the two agents' hidden-layer activations over the 600 pool items (each agent's own world state, as above), as normalised excess over the exact permutation-null mean, E = (K − K₀)/(1 − K₀).

- **L4:** on overlap-home items, mean κ(same-world) − κ(natural) ≥ 0.2; κ(different-world) − κ(natural) ≤ 0.05; on unambiguous-home items all three κ ≥ 0.9 and within 0.05 of each other. Judged per shaping rule. **Against:** no same-world advantage on overlap-home items (difference ≤ 0).
- **L5:** E_K(same-world) − E_K(different-world) > 0, with the bootstrap interval over worlds above 0.
- **E5 (decay).** Remove every agent from A and from B; drift only for 5/δ = 10,000 steps (sanity checks c and d in section 3). Then raise newborn pairs in the abandoned A and B (still drifting, no builders) for 60,000 steps, and natural pairs. **L6:** same-world advantage, κ(same-world) − κ(natural) on overlap-home items, < 0.05.
- **E6 (no prediction).** Natural world, pairs of newborns on one map of each type (map i with map 4 + i, i = 0..3), 5 pairs per seed. Reported: κ on overlap-home items with its interval (κ > 0 is above chance), beside same-map and same-type-different-map pairs as references.

## 8. E8–E10: the lifecycle (unequal world)

**Raised agents.** Wild-raised (natural world, alone) and built-raised (in a world built by 6 builders on map 0 under wear η = 0.15, or shaping by eye κ = 1, s_u = 0.3, after the warm-up, builders still acting) get equal environment steps; newborns on map 0. **Update-count control:** for each seed, rule and main condition at the primary cell, a wild agent trained in 2,000-step chunks until its acted turns reach the built-raised mean (cap 240k steps); E8–E10 conclusions are reported both ways and any difference is stated.

### E8: what gets forgotten
Pretrain in the natural world until edge accuracy (mean of wide and narrow, natural test set, every 2,000 steps) changes by < 0.01 between successive checks (cap 100k; capped agents reported). Copies then train for T2 = 30,000 steps in: treatment (built world, builders acting), filtered control (natural world, the offered-edge share matched to the treatment world's current edge share, sampler redrawn every 50 steps), natural control. Widths: the ladder from section 3; both rules; 3 agents per width per seed; seeds 0–9.
- **L8a:** at the narrowest width, edge-accuracy fall (before − after) in treatment minus in natural control ≥ 0.10. **Against:** < 0.03.
- **L8a′:** treatment and filtered falls within 0.03 is the expected result; treatment falling ≥ 0.05 more is reported as a building-specific effect. Offered edge shares (within 0.5 points) and acted turns (within 10%) are reported.
- **L8b:** narrowest width's extra fall ≥ 2 × the widest's.
- **Figure, no criterion:** population built share and individual edge accuracy across T2.

### E9: fragility
The primary-cell raised agents (width 32, 60k) of E10 move into the natural world snapped to home values, no builders, and keep learning under their own feedback condition for 500 steps. Decisive payoff is scored on the natural test set at steps 0, 100, …, 500 and averaged, wide and narrow pooled 50/50.
- **L9 (skip rule):** (wild − built) / wild ≥ 0.30, world-level means, per rule. **Against:** < 0.10.
- **L9b:** log-loss jump at the snap (home log loss on the snapped world minus on the agent's home world just before), built minus wild ≥ 0.1 nats.
- **Mechanism, no criterion:** act rate and refusals per act per arc type, beside each agent's shift, under the skip rule and passive feedback (REVIEW_1 B5).

### E10: naive realism (secondary)
- **Grid:** seeds 0–9 × 3 agents × {wild, wear, κ = 1} × {skip rule, passive} × the five widths × budgets {30k, 60k} (the 30k budget is the same agent scored at 30k steps).
- **Primary cell (REVIEW_1 B6):** L10a, L10-shift, L10m and L10-dose are judged at **width 32 and 60,000 steps**. Every other width × budget cell is secondary and reported in full, with the number of cells in which each criterion holds.
- **L10a (primary within E10), per main condition and per built rule:** confident-error ratio ≥ 1.5, unbuilt-edge log-loss ratio ≥ 1.5, and world-mean home log loss built ≤ wild. **Against:** either ratio ≤ 1, or built home log loss higher.
- **L10a-matched:** built worlds with the tuned learner-only ε_r; 4 agents per seed (40 per kind), skip rule and passive. Agents pooled per condition and rule are binned by home log loss into tertiles of the pooled distribution; bins need ≥ 10 agents of each kind (otherwise merge to the largest number of bins that qualifies; with fewer than 3 the test is reported as not runnable). In every bin both error ratios (agent means within the bin) ≥ 1.5 and built home log loss ≤ wild + 0.005 nats.
- **L10-shift (skip rule):** built − wild unsigned boundary shift, averaged over the two arc types, ≥ 2.5°. **Against:** < 1°.
- **L10-dose (skip rule):** shaping by eye κ = 1 at s_u = 0.03, 0.1, 0.3 (section 1, item 3); both the pooled-shift difference and the confident-error ratio rise strictly across the three levels ordered by the share of edge items removed (measured, not assumed).
- **L10m:** mean stretch = (narrow − wide)/2 of the toward-45° shifts, built − wild, under the skip rule at ε_min = 0.02, 0.1, 0.3 (learner only); prediction: strictly falls. **Against:** not falling. Also reported: raw stretch, near- and far-side refusal shares, acceptances per far-side refusal and its rank correlation with raw stretch.
- **L10a-ens:** standard, diverse (member widths 6, 13, 26, 45, 64) and randomized-prior (β = 10) ensembles of K = 5, skip rule and passive, both rules, 3 per seed. First the manipulation check: randomized-prior member spread on unbuilt edges ÷ at centres ≥ 2, else the arm is dropped and reported. Readings as pre-stated in BRIEF.md E10.
- **Exploratory:** the symmetric-arc check (wild vs wear, skip and passive, label shares, refusal sources, angle histograms; labelled a partial gap); within built-raised agents, the rank correlation of home log loss and confident error. Biased offers are not run.

## 9. E7 (optional; runs only if David approves after stage 3)

**Geometry (confirmation 1): the unequal world.** E7 belongs with the lifecycle experiments (E7–E10), which use the unequal world, and its question (what is worth building near an edge) needs edges where one action ends and no neighbour takes over: in the unequal world an edge item is refused by every other action except inside the 5° overlaps, while in the symmetric world every edge lies 30° inside a neighbouring arc.
- One agent (width 32, map 0) with 400 private items, homes within 10° of an arc edge, so there are no free riders. A 2 × 2 design: encounter frequency (high: 4× the visit rate of low) × prediction difficulty (observation noise 0.01 vs 0.05), 100 items per cell.
- Each visit, the agent chooses Act or Build-and-act. Building costs c, then pulls the angle a fraction η = 0.15 toward the centre of the agent's predicted-best action. The choice is learned by an ε-greedy bandit (ε = 0.1, incremental mean) over the item's observable context, which is its cell (frequency × noise level), given to the bandit as a label. c is set on tuning seeds so the overall build rate lies in [0.1, 0.5], with no cell-specific tuning.
- **L7:** build choices in the frequent-and-noisy cell ≥ 2 × as common as in any other cell, and its excess over the base rate larger than the sum of the two single-factor excesses. 10 seeds.

## 10. What happens at check-ins
After E1 (stage 2), after E8–E10 (stage 3), and before REPORT.md (stage 4), work stops for David. REPORT.md has one row per prediction with the criterion above, the result, the regenerating command, and what it is sensitive to. Failed predictions are reported as plainly as passed ones.

## 11. Frozen settings (addendum, filled before the first real seed)

Completed 20 Nov 2026 after tuning (seeds 7000–7032) and the smoke test (seed 7010); committed before any real seed runs. **Frozen code: commit `de619a5`** (`src/world.py`, `src/config.py`, `src/exps.py`, `runs/run.py`). Any later code change that touches a real run goes in DEVIATIONS.md first.

**Tuned or confirmed values.**
- **λ = 0.4 in both worlds; raising budget 60k:** kept by David's decision although the legibility check isn't met on every tuning seed (DEVIATIONS.md D1; 3-agent seed means pass on 4/5 seeds in the unequal world, 2/5 in the symmetric world; half-widths within ±3° on every seed mean).
- **E8 width ladder: 4, 8, 16, 32, 64** (DEVIATIONS.md D1, width-ladder note). L8a's narrowest width is 4; L8b compares 4 with 64.
- **Exploration unchanged** (m = 0.2, ε_m = 0.6, ε_min = 0.02): a lone wild agent reached competence ≥ 0.9 within 60k on every tuning seed (by 10k in the symmetric world, by 35k in the unequal world at any width; `results/tuning/leg/leg.csv`).
- **ε_r for L10a-matched** (home log loss only; `results/tuning/epsr_summary.csv`): wear 0.01 (skip), 0.02 (passive); shaping by eye 0.01 (skip), 0.02 (passive). Built ÷ wild home log loss at those values: 0.93, 1.18, 0.87, 1.18, all within ±20%.
- **E2's test range: not set.** No cell block satisfies section 5's rule, even after the preregistered extension (s_u 0.003, δ 0.008): the cells in [0.1, 0.9] lie on a diagonal band where built share tracks M/δ (and s_u), so no 3 × 3 rectangle of M × δ levels fits inside the band on the power-of-2 grid (`results/tuning/e2/cells.csv`). The code refuses to run real E2 seeds until a range is set. E2 runs in stage 4; the range is a decision for the E1 check-in, logged in DEVIATIONS.md before any real E2 seed.

**Sanity checks (all passed).** (a) Wear minus control at M = 6: built share +0.86. (b) Wear's built share falls monotonically with δ: 0.99, 0.91, 0.54, 0.29, 0.17 at δ = 0.00025 … 0.004 (`results/tuning/sanity_wear/summary.csv`). (c) Half-time after abandonment 1,358–1,387 steps against ln 2 / δ = 1,386. (d) Frozen builders keep the built share within 3.5–8.1% of its start (`results/smoke/e4/decay_sanity.csv`; seed 7010).

**Smoke test.** Seed 7010 through E1, E2 (code-check range only), E3, E4/E5, E6, E8, E9 and E10 at full size: every table and column filled. The one column with gaps is `target_edge_share` in E8, empty by design outside the filtered arms (`results/smoke/smoke_check.txt`). Wall time on 14 workers: E10 about 2.5 min per seed, the rest under 3 min each.

**Implementation details fixed here (no change to any criterion).**
- Pooled unsigned shift per agent = ½(|shift_wide| + |shift_narrow|), each the mean signed toward-45° shift of its two actions, as in pilot 6.
- E10's primary cell (width 32, 60k) has 4 agents per seed for every kind and main condition, so wild agents serve as L10a-matched's comparator (40 per kind); other grid widths, mechanism and dose arms have 3.
- E4: same-world and natural pairs are raised together in one copy of their world (newborns don't shape, so they meet only through what they see); different-world partners are raised in separate copies of A and B.
- E6: 5 pairs per kind; cross-type pairs use maps (0,4), (1,5), (2,6), (3,7), (0,5); same-map (0,0) … (4,4); same-type (0,1), (1,2), (2,3), (3,0), (4,5).
- E8: the filtered arm's target is the treatment arm's realised offered-edge share for the same agent and rule.
- E3: one newborn per (map, j) provides the early area, steps to criterion and the 2,000-step home log loss; code length uses a fresh newborn with the same initial weights.
- Sanity (c) uses the signed displacement projected on its sign at abandonment, whose expectation decays exactly as (1 − δ)^t.
