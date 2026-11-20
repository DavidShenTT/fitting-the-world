> **Reference only (added 20 Nov, superseded).** Version 2 of the brief, recovered from git. It still carries version 1's text for E1–E6 and the agreement measures, which version 7.2 cites as "as in version 1". Where anything here differs from BRIEF.md, BRIEF.md wins, apart from those definitions.

# Brief: learning agents in a world they can shape (version 2, 20 Nov 2026)

**Status:** a draft for David to review. Don't start work from it until he says so.

**For:** a Claude Code agent on David's PC, in a new folder next to "Convergence Experiments" (suggested name: "Manifestation Experiments"). Copy that project's `CLAUDE.md` over; its standing rules apply here too. This task needs no GPU and no downloads, so the PyTorch and model-cache rules don't bind, but the research-integrity, git and check-in rules do.

**What changed from version 1 (12 Nov).** Sections 1–4 and E1–E6 are unchanged apart from one added measure in E3. New: section 5b (E7–E10, the lifecycle experiments), the pilot rule in section 6 and the order of work in section 7. A small pilot of E1, E8, E9 and E10 runs in a cloud workspace on 20 Nov, on seeds 1000 and up; its report is `PILOT.md`. The thresholds below were committed to git before any pilot code was written.

## 1. Why this exists

**The theory, in three lines.**
- The world supplies only what actually happens; models supply the alternatives. At an agent's boundary (its gate), the world only ever refuses or doesn't.
- Where the world refuses nothing, what agents do is convention. Acting on a convention reshapes the world, so the convention becomes a feature others can read off. The theory calls this manifestation.
- Built features last only while someone keeps them up.

**New in version 2: the lifecycle.** A regularity can give an agent zero surprise three ways: the agent doesn't carve it, the agent predicts it, or the world is held constant so there's nothing to predict. The proposal under test is that these are stages: a lineage notices a variable, learns to predict it, builds the world so it stays put, then stops tracking it. Two consequences can fail:
- **Offloading has a price.** Agents raised in a built world handle the unbuilt world worse, and crash harder when the built structure fails.
- **A mechanism for naive realism.** Low surprise from a built world looks, from inside, like low surprise from a good model. Agents raised in a built world should be confident and wrong when they meet the unbuilt world.

**What has been tested so far.**
- The Convergence programme compared frozen, passive models. It couldn't test any of the above, because none of those models acts.
- Check R, a hand-coded simulation (`check_r/RESULTS.md`, also in the project note `check-r-results.md`), derived what the theory's mechanisms imply. Its main lesson: results turned on costs the simulation had to assume. Here, nothing about the outcome may be coded in. Agents learn, and what they build is whatever their learning produces.

**Prior work to read first (abstracts are enough), and to position against in the report:**
- Hu, Lerer, Peysakhovich & Foerster (2020), "Other-Play for Zero-Shot Coordination", and Treutlein et al. (2021): independently trained agents settle on arbitrary conventions that fail with new partners.
- Cook, Lu, Hughes, Leibo & Foerster (2024), "Artificial Generational Intelligence": cultural accumulation across generations of reinforcement-learning agents.
- Work on stigmergy: coordination by changing a shared environment.
- For E8, work on catastrophic forgetting under distribution shift (one review is enough), so the report can say what is and isn't ordinary forgetting.

What differs here: conventions are written into the world by use, they decay, and the outcome measures are learning cost, agreement between later agents, and what agents raised in a built world can still do without it.

## 2. The world

Propose exact sizes in PLAN.md; the values below show the intended scale.

**Items.**
- A pool of about 2,000 items. Each has an angle on a ring plus a few nuisance coordinates.
- Each item has a home angle, drawn once. Its angle drifts back toward home at rate δ, with a little noise.

**What the world refuses.**
- There are 4 actions. Action a is accepted when the item's angle is within 60° of a's centre (centres at 0°, 90°, 180°, 270°); otherwise it is refused.
- So a third of the ring is overlap, where two actions are accepted, and two thirds is unambiguous.
- An agent only ever learns "refused" or "not refused" for the action it took. It never sees a label.

**Gates.**
- An agent senses an item through its body: a fixed random map from the item's coordinates to a 32-dimensional observation, plus noise.
- Two agents have the same body if they share that map. A dial γ sets how nonlinear the map is, from a rotation (γ = 0) to strongly scrambling.

**Use shapes items.** An accepted use moves the item. Two rules, as a preregistered factor:
- **Wear:** the item moves a fraction η toward the action's centre. The same for every body.
- **Shaping by eye:** the item moves a small step in the direction that raises the acting agent's own confidence in the action it took, computed through that agent's body and network. This is the theory's "make the world match the model".
- A control has η = 0: nothing is ever built.

## 3. Agents

- A small MLP from observation to 4 outputs, each the predicted probability that that action is accepted. It is trained online from the refusal bit alone (binary cross-entropy on the action taken). The policy picks the action with the highest predicted acceptance, with ε-greedy exploration.
- Fix the learning rate, exploration and run length once, on pilot worlds with their own seed range. The only criterion is that a lone agent reaches 95% acceptance in the unbuilt world. Then freeze them and log them.
- A population is M agents with independent networks acting on the same pool. They never see each other's actions; they meet only through the items.

## 4. Measures

- **Built share:** of the items whose home is in an overlap zone, the share currently sitting outside every overlap zone.
- **Learning cost:** steps for a newborn agent to reach 95% acceptance.
- **Behavioural agreement:** for two agents, the chance-corrected rate of choosing the same action on the same items, reported separately for overlap-home and unambiguous-home items.
- **Representational agreement:** E_K on the penultimate layer, using the Convergence code (`src/`) with its permutation null.
- **New in version 2:**
  - **The natural test set:** fresh items placed at fresh home angles with no shaping, scored without learning. Its **edge items** are those within 10° of an acceptance boundary: the hard cases a built world removes.
  - **Edge accuracy:** the share of edge items on which the agent's greedy action is accepted.
  - **Calibration gap:** on a set of items, the agent's mean predicted acceptance for its greedy action minus the share actually accepted. Positive means overconfident.
  - **Home log loss:** the agent's mean binary cross-entropy on the world it was raised in, scored without learning. This stands in for "how surprised the agent is at home".

## 5. Experiments and predictions (from version 1)

Each prediction needs a numeric criterion in PREREGISTRATION.md before any real run. Mine are below; say in PLAN.md if one can't be measured as written. At least 10 worlds (seeds) per condition, and the world is the unit for statistics.

**E1. Does building emerge?** M agents start in the natural world, under wear.
- **L1:** at steady state the built share is at least 0.3 above the η = 0 control, for every M ≥ 2 tested.
- **Against:** within 0.1 of the control. That would mean agents undo each other's work.

**E2. The upkeep law, from Check R.** Vary M, δ and network width, in a range where the pool isn't exhausted.
- **L2:** the number of built items scales with M and with 1/δ (log–log slopes between 0.75 and 1.25) and not with width (slope within ±0.25).
- **Against:** any slope outside its range.

**E3. Is the benefit the observer's? (The "who benefits" test.)** Newborns learn in a world built by body-A agents, against the natural world.
- **L3:** under shaping by eye, same-body newborns speed up at least 1.5 times as much as different-body newborns. Under wear, the two speed-ups are within 20%.
- **L3b (new):** after a fixed budget of 2,000 steps, under shaping by eye, the drop in home log loss from natural to built world is at least 1.5 times larger for same-body than for different-body newborns. This is the "compresses only for readers with the key" claim.
- **Against:** under shaping by eye, the two speed-ups are within 20%. Building would then be reading off the world's own structure, whoever looks.

**E4. Routes, for agents.** Pairs of newborns raised in the same built world, in two worlds built separately from the same home angles, or in the natural world with η = 0.
- **L4, behaviour:**
  - On overlap-home items, same-world agreement is at least 0.2 above natural-world agreement.
  - On overlap-home items, different-world agreement is no more than 0.05 above natural-world agreement.
  - On unambiguous-home items, all three are at least 0.9 and within 0.05 of each other.
- **L5, representation:** E_K is higher for same-world pairs than for different-world pairs, with the interval above 0.
- **Against:** no same-world advantage on overlap-home items.

**E5. The decay test.** Remove every agent from a built world.
- **L6:**
  - The built share halves in ln 2 / δ steps, within 25%.
  - Newborns raised after 5/δ steps of abandonment show a same-world advantage (as in L4) under 0.05.
  - With the builders frozen but still acting, the built share stays within 10% over the same span.

**E6. Open question, no prediction.** In the natural world with η = 0, do different-body newborns agree on overlap-home items above chance, and how does that change along the dial γ?
- **If agreement is near chance once bodies differ:** this fits "the world only refuses".
- **If it stays high:** the world's geometry is fixing more than its refusals do, and the report should say so plainly. It bears on the open result from the Convergence programme, where label-free models agreed most.

## 5b. The lifecycle experiments (new in version 2)

**Two kinds of raised agent.** Both get the same number of learning steps.
- **Wild-raised:** learns alone in the natural world (η = 0).
- **Built-raised:** learns in a world that M builders keep built, at E1's steady state, with the builders still acting. Run under wear and under shaping by eye separately.

**E8. What gets forgotten.** Agents first reach 95% acceptance in the natural world. Then for T2 steps they keep learning either in a built world (treatment) or in the natural world (control). Network width varies over at least four values, from barely enough for 95% to 16 times that; width is the state cap.
- **L8a, forgetting:** edge accuracy on the natural test set falls by at least 0.10 more in treatment than in control.
- **L8b, the cap:** that extra fall is larger for narrower networks: the narrowest width's extra fall is at least twice the widest's.
- **L8c, freed capacity:** during T2 both groups also meet a second item type whose acceptance depends on a nuisance coordinate. Treatment agents reach 95% on it in at most 0.8 times the steps control agents need.
- **Against:** L8a's extra fall under 0.03. Then a built world doesn't make agents lose what they knew, and "forget" drops out of the lifecycle at this scale.
- **Neon's point to measure, no criterion:** while builders keep up the world, the population's built share holds even as individual agents lose edge accuracy. Report both curves on one figure.

**E9. Fragility.** Built-raised and wild-raised agents (same steps) are put in a world where every item has just snapped back to its home angle, as if upkeep had stopped at once. They keep learning.
- **L9:** in the first 500 steps, built-raised agents' acceptance is at least 0.15 lower than wild-raised agents', and they take at least 1.5 times as many steps to get back to 95%.
- **Against:** a gap under 0.05 in the first 500 steps.

**E10. Naive realism.** For many agents (seeds × the widths from E8 × two training budgets, both kinds of raising), record home log loss, then edge accuracy and the calibration gap on the natural test set.
- **L10a, confident and wrong:** on natural edge items, built-raised agents' calibration gap is at least 0.10 larger than wild-raised agents'.
- **L10b, matched surprise:** among agents binned by home log loss (at least three bins with 10 or more agents of each kind), built-raised agents' natural edge accuracy is at least 0.10 lower in every bin.
- **Exploratory, no criterion:** within built-raised agents, the rank correlation between home log loss and natural edge accuracy. For wild-raised agents this correlation is near-guaranteed, since home is the natural world, so report it only as a reference.
- **Against:** L10a under 0.03. Then low surprise at home transfers as well as it seems to, and the mechanism fails at this scale.

**E7. What gets built (optional, after E8–E10).** Under wear, building follows use, so "frequently used items get built" is coded in. To test the claim that building happens where a variable both recurs and is costly to predict, add a choice: on its turn an agent may shape an item instead of using it, and gets no acceptance that turn. Split the ring into four sectors: frequent or rare use, crossed with an easy or a strongly nonlinear body map.
- **L7:** shaping turns in the frequent-and-hard sector are at least twice as common as in any other sector, and the frequent-and-hard excess is larger than the sum of the two single excesses.
- This needs agents that value future acceptances (a reinforcement-learning objective). Propose the design in PLAN.md before building it; skip E7 if it would take more than a day.

## 6. Ground rules

- Write PLAN.md first, then stop for review. Then PREREGISTRATION.md, committed before any real run. Deviations go in DEVIATIONS.md, logged before they run.
- Unit-test the world: acceptance arcs, drift toward home, both shaping rules, the η = 0 control (items never leave their noise band), the snap-back in E9 and the natural test set (no shaping, no learning while scoring).
- Never tune anything on an outcome measure. The pilot seed range is for the 95% competence criterion only.
- **The pilot (new).** Read `PILOT.md` before PLAN.md. It shows which measures hit a floor or a ceiling at small scale and where the design broke. It isn't evidence for or against any prediction. Real runs use seeds 0–999, which the pilot never touched. If PLAN.md wants to change a threshold above because of the pilot, say so there and give the reason; David decides before PREREGISTRATION.md is committed.
- Every number in a report comes from a saved file. Report failed predictions as plainly as passed ones.
- No models are downloaded and no long jobs are expected; say so in PLAN.md if either changes.

## 7. Deliverables and check-ins

1. PLAN.md, then stop.
2. PREREGISTRATION.md, then E1 on its own. Stop and report, because every later experiment depends on building emerging.
3. E8, E9 and E10 (they share raised agents), then stop and report.
4. E2–E6, then E7 if approved, then REPORT.md with one row per prediction: the logged criterion, the result, and what it is sensitive to.
