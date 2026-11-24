# Brief: learning agents in a world they can shape (version 7.2, frozen 20 Nov 2026)

**Status:** frozen (version 7.2, approved by David on 20 Nov). Start with PLAN.md, then stop for his review.

**For:** a Claude Code agent on David's PC, in the folder "Manifestation Experiments" next to "Convergence Experiments". `CLAUDE.md` here is a copy of that project's standing rules with a "This project" section added at the top, and they apply here too. This task needs no GPU and downloads no models, so the PyTorch-on-GPU and model-cache rules don't bind, but the research-integrity, git and check-in rules do.

**Environment: reuse the Convergence venv, change nothing outside this folder (David's decision, 20 Nov).** The Convergence venv already has numpy, pandas, scipy, torch and pytest; only numba is missing. Reuse it without modifying it:
1. Make a small venv here from the same base Python 3.12: `& "<path to Python 3.12>\python.exe" -m venv .venv`.
2. Add one file, `.venv\Lib\site-packages\convergence_env.pth`, containing the single line `import site; site.addsitedir(r"<path to Convergence Experiments>\.venv\Lib\site-packages")`. Using `addsitedir`, rather than a bare path, also processes that folder's own `.pth` files. The shared packages then load from there, appended after this venv's own, so anything installed here takes precedence.
3. Install only what's missing into this venv, with pip's cache inside this folder: first `.venv\Scripts\python.exe -m pip install --dry-run --no-cache-dir numba`. If pip wants to install or change numpy, scipy, pandas or torch, stop and tell David. Otherwise run it without `--dry-run`.
4. Check and record in PLAN.md: each package's version and where it loads from (`module.__file__`), that `pip check` passes, and that the pilot tests pass (`pilot6/test_world6.py`, which also imports pilot 5's world).
5. If this can't work (for example, if numba won't run with the shared numpy), stop and ask before doing anything else. The fallback is a full venv here, which means downloading a CPU build of torch.

For E_K, copy `src/metrics.py` from Convergence Experiments (commit 9c7dd7d) into this repository and note where it came from, so this project's code reproduces even if the other folder changes. The "This project" section at the top of `CLAUDE.md` lists what must not be touched outside this folder.

**History of this brief.**
- **Version 1 (12 Nov):** the world, E1–E6.
- **Version 2 (20 Nov):** the lifecycle experiments E7–E10, then pilot 1 (`PILOT.md`).
- **Version 3 (20 Nov):** after Neon's first review. One offer plus a skip and a refusal cost; shapeable appearance; controls in E3, E8 and E10. Then pilot 2 (`PILOT2.md`): nobody could read the arc edges, and random bodies had no distinct key.
- **Version 4 (20 Nov):** after Neon's second review. Structured bodies (the angle plus one appearance axis), targeted exploration, three kinds of ensemble, entropy on each body's wild partition, split calibration, update counts. No optimistic initialisation: built-raised agents never visit the edges, so they would keep optimistic starting guesses there. Pilot 3 (`PILOT3.md`): the design works, and built-raised agents err toward refusal near the edges.
- **Version 5 (20 Nov):** after Neon's third review. Its thresholds were committed to git before pilot 4's code was written; pilot 4's report is `PILOT4.md`.
  - **A correction first.** Rechecking pilots 1–2 with calibration split by refused and accepted edges (`pilot4/recheck_p12.py`) shows that their "overconfidence" was false refusal, the same direction as pilot 3. Every pilot so far agrees: built-raised agents carry the built world's boundaries into the unbuilt world.
  - **Neon's mechanism.** Building erases the in-between cases, so a learner that sees only the clusters puts the boundary between them, toward the midpoint 45° from each centre. Where the true edge lies beyond the midpoint (wide arcs), the learned boundary moves inward and the agent wrongly refuses. Where it lies short of it (narrow arcs), the boundary moves outward and the agent wrongly accepts.
  - **The test.** Version 5 gives the four actions unequal arcs, two wide (±60°) and two narrow (±35°), so one world predicts shifts of opposite sign. L10 becomes a signed boundary shift; generic caution can't produce both directions.
  - **Also:** E9's threshold relative to the wild payoff; equal steps as the primary comparison with equal acted turns as the control; wear tightened so built worlds keep only 5–10% edge items; a filtered control matched to the built world's edge exposure; a passive prequential code length; four maps per type in E3; L8c kept only if a capacity bottleneck shows in wild agents; randomized priors kept only if they pass a manipulation check.
  - **Pilot 4 (`PILOT4.md`):** narrow arcs stretched as predicted, but wide arcs didn't shrink, and under passive feedback both arcs shrank. The sign tracked the share of acceptances among the labels the agent learned from. In every condition built-raised boundaries sat further from the truth, and home surprise gave no warning.
- **Version 6 (20 Nov):** after Neon's fourth review. Its thresholds were committed to git before pilot 5's code was written; pilot 5's report is `PILOT5.md`.
  - **Neon's two-pull account.** Two pulls add. A geometric pull G draws each boundary toward the middle of the window that building emptied: wide arcs inward, narrow arcs outward. A label-balance drift L pushes both arcs outward when acceptances dominate the agent's labels and inward when refusals do. In the toward-45° convention, wide = G − L and narrow = G + L. Any single pair of shifts fits this exactly, so the account is tested by changing the label balance within one regime (passive feedback with acceptances reweighted) and checking that G stays positive while L follows the balance.
  - **L10 restated (David confirms before preregistration).** L10a becomes an unsigned shift pooled over the two arc types, with a confidence condition and the home-surprise condition. L10s is the two-pull prediction.
  - **Also:** E10 under four feedback conditions; a symmetric-arc check arm with pilot 3's geometry; L10 read across E2's upkeep sweep, so erasure becomes a dose; β = 10 for randomized priors; E8's filtered control matched on the edge items actually offered; L8a scoped to the narrowest width; L8c dropped; E3's speed-up replaced by early learning-curve area, paired by map; E9's mechanism stated per regime.
  - **Pilot 5 (`PILOT5.md`):** the skip-rule result replicated: built-raised boundaries 3.4–4.7° further from the truth, confident error 3–5 times wild agents', no warning from home surprise. The two-pull account didn't hold up. Balanced labels left the label-balance pull nearly unchanged, and the symmetric world's shift didn't depend on the label balance. The test had two flaws: large weights at s = 0.9 broke wild learners, and reweighting moves the wild comparator.
- **Version 7 (20 Nov):** after Neon's fifth review. Its thresholds were committed to git before pilot 6's code was written; pilot 6's report is `PILOT6.md`.
  - **L10's hierarchy flipped (David confirms before preregistration).** Where no verdicts arrive, where a learner puts its boundary is set by its priors and its exploration, so the location can't be the core prediction. What naive realism requires is that home surprise stays silent while error beyond the built region is held with confidence. So L10a, the primary prediction, is confident error plus log loss on unbuilt edges, with home log loss no higher, under every main feedback condition. The boundary shift becomes a secondary prediction under the skip rule, the theory's own gate (verdicts only for actions taken).
  - **Reweighting replaced by biased offers.** Under passive feedback the offered action is drawn so that it is accepted with probability s. This changes the balance of labels without changing which outcome follows a given angle and action, and every learning step keeps weight 1.
  - **A mechanism contrast (L10m) replaces the two-pull prediction.** Skip rule against biased-offer passive feedback at about the skip rule's share, against unweighted passive feedback, plus a sweep of the skip rule's random-act floor. As Neon showed, biased offers at s = 0.9 starve far-side refusals almost exactly as the skip rule does (about 47:1 against 50:1). So biased offers against passive isolates refusal starvation, and the skip rule against biased offers isolates the agent's own selection of where it tests.
  - **Also:** at least 24 agents per kind per condition, spread over at least 8 independent seeds rather than stacked on a few.
  - **Pilot 6 (`PILOT6.md`):** L10a held under the skip rule and passive feedback (confident error 2.2–3.6 times wild agents', unbuilt-edge log loss 1.8–2.6 times, home log loss lower). L10-shift held under the skip rule. L10m's ε_min prediction held. Biased offers moved wild boundaries 4–5° (design flaw D1), so L10a under them can't be read.
- **Version 7.1 (20 Nov):** after Neon's sixth review. No new pilot.
  - **Biased offers dropped** from L10a and from L10m. A version biased only far from edges may run as an exploratory arm; nothing waits on it.
  - **Ratios defined:** every "at least k times wild" in L10a and L10-dose is the geometric mean, over worlds, of the paired per-world ratio (built ÷ wild, same seed and condition). Never the arithmetic mean of ratios, which a small wild denominator inflates.
  - **A tolerance in the matched-surprise arm only:** there, "home log loss no higher" means built ≤ wild + 0.005 nats, because ε_r is tuned to make them equal.
  - **A smoke test before the real runs:** one pilot-range seed through E1–E10 on the frozen code, since pilot 6 ran only E10.
  - **The edge-band test** (restoring a thin band of unshapeable edge items) is held back for after the freeze. It would need its own dose pilot, and passive agents may learn the boundary from those labels directly.
  - **One threshold left for David:** L10-shift's ≥ 3° (see E10).
- **Version 7.2 (20 Nov, this one): frozen for PLAN.md and PREREGISTRATION.md.** David approved the freeze: L10a primary and L10-shift secondary, L8a scoped to the narrowest width, biased offers dropped, the second-task test dropped. L10-shift's bar was lowered to 2.5° for both world types. Pilot 6 prompted that change, for power reasons, as E10 records. Framing: manifestation and its lifecycle (E1–E9) are the focus; naive realism (E10) is a secondary consequence. Further changes go through PLAN.md and David, as section 6 says.

## 1. Why this exists

**The theory, in three lines.**
- The world supplies only what actually happens; models supply the alternatives. At an agent's boundary (its gate), the world only ever refuses or doesn't.
- Where the world refuses nothing, what agents do is convention. Acting on a convention reshapes the world, so the convention becomes a feature others can read off. The theory calls this manifestation.
- Built features last only while someone keeps them up.

**What this project is about (version 7.2).** The focus is manifestation and its four-step cycle: notice, predict, build, forget. Do acting agents turn conventions into features of the world? Who can read what was built (the builders' key)? What does it cost to offload a regularity onto the world, and what happens when upkeep stops? E1–E9 test this, and the plan and the report should lead with them. Naive realism (E10) is a consequence that helps explain the theory to others, not its headline. Report it fully, but as secondary.

**The lifecycle.** A regularity can give an agent zero surprise three ways: the agent doesn't carve it, the agent predicts it, or the world is held constant so there's nothing to predict. The proposal under test is that these are stages: a lineage notices a variable, learns to predict it, builds the world so it stays put, then stops tracking it. Two consequences can fail:
- **Offloading has a price.** Agents raised in a built world handle the unbuilt world worse, and crash harder when the built structure fails.
- **A mechanism for naive realism.** Low surprise from a built world looks, from inside, like low surprise from a good model. Agents raised in a built world should take their boundaries for the world's. Where building erased the in-between cases, the gate gives no verdicts, so the learner's own defaults fill the gap. Its error there should be larger than a wild agent's and held with confidence, and its low surprise at home should give no warning. Which way the error cuts, and how far the boundary moves, depends on the learner: on its priors and on which verdicts it gets to collect (version 7). The core prediction is the silence and the confidence, not the location.

**What this can't settle, stated in advance.** The difference between "a built thing is one state that compresses well for readers with the builders' key" and active inference's "the niche is the environment's estimate of the organism" is mostly conceptual. Both predict that newcomers with the builders' body benefit more (E3). E3's forestry measures show the reader-relative value of one state directly, but they don't falsify the active-inference reading.

**What has been tested so far.**
- The Convergence programme compared frozen, passive models. It couldn't test any of the above, because none of those models acts.
- Check R, a hand-coded simulation (`check-r-results.md` in the project), derived what the theory's mechanisms imply. Its main lesson: results turned on costs the simulation had to assume. Here, nothing about the outcome may be coded in.

**Prior work to read first (abstracts are enough), and to position against in the report:**
- Hu, Lerer, Peysakhovich & Foerster (2020), "Other-Play for Zero-Shot Coordination", and Treutlein et al. (2021): independently trained agents settle on arbitrary conventions that fail with new partners.
- Cook, Lu, Hughes, Leibo & Foerster (2024), "Artificial Generational Intelligence": cultural accumulation across generations of reinforcement-learning agents.
- Work on stigmergy: coordination by changing a shared environment.
- For E8, one review of catastrophic forgetting and one on loss of plasticity; for E10, one paper on neural networks' overconfidence away from their training data, and Osband, Aslanides & Cassirer (2018), "Randomized Prior Functions for Deep Reinforcement Learning". The report must say what is and isn't ordinary forgetting, plasticity loss or ordinary overconfidence.
- Constant, Ramstead, Veissière, Campbell & Friston (2018), "A variational approach to niche construction", for the framing above.

## 2. The world

Propose exact sizes in PLAN.md; the values below show the intended scale.

**Items.**
- A pool of about 2,000 items. Each has an angle α on a ring and two appearance coordinates (u1, u2).
- Each item has a home value for all three, drawn once. Every coordinate drifts back toward home at the same rate δ, with a little noise.
- **A second item kind (B): dropped in version 6** with L8c (pilot 4: no width learned it).

**What the world refuses.**
- **There are 4 actions, with unequal arcs (changed in version 5).** Centres at 0°, 90°, 180°, 270°. Actions 0 and 2 are **wide**: accepted within ±60° of the centre. Actions 1 and 3 are **narrow**: accepted within ±35°. Every angle is accepted by at least one action. Neighbouring arcs overlap by 5° (for example, 55°–60°).
- **The midpoint between two centres (45° from each) lies inside a wide arc and outside a narrow one.** So a learner that places boundaries between the built clusters should shrink wide arcs and stretch narrow ones. Version 6 called this the geometric pull G. Pilots 4 and 5 found it small and unstable in sign, so version 7 reports it only as a description.
- Each turn an item offers one action, drawn at random. The agent either acts or skips. Payoffs: accepted +1, skip 0, refused −2. Acting breaks even at predicted acceptance 2/3.
- An agent only ever learns "refused" or "not refused" for actions it takes, except in E10's passive arms.
- **Passive feedback (E10 only, version 6):** the agent also learns the offered action's verdict on turns it skips. Payoffs are unchanged.
- **Biased offers (version 7; dropped from the predictions in version 7.1, exploratory only):** under passive feedback, with probability s the offered action is drawn uniformly from the actions that accept the item, and otherwise uniformly from those that refuse it. Every angle is accepted by some action, so the draw is always possible. This changes which labels the agent sees, not which outcome follows a given angle and action.
- **Symmetric-arc check world (E10 only, version 6):** all four arcs ±60°, as in pilot 3, so neighbouring arcs overlap by 30°.
- **Random refusal ε_r (E10 only):** the world refuses an otherwise accepted action with probability ε_r, **for the learner only (changed in version 4)**. In pilot 2 it also hit the builders, which thinned their shaping.

**Gates (changed in version 4).**
- There are two body types:
  - **type 1** reads (cos α, sin α, λ·u1, kind);
  - **type 2** reads (cos α, sin α, λ·u2, kind).
- Each body is a fixed random nonlinear map from those inputs to a 16-dimensional observation, plus noise. Angle and appearance are entangled in the observation, and neither type reads the other's appearance axis. The two types are symmetric by construction, so they have the same wild difficulty in distribution.
- **λ, the legibility scale, is chosen on wild agents only, before any other run:**
  - wild agents reach edge accuracy ≥ 0.85 and a decisive ratio ≥ 0.6, for wide and narrow actions separately;
  - wild agents' learned half-widths lie within ±3° of the true ones, so the boundary-shift measure starts unbiased.
  - Version 4's appearance slow-down criterion is dropped: E3 separated without it.
- Two agents have the same body if they share the map, not just the type.

**Use shapes items.** An accepted use moves the item. Two rules, as a preregistered factor:
- **Wear:** the angle moves a fraction η toward the action's centre. The same for every body. η = 0.15 (version 6): under unequal arcs it leaves about 1% edge items, as few as shaping by eye at κ = 1 leaves, and no η tested in pilot 4 left 5–10%. How much building erased is varied instead along E2's upkeep sweep (L10-dose).
- **Shaping by eye:** the item moves a small step in each coordinate in the direction that raises the acting agent's own predicted acceptance of the action it took, computed through its body and network. The angle steps κ·s_u·20° and appearance steps s_u. A coordinate the agent's body doesn't read has zero gradient and doesn't move (unit-tested).
- **A control with no shaping:** nothing is ever built.

## 3. Agents

- **Network.** A small MLP from observation to 4 outputs, each the predicted probability that that action is accepted. It is trained online from the refusal bit on acted turns (binary cross-entropy, plain SGD). Initialisation is neutral (zero output bias).
- **Policy for single networks (changed in version 4).**
  - Act when predicted acceptance is above 2/3.
  - Margin-targeted exploration: when the prediction is within ±m of 2/3, act anyway with probability ε_m.
  - A global floor: act with probability ε_min = 0.02 regardless.
  - No optimistic initialisation (see the history above).
- **Ensembles (E10).** K = 5 members trained together by Thompson sampling: each turn one member, drawn at random, decides; each member learns from a verdict with probability 0.5 (a bootstrap mask); the same ε_min floor applies. Three kinds:
  - **standard:** members differ in seed and mask only;
  - **diverse:** member widths from a fifth of the base width to twice it, plus the bootstrap;
  - **randomized prior:** each member adds the output of its own fixed, untrained random network, scaled by β, to its logits, so that members disagree where there is no data. β = 10 (version 6). In pilot 4 its member spread on unbuilt edges was 3.60 times the spread at the centres, against 1.96 for standard ensembles; β = 1 passed the old rule (2.32) but barely differed from standard. Real runs repeat this manipulation check at β = 10 and drop the arm if the ratio falls under 2, saying so.
- **Evaluation.** Always greedy, without exploration; ensembles use the member mean.
- **Fixed once, then frozen.** Set the learning rate, m, ε_m and run length on pilot worlds with their own seed range, using competence alone (≥ 0.9 for a lone wild agent within the budget). Then log them.
- **Populations.** M agents with independent networks act on the same pool. They meet only through the items.

## 4. Measures

- **Built share (changed in version 5):** of the kind-A items whose home angle lies within 10° of an arc boundary, the share currently sitting more than 10° from every boundary.
- **Competence:** on held-out offers, the agent's mean payoff divided by the optimal mean payoff.
- **Learning cost:** steps to competence 0.9, checked every 250 steps from step 0.
- **Early learning-curve area (version 6):** mean competence over the first 5,000 steps, checked every 250, minus the same newborn map's mean in the natural world. Pairing by map cancels map difficulty, which dominated steps-to-criterion in pilot 4.
- **Labeled acts (new):** every learning curve is reported against both environment steps and acted turns. Acted turns are gradient updates; skipped turns teach nothing.
- **The natural test set:** fresh items at fresh home values, never shaped, scored without learning.
  - **Edge items** are those within 10° of an arc boundary.
  - **Decisive edge turns (per action type in version 5):** for each action, items within 10° of that action's own arc edge, offered that action, sampled 50/50 just inside its arc (optimal: act, +1) and just outside (optimal: skip, 0). Reported separately for wide and narrow actions.
- **Boundary shift (new in version 5, the measure for L10):**
  - For each action, sweep the angle outward from its centre on both sides in 1° steps, average the agent's predicted acceptance over 20 random appearance values (with observation noise), and find where it first falls below 0.5. That angle, averaged over both sides, is the **learned half-width**.
  - The **shift toward the midpoint** is the learned minus the true half-width, signed so that positive means toward 45°: 60° − learned for wide actions, learned − 35° for narrow ones.
  - L10 compares built-raised with wild-raised agents, so any bias common to both cancels.
- **Edge accuracy:** on decisive edge turns, the share where the prediction for the offered action, thresholded at 0.5, matches the world. Also report it at the decision threshold 2/3.
- **Calibration, split:**
  - **False confidence on refused edges:** mean predicted acceptance of the offered action on decisive turns where it is refused.
  - **Underconfidence on accepted edges:** 1 minus the mean predicted acceptance where it is accepted.
  - Both are reported for wide and narrow actions separately. The prediction is underconfidence on wide actions and false confidence on narrow ones.
  - Mean confidence and accuracy are logged separately, and the old calibration gap is kept as a summary.
- **Decisive payoff:** mean payoff per decisive edge turn under the greedy policy, reported with its two parts: **act rate** and **refusals per act**.
- **Label acceptance share (version 6):** the share of acceptances among the labels an agent learned from, for wide and narrow actions separately.
- **Unbuilt-edge log loss (version 7):** on the same edge items as confident error, the mean binary cross-entropy of the predicted acceptance, in nats like home log loss. It is a proper scoring rule, so it needs no confidence cutoff.
- **Refusal sources (version 7):** among the refusal labels an agent learned from, the share on items within 20° beyond the action's own edge (near side) and on items further away (far side), and the ratio of acceptances to far-side refusals, per action type.
- **Confident error (version 6):** on natural test items within 10° of an arc edge, offered an action whose own edge is that near, the share where the prediction is confidently wrong: predicted acceptance ≥ 2/3 on a refused item (the agent would act and pay −2) or ≤ 1/3 on an accepted one (it would skip a sure +1).
- **Home log loss:** mean binary cross-entropy against the world's true acceptance probabilities, including ε_r where it applies, scored without learning.
- **Forestry entropy (changed in version 4):**
  - For each body, fit k-means with K = 32 on its noise-free observations of the natural pool, never on a built world.
  - The entropy of a world state under that body is the entropy, in bits, of how the state's kind-A items fall into those 32 cells.
  - The **drop** is the natural pool's entropy minus the built state's.
- **Code length (changed in version 5, passive):** a newborn is shown a fixed stream of 8,000 random (item, action) pairs from a frozen world state. There is no skip and no exploration. At each step its prediction is scored (−log2 of the probability it gave the true outcome) and it then learns the true outcome. The total, in bits, is the prequential code length of that state under that body. L3d compares paired savings (natural minus built, for the same newborn map), not ratios of totals.
- **Builders' appearance gradient (new):** mean |∂p/∂u_b| of each builder over the warm-up, to confirm that shaping by eye can move appearance at all.
- **Second-task curve (dropped in version 6, with L8c):** the area under the competence curve on kind-B items over T2. Also a linear probe on the frozen hidden layer at the end of T2, and the weight norms, to separate representation from plasticity loss. **Kind-B items follow a harder rule (version 5):** eight 45° sectors of their own angle, each action accepted in two opposite sectors, so the rule needs more capacity than the main ring.
- **Behavioural agreement** and **representational agreement (E_K)** as in version 1.

## 5. Experiments and predictions

Each prediction needs a numeric criterion in PREREGISTRATION.md before any real run. Mine are below; say in PLAN.md if one can't be measured as written. At least 10 worlds (seeds) per condition, and the world is the unit for statistics.

**Sanity checks, not predictions.** The rules guarantee these, so they verify the code and report no finding:
- E1 under wear: clustering whenever use pulls faster than drift returns.
- E2 under wear: the built share follows the drift–pull balance.
- E5's halving time of ln 2 / δ after everyone leaves.

**E1. Does building emerge under shaping by eye?** M agents start in the natural world, from random initialisation.
- **L1:** at steady state the built share is at least 0.3 above the no-shaping control, for every M ≥ 2 tested, at κ = 1. At κ = 0.25, where the angle barely moves, the criterion is instead that the builders' appearance coordinate moves at least 0.3 from home on average.
- **Against:** within 0.1 of the control (κ = 1), or under 0.1 of movement (κ = 0.25).

**E2. The upkeep law, under shaping by eye.** First sweep upkeep strength (M, s_u and δ, log-spaced) so the built share runs from about 0.1 to 0.9, and choose the test range where it's unsaturated.
- **L2:** within that range, the number of built items scales with M and with 1/δ (log–log slopes between 0.75 and 1.25) and not with width (slope within ±0.25).
- **Exploratory:** whether there is a minimum population or competence below which shaping by eye fails to hold any structure.

**E3. Who benefits?** Builders of one type (say type 1) build a world from scratch under shaping by eye at κ = 0.25. Newborns then learn in it, against the natural world:
- **same body:** the builders' exact map;
- **same type:** other type-1 maps, at least three;
- **outsider:** type-2 maps, at least four.

Half the worlds are built by type-2 builders instead, so map difficulty averages out.
- **L3 (changed in version 6):** same-body newborns' gain in early learning-curve area is at least 1.5 times outsiders'. Under wear, the two gains are within 20%. Steps-to-criterion speed-up is reported alongside.
- **L3b:** after 2,000 steps, the drop in home log loss from natural to built world is at least 1.5 times larger for same-body newborns.
- **L3c, forestry (the primary E3 prediction in version 6):** for the frozen built state, the forestry entropy drop is at least 0.5 bits under the builders' body and at most half as large under the outsider's.
- **L3d, code length (passive, version 5):** the reduction in code length from natural to frozen built state is at least 1.5 times larger for same-body newborns than for outsiders.
- **Exploratory:** where same-type newborns fall. Pilot 3 hinted that entropy separates by the axis read (any map reading the shaped axis sees the compression) while learning speed separates by the exact map. Neon's account: entropy is roughly invariant to smooth re-mappings, but learning speed depends on how linearly separable the clusters are in that map's own observation space. If it holds, the two measures track different things: compression at the gate versus fit to one reader's encoding.
- **Optional, later:** a spillover arm, where shaping u1 also perturbs u2, as in monoculture forestry, to test whether tidying for one reader makes the world messier for another.
- **Against:** the outsider's entropy drop at least 80% of the builders', or the area gains within 20%.

**E4. Routes, for agents.** As in version 1: pairs of newborns raised in the same built world, in two worlds built separately from the same home values, or in the natural world.
- **L4, behaviour:**
  - On overlap-home items, same-world agreement is at least 0.2 above natural-world agreement.
  - On overlap-home items, different-world agreement is no more than 0.05 above natural-world agreement.
  - On unambiguous-home items, all three are at least 0.9 and within 0.05 of each other.
- **L5, representation:** E_K is higher for same-world pairs than for different-world pairs, with the interval above 0.

**E5. Decay.** Remove every agent from a built world. Apart from the sanity check:
- **L6:** newborns raised after 5/δ steps of abandonment show a same-world advantage (as in L4) under 0.05.

**E6. Open question, no prediction.** In the natural world, do newborns of the two body types agree on overlap-home items above chance?

## 5b. The lifecycle experiments

**Two kinds of raised agent.** Both get the same number of environment steps, which is the primary comparison. Equal acted turns is the control (changed in version 5, following Neon: making equal updates primary would hide a real effect of building, since built worlds are safer to act in).
- **Wild-raised:** learns alone in the natural world.
- **Built-raised:** learns in a world that M builders keep built, at E1's steady state, with the builders still acting.

Run E8–E10 under wear and under shaping by eye at κ = 1. At κ = 0.25 the angle barely moves, so that rule is for E3.

**Update-count control.** One extra wild-raised arm gets as many acted turns as the built-raised agents, by running longer. E8–E10's conclusions are reported both ways; if they differ, say so.

**E8. What gets forgotten.** Agents first train in the natural world until edge accuracy levels off: a change under 0.01 between two successive 2,000-step windows. Then for T2 steps they keep learning in one of three worlds:
- **Treatment:** a built world, with its builders still acting.
- **Filtered control (matched on what is offered, version 6):** the natural world, with edge items offered only at the rate the built world offers them. The sampler is redrawn every 50 steps, because drift carries natural items into the edges between draws (pilot 4: 3.4% offered against a 1.2% target). The two arms' offered edge shares should be within 0.5 points. This is the same shift in what the agent sees, with nothing built.
- **Natural control:** the natural world.

Kind-B items and L8c are dropped (version 6): in pilot 4 no width learned the second task, so there was no capacity bottleneck to free. Widths run from the smallest reaching the legibility targets to 16 times that, in at least four steps.
- **L8a, forgetting (scoped in version 6):** at the narrowest width, edge accuracy falls by at least 0.10 more in treatment than in the natural control. Wider networks aren't expected to forget as much; L8b carries the width dependence. This scoping follows a pilot result and is David's decision.
- **L8a′, building versus shift:** the expected result is that treatment and the filtered control fall within 0.03 of each other. Then building matters as the cause of the shift that makes ordinary forgetting safe, which is the claim. If treatment falls at least 0.05 more, report it as a building-specific effect that needs explaining. Treatment's and the filtered control's acted-turn counts should be within 10%; report them.
- **L8b, the cap:** the narrowest width's extra fall is at least twice the widest's.
- **Against:** L8a's extra fall at the narrowest width under 0.03.
- **Measure, no criterion:** while builders keep up the world, the population's built share holds even as individual agents lose edge accuracy. Report both curves on one figure.

**E9. Fragility.** Built-raised and wild-raised agents are put in a world where every item has just snapped back to its home values, as if upkeep had stopped at once. They keep learning.
- **L9 (relative in version 5):** over the first 500 steps, built-raised agents' decisive payoff is at least 30% below wild-raised agents'. Wide and narrow actions are pooled 50/50.
- **L9 mechanism, no criterion (per regime in version 6):** where an agent's arc stretched, more refusals per act (−2s); where it shrank, a lower act rate (missed +1s). Report it per arc type and per feedback condition, beside that agent's shift.
- **L9b:** the jump in log loss at the snap is at least 0.1 nats larger for built-raised agents.
- **Against:** a decisive-payoff gap under 10% of the wild payoff.

**E10. Naive realism (changed in version 7).** For many agents (seeds × the widths from E8 × two training budgets × both kinds of raising × feedback condition), record home log loss, confident error, unbuilt-edge log loss, the boundary shift, the label acceptance share, the refusal sources and the edge measures on the natural test set. Use at least 24 agents per kind per condition, spread over at least 8 independent seeds (in pilot 6's variance terms, a built-minus-wild standard error of about 0.38° on the pooled shift). Pilots 5–6 found an agent-to-agent sd of about 0.5–0.6° in the pooled shift and a world-to-world sd of about 0.4–0.7°, so worlds matter as much as agents. Pilot 5's first estimate, 1.7°, included a broken arm.
- **The two main feedback conditions (version 7.1):**
  - the skip rule, as everywhere else in this brief (ε_min = 0.02);
  - passive feedback.

  Biased offers were dropped after pilot 6: they moved wild boundaries 4–5°, because offer density near an edge acts like a reweighting on what the agent sees.
- **The mechanism arms:** the skip rule with ε_min = 0.1 and 0.3.
- Built minus wild always compares agents raised under the same condition, so whatever lean a condition gives every learner cancels.
- **L10a, silent home, confident error (primary; changed in version 7).** Under each main condition, with ratios as geometric means of paired per-world ratios:
  - built-raised agents' confident-error rate is at least 1.5 times wild-raised agents';
  - their unbuilt-edge log loss is at least 1.5 times wild-raised agents';
  - their home log loss is no higher than wild-raised agents'.
- **L10a-matched, matched surprise:** built worlds get a learner-only random refusal rate ε_r, chosen on pilot seeds using home log loss only, in steps of 0.01, so that built-raised agents' home log loss lands within ±20% of wild-raised agents'. Among agents binned by home log loss (at least three bins with 10 or more agents of each kind), L10a's two error conditions hold in every bin, and built home log loss is at most wild's + 0.005 nats.
- **L10-shift, the boundary moves (secondary, skip rule only; version 7).** Under the skip rule, built-raised agents' unsigned boundary shift, averaged over the two arc types, exceeds wild-raised agents' by at least 2.5°. Restricting this to the skip rule follows pilot 5, where passive feedback gave 1.2–1.7°. The bar was 3° until version 7.2; it was lowered because pilot 6 put wear at 3.06°, which a true mean that close would miss by chance about half the time at 8 seeds × 3 agents. Both changes followed pilot results and are stated as such.
- **L10-dose, erasure as a dose (skip rule):** at three upkeep levels in E2's unsaturated range, both L10-shift's pooled difference and the confident-error ratio rise with the share of edge items building removed, across the three levels.
- **L10m, what sets the direction (a preregistered contrast; version 7).** The mean stretch is the average amount by which built-raised arcs are wider than wild-raised ones, over both arc types, in degrees (positive means wider). In the toward-45° convention it equals (narrow − wide)/2.
  - **Prediction:** under the skip rule the mean stretch falls as ε_min rises from 0.02 to 0.1 to 0.3. Restoring far-side refusals should undo the stretch.
  - **Also reported (no criterion):** raw stretch beside built minus wild, the near-side and far-side refusal shares under each condition, and the rank correlation between built-raised agents' raw stretch and acceptances per far-side refusal across conditions (pilot 6: 1.00 in both worlds). G and L from version 6 are reported only as descriptions.
- **L10a-ens, the overconfidence controls:** repeat L10a and L10-shift with the ensembles, under the skip rule and passive feedback. Pre-stated readings:
  - If all three kinds meet L10a, the effect isn't specific to single networks.
  - If only the randomized-prior ensemble fails L10a's error conditions, while its shift and its E9 deficit remain, prior humility trades confident error for timid blindness. The verdicts never arrived, so the prior decides (the briefing's Theorem 3.1).
  - If all three fail them, the effect needs a learner with no sense of where it is extrapolating, and the report says so.
- **Symmetric-arc check (exploratory):** in the symmetric-arc world, raise wild and built (wear) agents under the two main conditions. Report each shift with its label share, the refusal sources and the item-angle histogram. Pilot 5 found every arc moving 2–3° inward under every condition; wear keeps 23.7% edge items there, so it's a partial gap.
- **Exploratory:** within built-raised agents, the rank correlation between home log loss and confident error.
- **Against:**
  - L10a: in either main condition, confident error or unbuilt-edge log loss no higher than wild agents', or home log loss higher.
  - L10-shift: the pooled difference under 1° under the skip rule.
  - L10m: the mean stretch not falling as ε_min rises.

**E7. What gets built (optional, after E8–E10; changed in version 4).** One agent with its own private items, so there are no free riders.
- **All items share one home-angle distribution near edges.** Items differ in two ways:
  - **encounter frequency:** high or low;
  - **prediction difficulty:** low or high observation noise.

  So building does the same physical work in every cell, and only its payoff differs. This avoids coding in "don't build what's already central".
- **The choice:** on each visit the agent chooses Act or Build-and-act. Building costs c now and pulls the angle toward the centre of the agent's predicted-best action. The choice is learned with a simple value estimate (TD(0) or a bandit over the item's observable context).
- **L7:** build choices in the frequent-and-noisy cell are at least twice as common as in any other cell, and its excess over the base rate is larger than the sum of the two single excesses.
- Propose the details in PLAN.md; skip E7 if it would take more than a day.

## 6. Ground rules

- Write PLAN.md first, then stop for review. Then PREREGISTRATION.md, committed before any real run. Before the first real seed, smoke-test E1–E10 end to end on one pilot-range seed with the frozen code, and check every logged column is filled. E1–E9 have never run together in the pilots; pilots 5–6 ran only E10. Deviations go in DEVIATIONS.md, logged before they run.
- Unit-test the world: the unequal acceptance arcs, the symmetric-arc world, passive updates, biased offers (that the acceptance share hits its target), the label acceptance share, the boundary-shift measure (on a hand-built predictor with known half-widths), the payoff and skip rule, drift toward home in every coordinate at one rate, both shaping rules (including that shaping by eye never moves an unread coordinate), the no-shaping control, learner-only random refusal, the matched filtered sampler, Thompson sampling and the bootstrap mask, the randomized priors staying fixed, the snap-back, and the natural test set (no shaping, no learning while scoring).
- Never tune anything on an outcome measure. The pilot seed range is for the legibility scale (wild agents only), the exploration settings (competence only) and ε_r (home log loss only).
- **The pilots.** Read `PILOT.md` to `PILOT6.md` before PLAN.md; the later reports supersede earlier readings where they differ. They show where the design broke and which measures hit a floor or a ceiling at small scale. They aren't evidence for or against any prediction. Real runs use seeds 0–999, which the pilots never touched. If PLAN.md wants to change a threshold because of a pilot, say so there and give the reason; David decides before PREREGISTRATION.md is committed.
- Every number in a report comes from a saved file. Report failed predictions as plainly as passed ones.
- No models are downloaded and no long jobs are expected; say so in PLAN.md if either changes.

## 7. Deliverables and check-ins

1. PLAN.md, then stop.
2. PREREGISTRATION.md, then the legibility and exploration settings, the sanity checks and E1. Stop and report, because every later experiment depends on building emerging under shaping by eye.
3. E8, E9 and E10 (they share raised agents), then stop and report.
4. E2–E6, then E7 if approved, then REPORT.md with one row per prediction: the logged criterion, the result, and what it is sensitive to.
