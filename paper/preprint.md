---
title: "Fitting the World to the Model:<br>What It Costs, and Who It Serves"
pagetitle: "Fitting the World to the Model: What It Costs, and Who It Serves"
author: "David Shen"
date: "Preprint, November 2026"
---

## Abstract

Agents don't only adapt to their world; they also reshape it to suit what they already do. Active-inference work on niche construction shows that persistent changes leave the world fitted to its users. We ask what it costs, and who pays.

In a small preregistered simulation, agents act on items on a ring, learning only whether each act was refused. Acting reshapes items, by wear or by "shaping by eye" (nudging each item toward what its user predicts).

Four costs appear:

- **Upkeep:** the amount built follows builders × shaping strength ÷ decay, and fades without upkeep.
- **The builders' key:** a built world is 0.63 bits simpler for the builders' body, 0.52 for its type and 0.045 for outsiders; code-length savings are mostly shared.
- **Newcomers:** a built world doesn't erase prior knowledge, but nobody learning there acquires what it no longer shows.
- **A trap:** newcomers are less surprised at home, yet confidently wrong beyond the built region, and earn 42–73% less when the world snaps back instantly (gradual decay: future work). Much of this error comes from the edge cases building removes; missing refusals beyond the edge stretch learned boundaries on top, and under shaping by eye newcomers also learn the builders' marks as a shortcut.

Several predictions failed; we report them plainly.

## 1. Introduction

A learner can reduce its surprise in two ways. It can improve its model of the world, or it can change the world until the model fits. Beavers build dams, people wear paths into lawns, and institutions standardise forms. In each case the world comes to match the habits of its users, and it stays that way for as long as someone keeps it up. We call this **persistent world-fitting**, or fitting the world to the model.

This is not a new idea. In active inference, an agent minimises free energy (roughly, long-run surprise) by perceiving and by acting. Bruineberg, Rietveld, Parr, van Maanen and Friston (2018) simulated an active-inference agent whose walking slowly opens a path through a grid that "remembers" where it went: a desire path. Whether the agent adapts to the grid or the grid to the agent depends on their relative inertia. Constant, Ramstead, Veissière, Campbell and Friston (2018) gave niche construction a variational description in which organism and niche come to infer each other. Both papers show that world-fitting happens and why it pays the fitter. Neither measures what keeping a fitted world costs, nor who else pays. Bruineberg et al.'s environment keeps its changes without upkeep and doesn't model agents who arrive later. Constant et al. name decay, upkeep and negative feedbacks as open problems. Our agents are simple learners trained by gradient descent, not active-inference agents, so this study parallels that work rather than modelling it. Its building rule "shaping by eye" stands for interfaces or curation tuned toward what works for their current users.

A nearby idea is **cognitive offloading**: using the world to hold or process information that would otherwise be held in the head (Risko & Gilbert, 2016). Our results speak to it, but we avoid the term for three reasons.

- **Veterans keep what they knew.** In our simulation, agents who knew the domain before entering a built world kept what they knew; nothing moved out of their heads.
- **Newcomers never acquire it.** Agents who learn inside the built world never acquire what it no longer shows. That is non-acquisition, not offloading.
- **Nothing is stored for every reader.** What the built world "holds" is readable mainly by readers with the builders' kind of body (section 4.2), not by everyone.

**What is expected, and what is new.** Two of our results are expected. The upkeep scaling is what a balance between pushing and decay predicts: if each builder pushes items away from home in proportion to shaping strength and drift pulls them back at rate δ, the amount held away scales with M·s~u~/δ. And outsiders' near-zero entropy drop follows from their bodies, which don't read the coordinate the builders moved. The contribution is the cost ledger, its measurements and a mechanism test. Agents create their own distribution shift (covariate shift) and their own missing verdicts (selective labels), and home surprise gives no sign of either.

**What this paper shows.** We build a toy world where learning agents reshape items by acting on them, and we preregister tests of four costs:

1. the upkeep a built world needs;
2. the reader-specific value of what is built (the builders' key);
3. what newcomers learn inside it;
4. how surprise and error behave when agents raised there meet the unbuilt world.

**Results in brief.**

- Building emerges and follows a simple upkeep law: the amount built scales with builders × shaping strength ÷ decay, and it fades when upkeep stops.
- What is built is much simpler for readers built like the builders than for outsiders, though part of its value is shared.
- A built world doesn't erase what was learned before entering it; it stops anyone learning there from acquiring what it no longer shows.
- Agents raised in it are less surprised at home yet confidently wrong beyond the built region, and they fare much worse when the world snaps back. Much of this error comes from the edge cases building removes; missing refusals stretch learned boundaries on top, and under shaping by eye an appearance shortcut adds a smaller part.
- Several predictions failed (section 4.5), and one design limit rules out any claim about arbitrary conventions.

## 2. World and agents

Figure 1 shows the set-up; Appendix B lists every parameter.

**Items.** A world holds 600 items. Each has an angle on a ring and two appearance coordinates (u1, u2), each with a home value drawn once. Every coordinate drifts back toward home at rate δ = 0.0005 per step, with a little noise.

**What the world refuses.** There are four actions, centred at 0°, 90°, 180° and 270°. An action is accepted if the item's angle lies within its arc and refused otherwise. We use two geometries:

- **Symmetric arcs (±60°):** a third of the ring is overlap. These carry the building experiments (E1, E2, E4–E6).
- **Unequal arcs:** two wide (±60°) and two narrow (±35°), so overlaps are only 5° wide. These carry the experiments on readers and newcomers (E3, E8–E10).

**The gate.** Each turn, an item is drawn and offers one action, at random. The agent either acts or skips: an accepted act earns +1, a refused act −2, a skip 0. An agent learns only whether the actions it took were refused. We call this the **skip rule**. In some arms (**passive feedback**) agents also learn the verdict on offers they skip.

![Figure 1. **The world.** (A) Items sit on a ring, and four actions are each accepted inside an arc. Experiments use one of two geometries: symmetric arcs (±60°), where neighbouring arcs overlap by 30° and a third of the ring is accepted by two actions; and unequal arcs (±60° for actions 0 and 2, ±35° for actions 1 and 3). (B) Building empties the overlap. Share of items per 5° bin, folded onto 0–90°, in a world kept up by wear (symmetric arcs; mean of 10 worlds), against the uniform natural world. (C) The cycle studied: builders fit the world and keep it up; newcomers learn inside it; when upkeep stops, items drift back home.](figures/fig1_schematic.png)

**Bodies and reader axes.** An agent sees an item through its **body**: a fixed random nonlinear map from (angle, one appearance coordinate) to a 16-number observation, plus noise. Type-1 bodies read u1, type-2 bodies read u2, and neither reads the other's coordinate. Each world has four bodies of each type.

**Agents.** Each agent is a small network (one hidden layer of tanh units) that outputs, for each action, the probability that it will be accepted. It learns online by gradient descent. It acts when the predicted acceptance exceeds 2/3 (the break-even point) and explores a little near that threshold.

**Building.** An accepted act moves the item, under one of two preregistered rules:

- **Wear:** the angle moves 15% of the way toward the action's centre. This is the same for every body.
- **Shaping by eye:** the item moves a small step in each coordinate in the direction that raises the actor's own predicted acceptance, computed through its own body. The step size s~u~ is the shaping strength, and κ sets the mix of angle and appearance movement: κ = 1 moves both, κ = 0.25 moves mainly appearance. A coordinate the body doesn't read never moves.

**Builders and newcomers.** Six builders sharing one body learn and build for 30,000 steps, starting from random weights; then frozen copies keep acting, so the world stays built. A **newcomer** is any agent that learns the domain inside a built world, whether it starts untrained or partly trained. Agents raised alone in the unbuilt world are called **wild**.

**A design limit.** Agents never choose between actions; they act or skip on one offer. So building here is items draining out of overlap zones toward whichever action happened to be used. It isn't an arbitrary convention that a population settled on, and the design can't test conventions (section 4.5).

**Settings by experiment.** E8–E10 use shaping by eye at κ = 1; only E3 uses κ = 0.25. Table B1 (Appendix B) lists each experiment's geometry, building rule, network width, steps and feedback.

## 3. Preregistration and analysis

**Preregistration.** The predictions, thresholds and analysis were committed before any real run, at commit `47da90e` (PREREGISTRATION.md). Tuned settings and the frozen code followed at `1433ed6` (its section 11) and `404476d` (code). The repository was private until this preprint was released, so these commit times rest on the author's local record; before release the history was rewritten only to change the author name and remove local working files (HISTORY.md).

**Statistics.**

- **Seeds:** real runs use seeds 0–9, never touched by the pilots or by tuning.
- **Unit:** the world (seed) is the unit of analysis. Agents within a world are averaged first.
- **Criteria** are judged on point estimates exactly as preregistered, with 95% bootstrap intervals over worlds (10,000 resamples) given in brackets. With 10 worlds per condition these intervals are approximate.
- **Ratios** ("built-raised ÷ wild") are geometric means over worlds of paired per-world ratios.
- **Primary cell** for the newcomer experiments: network width 32 and 60,000 training steps. Other widths (4–64) and a 30,000-step budget are secondary.

Four deviations were logged before the runs they affect, and one exploratory arm was added after review (Table 1).

*Table 1. Deviations from the preregistration (full text in DEVIATIONS.md).*

| | Deviation | Effect on this paper |
| --- | --- | --- |
| D1 | The legibility check (wild agents read arc edges well) wasn't met on every tuning seed. Lowering the appearance scale λ didn't help, so λ = 0.4 and 60k steps were kept. | Borderline edge legibility: mean edge accuracy 0.84–0.90 against a 0.85 target. Read E8–E10 with this in mind. Width ladder 4–64. |
| D2 | No 3 × 3 block of unsaturated cells existed for E2, so the law is fitted on all 29 eligible cells, with shaping strength as a covariate. | Section 4.1 explains why the covariate is needed. |
| D3 | An added confirmatory arm: E8 with fixed 60k-step pretraining, run because the preregistered pretraining stopped early. | Gives the "fully trained" newcomer case. |
| D4 | The passive-feedback noise level for the matched-surprise test was retuned on 20 more tuning seeds. | The test still couldn't be run (section 4.5). |
| D5 | Exploratory and post hoc, after review: newcomers with an outsider body in the E10 worlds. Logged in DEVIATIONS.md before running, but committed together with its results, so git doesn't timestamp the order. | Tests an appearance shortcut (section 4.4); changes no preregistered result. |

## 4. Results

### 4.1 Building and its upkeep (E1, E2, E5)

**Building emerges.** With shaping by eye in the symmetric world, the **built share** is the share of items whose home lies in an overlap zone that now sit outside every overlap. It rises far above the no-building control for every population tested: +0.43, +0.81, +0.93 and +0.96 above control with 2, 4, 6 and 12 builders (all pass the 0.3 bar; E1). When shaping mostly moves appearance (κ = 0.25), the builders' read appearance coordinate moves 0.37–1.02 from home, while the coordinate they don't read stays at drift level (0.051).

::: {.sidefig}
![Figure 2. **The upkeep law (E2).** Built items at steady state against builders × shaping strength ÷ decay, log–log, shaping by eye in the symmetric world. One point per cell (mean over 10 worlds, bars ±1 s.d.), coloured by shaping strength. Dashed line: a single-predictor fit, slope 0.93 [0.90, 0.97]. The preregistered joint fit gives 0.91 (M), 0.91 (1/δ) and 0.98 (s~u~). The cells were selected inside the unsaturated band, so the fit needs the strength term (text).](figures/fig2_upkeep.png)
:::

**The amount built follows an upkeep law (Fig. 2).** We varied the number of builders M, shaping strength s~u~ and decay δ, and kept the 29 combinations whose built share lay between 0.1 and 0.9 (unsaturated). On those cells, a joint log–log fit gives slopes of 0.91 [0.87, 0.94] on M and 0.91 [0.88, 0.95] on 1/δ, within the preregistered [0.75, 1.25], and 0.98 on s~u~. Restricted to cells between 0.2 and 0.8, the slopes are 0.96 and 0.96. Cells with the same product M·s~u~/δ hold the same number of built items, whatever the mix: 47–49 items at 600 (five cells, M from 1 to 16) and 58–60 at 800 (three cells). Builders' network width adds nothing above the level needed to learn the task (slope 0.001 [−0.015, 0.014], widths 16–64).

**The fit needs the strength covariate.** Without it, the slopes on M and 1/δ collapse to about 0.2 (0.19 and 0.22), and this is caused by how the cells were selected. A cell entered the fit only if its built share stayed inside the 0.1–0.9 band. A cell with more builders or slower decay could do so only with weaker shaping. Among the chosen cells, log M and log(1/δ) each correlate −0.44 with log s~u~. Leaving s~u~ out then lets its opposing effect cancel most of theirs.

**What is built fades without upkeep (E5).** When every agent leaves, built displacement halves in 1,335–1,450 steps, against ln 2/δ = 1,386 for pure drift. After 5/δ steps, newcomers raised in the abandoned worlds show no trace of which world they grew up in: the same-world advantage in agreement is −0.002 (L6, pass).

### 4.2 The builders' key (E3)

Builders on one body built a world under shaping by eye at κ = 0.25, so most of the change is in appearance, along the builders' read axis. We then measured how much simpler the built world is for different readers (Fig. 3). By **reader-relative information** we mean how much structure a particular reader can find in a world, given the fixed map through which it sees the world: the same world state can be highly compressible for one reader and close to random for another. We measure it in two ways below: compression at the gate, from each reader's own view of the items, and the code length a learner with that body needs. (A fuller treatment is in preparation: Shen, *Shannon's Zero*.)

**Compression at the gate is almost all key-specific.** We fit 32 clusters to each body's view of the natural world, then measure the entropy of a world state's items over those clusters. The built state's entropy drops by 0.63 bits for the builders' own body, 0.52 for other bodies of the same type, and 0.045 for outsiders (L3c, pass). So 93% of the builders' drop, and 91% of the same type's, disappears for outsiders.

**Code length is mostly shared.** The prequential code length of the frozen world is how many bits a newcomer needs to encode the verdicts on 8,000 item–action offers while learning. It falls by 531 bits for the builders' body, 497 for the same type and 416 for outsiders. Only 22% of the builders' saving is key-specific, so the 1.5 × criterion fails (1.27 [1.20, 1.34]; L3d).

**Learning measures fall in between.** Same-body newcomers' early learning gain is 1.56 [1.39, 1.73] times outsiders' (L3, pass, narrowly), and their home log loss after 2,000 steps drops 1.73 [1.31, 2.25] times as much (L3b, pass). Both pass on point estimates, as preregistered, but both intervals include the 1.5 bar. Under wear, which every body sees alike, all readers gain alike (entropy drops 0.27, 0.25 and 0.24 bits). Wear also saves more code length than shaping by eye, for every reader (565, 580 and 574 bits; dashed marks in Fig. 3B). That includes the builders' own body (565 against 531 bits). Wear pulls angles straight to arc centres, which helps every reader, whereas at κ = 0.25 shaping by eye moves mostly appearance.

**Useful but illegible to outsiders.** The perceptual and the functional gains split. Outsiders see almost none of the builders' compression (93% of it is key-specific), yet they keep most of the code-length saving (78% of it is shared). The built world is useful to outsiders but illegible to them. The split has a simple source. At κ = 0.25 the angle still moves a little (the built share is 0.60), and every reader can use that; the appearance changes compress the world only for readers of that axis. For the illegibility only, an everyday analogy (not a result): a language or a musical notation is effortless for those raised in it and near-noise to outsiders.

![Figure 3. **The builders' key (E3; shaping by eye at κ = 0.25, unequal world).** (A) Entropy drop of the built state under each reader's body. (B) Code-length saving over 8,000 item–action offers. Bars are means over 10 worlds with 95% bootstrap intervals; dashed marks show the wear control, where nothing depends on the reader.](figures/fig3_key.png)

### 4.3 Newcomers (E8, E9)

We place agents into a built world at three levels of prior knowledge and measure their **edge accuracy**: whether they predict correctly on unbuilt items near an arc's edge, the cases building removes (Fig. 4). The forgetting tests below use network width 4, not the primary width 32: the preregistered forgetting test was set at the narrowest width, where forgetting was expected to be strongest, and the other widths behaved similarly (L8b, section 4.5).

![Figure 4. **Newcomers and fragility.** (A, B) Edge accuracy on entry and after 30,000 steps in a built world (orange: wear; green: shaping by eye) or in the natural control (blue), for fully trained (solid) and partly trained (dashed) agents. Diamonds: untrained newcomers raised for 60,000 steps in the built world, and agents raised wild, for reference. Panels A–B use network width 4, where the preregistered E8 test is judged. Only start and end points are shown, because full trajectories were saved for the built-world arms only. (C) Payoff per decisive edge turn over the first 500 steps after the world snaps back (E9; width 32, 60,000 steps, skip rule), with 95% intervals over 10 worlds.](figures/fig4_newcomers.png)

**Fully trained agents kept what they knew but largely stopped improving.** Agents pretrained for 60,000 steps in the natural world (edge accuracy 0.868) were then moved into a built world for 30,000 steps (D3, an added confirmatory arm). Their edge accuracy ended at 0.885 (wear) and 0.878 (shaping by eye), against 0.897 for the same agents kept in the natural world. The extra fall relative to that control was 0.012 [0.003, 0.019] and 0.019 [0.007, 0.031], below the 0.03 "against" bar of the preregistered forgetting prediction (L8a). A built world doesn't make agents forget what they know. These agents, though, had no internal decay and no second task competing for capacity, so the null says nothing about agents that do.

**Partly trained agents stop learning.** In the preregistered version of the same test, pretraining stopped at a plateau rule that fired early, at edge accuracy 0.704. In the built world these agents kept what they had but rose only to 0.762 (wear) and 0.742 (shaping by eye), while the same agents in the natural world rose to 0.856. By the letter this passes L8a for shaping by eye (0.114 ≥ 0.10). But it is withheld learning, not forgetting. It is also not catastrophic forgetting in the usual sense, interference from conflicting data (Parisi et al., 2019): the built world withheld data rather than contradicting it. Nor did we test loss of plasticity (Dohare et al., 2023); the fully trained agents give no sign of it.

**Untrained newcomers never learn the edges.** Agents raised from scratch for 60,000 steps inside the built world reach edge accuracy 0.761 (wear) and 0.699 (shaping by eye), against 0.887 for agents raised wild (width 4).

**Fragility.** When upkeep stops and every item snaps back home instantly, newcomers raised in the built world earn 42% (wear) and 73% (shaping by eye) less per decisive edge turn than wild agents over the first 500 steps (0.204 and 0.097 against 0.353; L9, pass). Under passive feedback the gap is 31% and 53%. The jump in surprise at the snap is smaller than predicted, 0.056 and 0.095 nats against a 0.1 bar (L9b, fail). Whether that jump is concentrated on edge items is untested, because only whole-world surprise was recorded. Only an instant snap-back was tested; a world that decays gradually is future work (section 5).

### 4.4 Silent home, confident error (E10)

**Silence at home, confidence away from it.** Newcomers raised in a built world are *less* surprised by their home world than wild agents are by theirs. Home log loss under the skip rule is 0.016 (wear) and 0.014 (shaping by eye) against 0.031. Yet on unbuilt edge items they are confidently wrong (predicted acceptance ≥ 2/3 on a refused item, or ≤ 1/3 on an accepted one) far more often (Fig. 5).

![Figure 5. **Less surprise at home, more confident error away from it (E10).** Home log loss (log scale) against confident error on unbuilt edge items, one point per world, for agents raised wild (blue) and newcomers raised in the built world (orange: wear; green: shaping by eye), under the skip rule (filled circles) and passive feedback (open triangles). Width 32, 60,000 steps.](figures/fig5_silent_home.png)

*Table 2. The primary test (L10a) passes in all four cells: both ratios ≥ 1.5, home log loss no higher. Rates: share of edge test items answered with confident error.*

| Feedback | Building rule | Confident error, ratio to wild | Rate, newcomers vs wild | Edge log loss, ratio to wild |
| --- | --- | --- | --- | --- |
| Skip rule | wear | 2.70 [2.46, 2.93] | 14.5% vs 5.4% | 1.95 [1.89, 2.02] |
| Skip rule | shaping by eye | 4.02 [3.58, 4.50] | 21.3% vs 5.4% | 2.58 [2.42, 2.76] |
| Passive | wear | 1.83 [1.59, 2.11] | 10.4% vs 5.6% | 1.78 [1.68, 1.90] |
| Passive | shaping by eye | 2.65 [2.33, 3.04] | 14.9% vs 5.6% | 2.52 [2.28, 2.77] |

How robust this is:

- **Other cells:** the pattern holds in 34 of 40 width × budget cells; the six misses are all at the shorter budget.
- **Equal updates:** it also holds against wild agents given as many learning updates as the newcomers (ratios 2.53 and 1.89 for wear, 3.88 and 2.52 for shaping by eye; confident-error rates 14.5% and 21.3% against 5.8%).
- **Ensembles:** five-network ensembles of three kinds show it too (confident-error ratios 2.26–4.18; rates 9.7–27.3% against 3.6–6.7% for wild ensembles). That includes ensembles with randomized prior functions (Osband et al., 2018), whose members disagree about inputs they have no data for (their spread is 3.7–6.2× larger at edges than at arc centres).

So it isn't specific to single networks. Hein et al. (2019) showed that ReLU networks are confident far from their training data. Their argument is about ReLU networks and these agents use tanh, but the behaviour is alike: the agents are confident where their data have no support. What is new here is that their own world, built by their predecessors, removed that support, and that low surprise at home gives no warning.

::: {.sidefig}
![Figure 6. **More random acts undo the stretch (L10m).** Extra arc stretch of newcomers over wild agents (degrees) against the learner's floor of random acts ε~min~, under the skip rule, with 95% bands over 10 worlds. The builders' policy is the same in every arm.](figures/fig6_mechanism.png)
:::

**Boundaries move.** Under the skip rule, newcomers' learned arc edges sit 3.69° [3.26, 4.14] (wear) and 5.17° [4.59, 5.72] (shaping by eye) further from the truth than wild agents' (L10-shift, pass). This is the **unsigned shift**: how far each learned edge lies from the true one, in either direction. The displacement grows with how much building erased. For shaping strengths removing 7%, 22% and 28% of edge items, the extra shift is 0.18°, 1.53° and 5.17°, and the confident-error ratio 1.03, 1.63 and 4.02 (rates 5.7%, 8.9% and 21.3% against 5.4%; L10-dose, pass). Under passive feedback the extra shift is small (1.41° and 0.88°), while the confident error remains. So under passive feedback, removing the edge items (with, under shaping by eye, the appearance shortcut) gives 1.8–2.7× confident error (10.4–14.9% against 5.6%); under the skip rule, the missing refusals add the stretch and raise it to 2.7–4.0× (14.5–21.3% against 5.4%).

**One mechanism: building empties the margin where agents test themselves (Fig. 6).** Under the skip rule an agent collects verdicts only where it acts. Wild agents keep testing just beyond their own edge: 55% of their refusals on wide actions come from within 20° beyond the edge. In a built world those items are gone, and so are those refusals: 24% (wear) and 11% (shaping by eye) of newcomers' refusals come from that margin. Without refusals from beyond the edge, newcomers' arcs stretch.

We raised the learner's floor of random acts (ε~min~) from 0.02 to 0.1 to 0.3, which brings back refusals from beyond the edge. This test uses a different measure, the **signed outward stretch**: how far learned edges extend outward past the true ones, averaged over the wide and narrow arcs, and negative if the arcs shrink. It is not the unsigned shift above, so 3.98° here and 3.69° there are different quantities. The extra stretch then falls from 3.98° to 1.83° to 0.16° under wear, and from 5.44° to 2.89° to 0.93° under shaping by eye (L10m, pass). Only the learner's ε~min~ changes: the builders keep the same fixed policy in every arm, so the built world doesn't change with the condition. (In a pilot, the builders had inherited the learner's setting, which confounded this test.)

**The stretch is not the whole effect.** Raising ε~min~ removes most of the stretch but not the confident error. The confident-error ratio falls from 2.70 to 1.65 to 1.49 under wear and from 4.02 to 2.83 to 2.52 under shaping by eye (newcomers' rates 14.5%, 10.6% and 9.4%, and 21.3%, 17.5% and 15.1%, against 5.4%, 6.2% and 6.0% for wild agents). Payoff per decisive edge turn at the moment of snap-back stays below wild agents': 51%, 32% and 26% less under wear, and 90%, 66% and 53% less under shaping by eye. (These payoffs are the first point of the E9 measure, scored before any learning after the snap; the 500-step E9 run, which gives 42% and 73%, was made only at ε~min~ = 0.02.) On absolute rates, the newcomers' excess over wild agents that remains under passive feedback is 53% (wear) and 58% (shaping by eye) of the skip-rule excess, and 37% and 57% at ε~min~ = 0.3. For outsider-body newcomers (D5, below), who can't use the appearance shortcut, it is 66% and 50% of their own skip-rule excess. So the missing refusals account for the stretch and part of the error. The rest comes from the missing edge items themselves, as under passive feedback, and under shaping by eye partly from a second mechanism.

**A second mechanism: an appearance shortcut (exploratory, post hoc; D5).** Under shaping by eye the builders move the appearance coordinate their body reads, so in the built world appearance comes to predict acceptance. Same-body newcomers read that coordinate and can learn it as a shortcut: a rule that works in the built world and fails when the world changes (Geirhos et al., 2020). Newcomers with an outsider body can't read it. We raised outsider-body newcomers in the same shaping-by-eye worlds (seeds 0–9, E10 primary cell, unchanged code apart from the body), and wild agents with the same body for comparison. Under the skip rule their confident error is 3.22× [2.88, 3.57] that of their wild counterparts (19.2% against 6.0%), against 4.02× for same-body newcomers (21.3% against 5.4%). Per world, the same-body ratio is 1.25× [1.09, 1.44] the outsider ratio (passive feedback: 2.11× against 2.65×, from rates of 12.7% against 6.1% and 14.9% against 5.6%; 1.26× [1.05, 1.51]). Under wear, which moves only angle, the body makes no difference (0.99× [0.84, 1.16]; passive 0.86× [0.64, 1.14]). So under shaping by eye an appearance shortcut adds about a quarter to same-body newcomers' confident-error ratio, and most of the effect remains without it.

### 4.5 What failed or couldn't be tested

- **L3d failed:** the code-length saving is only 1.27 × larger for the builders' body (section 4.2).
- **L4 failed, and says nothing about conventions.** Newcomers raised in the same built world agree almost perfectly on which action suits an overlap item (Cohen's kappa 0.99–1.00). But naturally raised pairs already agree at kappa 0.88, so the preregistered +0.2 gain was out of reach.
  - Most of that baseline comes from the readout. We read an agent's choice as the action it rates most likely to be accepted, which in an overlap is the nearer arc's.
  - E6 (no prediction; natural world): pairs with different body types agree at Cohen's kappa 0.83 [0.81, 0.85], again mostly the readout.
  - Because agents never choose freely between actions, this design can't produce or detect an arbitrary convention of the kind studied in coordination, where self-play can settle on conventions that fail with new partners (Hu et al., 2020).
  - The representational measure E~K~ does separate worlds built separately (L5, pass: same-world minus different-world E~K~ is +0.34 under wear and +0.76 under shaping by eye). E~K~ asks how many of each item's six nearest neighbours in one agent's hidden layer are also among its six nearest in the other's, rescaled so that 0 is chance and 1 is identical neighbourhoods (Appendix B).
- **L8b failed:** the E8 gap barely depends on network width (ratios 1.11 and 1.30 against 2).
- **No individual forgetting:** the fully trained arm (D3, an added confirmatory arm) is a null for individual forgetting (section 4.3).
- **L9b failed:** the surprise jump at the snap-back is smaller than predicted, and whether it sits on edge items is untested.
- **L10a-matched couldn't be run.** Even with a learner-only random-refusal rate tuned to equalise home surprise (retuned once, D4), the two groups' home log losses never overlapped enough to form three comparison bins.
- **E7 was not run.** It was optional and asked when agents choose to build rather than keep a regularity internal.

## 5. Discussion

**The cost ledger.** Persistent world-fitting has at least four entries:

- **Upkeep:** a built world is held in place by labour, in proportion to builders × strength ÷ decay, and it decays at the drift rate when the labour stops.
- **The builders' key:** most of what building does to a world's structure is readable only by readers built like the builders; outsiders get the shared part.
- **Newcomers:** a built world stops anyone learning there from acquiring what it no longer shows. Those who learned elsewhere keep their knowledge.
- **The trap:** newcomers are less surprised at home and more confidently wrong away from it, much of it because the edge cases are gone, and they pay when the world snaps back.

**Relation to active inference and niche construction.** Because our agents are simple learners, these are parallels to active-inference work, not tests of it. With that caveat, the results extend the agent–niche picture in three directions.

1. **Upkeep.** They make it quantitative, in a regime where changes decay unless kept up. Bruineberg et al.'s environment remembers without cost; here memory is paid for continuously.
2. **Who reads.** Both earlier papers treat fit as a relation between one organism and its niche. Here the same built state fits different readers to very different degrees, so "the niche" is simpler for some bodies than others.
3. **Who arrives later.** Newcomers, agents who learn inside a world others built, are a cost group that neither paper models. Cook et al. (2024) obtain accumulation across generations of reinforcement-learning agents through social learning. Here what passes to later agents is the state of the world, which lets them skip learning what it no longer shows.

**Related ideas in machine learning.**

- **Performative prediction.** Perdomo et al. (2020) study predictions that change the outcomes they predict, and ask when repeated retraining settles at a stable point. Our builders are a performative case in which acts, not published predictions, reshape the inputs. We ask what the settled world costs agents who didn't make it.
- **Selective labels.** Lakkaraju et al. (2017) study decisions where an outcome is observed only for cases the decision-maker acted on. The skip rule is such a setting: an agent learns a verdict only where it acts. Building adds a second selection, because it removes the items that would supply refusals beyond the edge.
- **Model collapse.** Shumailov et al. (2024) show that generative models trained on model-generated data lose the tails of the original distribution, irreversibly. Here the tails, the edge items, are removed through the world rather than through training data, within one generation of builders. Whether the loss compounds when newcomers become builders is untested.

One reading of Constant et al. is that the niche comes to model its constructors. It predicts the builders' advantage in E3 as well, so our results don't separate it from the account tested here.

**Would an information-seeking agent avoid the trap?** An active-inference reader might expect an agent that seeks information to go back and test the edges. No learner can recover items that building removed: passive feedback, which gives a verdict on every offer, still leaves 1.8–2.7× confident error. Home surprise gives no warning. A warning signal does exist: in ensembles with randomized priors, members' spread is 3.7–6.2× larger at unbuilt edges than at arc centres. Our agents don't use it as a warning. Thompson sampling lets the disagreement randomise single acts, but nothing makes an agent look for test cases where its members disagree, and these ensembles still fall into the trap. Whether an agent that acted on its own uncertainty would escape is untested (future work).

**An interpretation (optional; not tested here).** Building compresses the distribution of inputs along the builders' read axes. A newcomer with a matching body learns that compressed distribution well, so it is less surprised at home. Its error away from home then grows with the compression, a standard result for learning under covariate shift. Where building empties the support completely, the newcomer's predictions are set by its priors and its own sampling, not by data. That is the regime the skip-rule results and Fig. 6 describe. The outsider-body arm (D5) adds a second, smaller route for readers of the builders' axis: a shortcut through appearance.

**Limits.**

- The world is a toy: 600 items, tanh networks, 10 worlds per condition.
- Agents act or skip on one offer and never choose between actions, so nothing here bears on arbitrary conventions.
- Edge legibility was borderline (D1).
- Capacity was varied only as network width, over a range where the task was already learnable. Extra width added nothing to building, and the newcomer effects barely changed with width; we make no claim beyond the widths tested.

**Future work.** Five extensions follow from these results and limits:

- **Generations:** newcomers who become the next builders, and whether their errors compound across generations.
- **Agents that act on their own uncertainty:** for example, seeking test cases where ensemble members disagree.
- **Contested worlds:** builders with different bodies shaping the same world, so that no single key fits it.
- **Gradual decay:** newcomers facing a world that decays slowly when upkeep stops, instead of the instant snap-back tested in E9.
- **Capacity and internal decay:** bounded agents choosing between building and keeping a regularity internal, or reinvesting freed capacity in a second task (§6).

## 6. Where this points (speculation)

Nothing here tests capacity. E3's outsiders were outsiders because their bodies didn't match the builders', and the built world stayed useful to them although they couldn't read it. Artifacts built for a different kind of reader, such as machine-made representations that people can't read, could leave people in the same two positions: as outsiders who can use them but not read them, and as newcomers who never learn what they no longer show. Whether a gap in capacity, rather than a mismatch of bodies, carries the same costs is the next question; the experiment that would test it asks when agents build instead of keeping a regularity internal, with builders and readers of different capacity (E7 plus a capacity sweep).

## 7. Reproducibility

All code, preregistration documents, results tables and figures are in the accompanying repository.

- **Licences:** code MIT; text CC BY 4.0.
- **Commits:** preregistration `47da90e`; frozen settings `1433ed6`; frozen code `404476d`.
- **Layout:**
  - `src/world.py`: the world, agents and measures (numba);
  - `src/exps.py`: experiment tasks;
  - `src/metrics.py`: representational agreement (E~K~), copied unchanged from an earlier project;
  - `runs/run.py`: parallel, resumable runner;
  - `analysis/`: one script per experiment;
  - `analysis/figures/make_figures.py`: all figures;
  - `paper/verify_numbers.py`: checks every number in this paper against the saved results.
- **Regenerating everything:**
  - `bash runs/stage2_tuning.sh`
  - `bash runs/stage2_smoke.sh`
  - `python runs/run.py e1 0-9`
  - `bash runs/stage3.sh`
  - `bash runs/stage4.sh`
  - `python runs/run.py e10_outsider 0-9` (D5; this task was added to `src/exps.py` after the frozen code, in commit `62af047`)
  - then the analysis scripts, including `analysis/e10_extra.py` for D5, the ε~min~ extras and the absolute confident-error rates (section 4.4).
- **Cost:** the whole study runs in a few hours on a 16-core desktop CPU, with no GPU.

## Acknowledgments

This study was designed in discussion with AI assistants. Claude Code (Anthropic) wrote and ran the code and drafted the text under the author's direction; the author made all design decisions and is responsible for the content.

## References

- Bruineberg, J., Rietveld, E., Parr, T., van Maanen, L. & Friston, K. J. (2018). Free-energy minimization in joint agent-environment systems: A niche construction perspective. *Journal of Theoretical Biology* 455, 161–178.
- Constant, A., Ramstead, M. J. D., Veissière, S. P. L., Campbell, J. O. & Friston, K. J. (2018). A variational approach to niche construction. *Journal of the Royal Society Interface* 15, 20170685.
- Cook, J., Lu, C., Hughes, E., Leibo, J. Z. & Foerster, J. (2024). Artificial generational intelligence: cultural accumulation in reinforcement learning. arXiv:2406.00392.
- Dohare, S., Hernandez-Garcia, J. F., Rahman, P., Mahmood, A. R. & Sutton, R. S. (2023). Maintaining plasticity in deep continual learning. arXiv:2306.13812.
- Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M. & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence* 2, 665–673; arXiv:2004.07780.
- Hein, M., Andriushchenko, M. & Bitterwolf, J. (2019). Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem. CVPR 2019; arXiv:1812.05720.
- Hu, H., Lerer, A., Peysakhovich, A. & Foerster, J. (2020). "Other-play" for zero-shot coordination. arXiv:2003.02979.
- Lakkaraju, H., Kleinberg, J., Leskovec, J., Ludwig, J. & Mullainathan, S. (2017). The selective labels problem: Evaluating algorithmic predictions in the presence of unobservables. *Proceedings of KDD 2017*.
- Osband, I., Aslanides, J. & Cassirer, A. (2018). Randomized prior functions for deep reinforcement learning. arXiv:1806.03335.
- Parisi, G. I., Kemker, R., Part, J. L., Kanan, C. & Wermter, S. (2019). Continual lifelong learning with neural networks: a review. *Neural Networks* 113, 54–71; arXiv:1802.07569.
- Perdomo, J. C., Zrnic, T., Mendler-Dünner, C. & Hardt, M. (2020). Performative prediction. ICML 2020; arXiv:2002.06673.
- Risko, E. F. & Gilbert, S. J. (2016). Cognitive offloading. *Trends in Cognitive Sciences* 20, 676–688.
- Shen, D. (in preparation). *Shannon's Zero*.
- Shumailov, I., Shumaylov, Z., Zhao, Y., Papernot, N., Anderson, R. & Gal, Y. (2024). AI models collapse when trained on recursively generated data. *Nature* 631, 755–759. Earlier version: arXiv:2305.17493.

## Appendix A. Every preregistered prediction

| Prediction | Criterion | Result | Verdict |
| --- | --- | --- | --- |
| L1, κ = 1 | built share − control ≥ 0.3 for every M | +0.43, +0.81, +0.93, +0.96 (M = 2, 4, 6, 12) | Pass |
| L1, κ = 0.25 | read axis moves ≥ 0.3 | 0.37, 0.60, 0.76, 1.02 | Pass |
| L2 | slopes on M, 1/δ in [0.75, 1.25]; width ±0.25 | 0.91, 0.91; 0.001 (with strength covariate, D2) | Pass |
| L3c | entropy drop ≥ 0.5 bits for builders; outsiders ≤ half | 0.63 vs 0.045 | Pass |
| L3 | early learning gain ≥ 1.5 × outsiders' | 1.56 | Pass |
| L3b | log-loss drop at 2k steps ≥ 1.5 × | 1.73 | Pass |
| L3d | code-length saving ≥ 1.5 × | 1.27 | Fail |
| L4 | same-world − natural agreement ≥ 0.2 on overlap items | +0.115, +0.120 (ceiling 0.123) | Fail |
| L5 | E~K~ same > different world | +0.34, +0.76 | Pass |
| L6 | same-world advantage after abandonment < 0.05 | −0.002, −0.002 | Pass |
| L8a, partly trained (preregistered) | extra edge-accuracy fall at width 4 ≥ 0.10; against if < 0.03 | 0.094 (wear), 0.114 (shaping by eye); at width 32: 0.064, 0.078 | Wear: inconclusive. Shaping by eye: passes by the letter, but it is withheld learning, not forgetting |
| L8a, fully trained (D3, an added confirmatory arm) | same criterion | 0.012 (wear), 0.019 (shaping by eye); at width 32: 0.002, 0.003 | Against |
| L8b | narrowest width's extra fall ≥ 2 × widest | 1.11, 1.30 | Fail |
| L9 | built decisive payoff ≥ 30% below wild | 42%, 73% | Pass |
| L9b | surprise jump at snap, built − wild ≥ 0.1 nats | 0.056, 0.095 | Fail |
| L10a | both error ratios ≥ 1.5, home log loss no higher | Table 2 | Pass (all four) |
| L10a-matched | ≥ 3 home-log-loss bins with ≥ 10 agents each | 2 bins | Not runnable |
| L10-shift | extra unsigned shift ≥ 2.5° | 3.69°, 5.17° | Pass |
| L10-dose | shift and error ratio rise with erasure | 0.18 → 1.53 → 5.17°; 1.03 → 1.63 → 4.02 | Pass |
| L10m | extra stretch falls with ε~min~ | 3.98 → 1.83 → 0.16°; 5.44 → 2.89 → 0.93° | Pass |
| L10a-ens | pre-stated readings | all three ensemble kinds meet L10a | Not specific to single networks |
| E7 | (optional) | not run | — |

## Appendix B. World and agent parameters

*Table B1. Settings by experiment. Unless varied by the experiment (E1, E2: number of builders) or absent (E6: natural world), six builders on one body warm up for 30,000 steps and frozen copies keep the world built.*

| Experiment | Geometry | Building rule (κ) | Network width | Training steps, unless marked | Feedback |
| --- | --- | --- | --- | --- | --- |
| E1: building emerges | symmetric | shaping by eye (κ = 1 and 0.25); no-building control | builders 32 | 60,000 building steps; no newcomers | skip rule |
| E2: upkeep law | symmetric | shaping by eye (κ = 1), s~u~ 0.01–0.3 | builders 32 (16, 64 at one cell) | 60,000 building steps; no newcomers | skip rule |
| E3: builders' key | unequal | shaping by eye (κ = 0.25); wear control | newcomers 32 | up to 20,000; code length over 8,000 offers | skip rule; code length passive |
| E4, E5: agreement, abandonment | symmetric | wear; shaping by eye (κ = 1) | newcomers 32 | 60,000; abandonment 10,000 | skip rule |
| E6: agreement across bodies | symmetric | none (natural world) | 32 | 60,000 | skip rule |
| E8: newcomers with prior training | unequal | wear; shaping by eye (κ = 1) | 4–64 (judged at 4) | pretraining, then 30,000 | skip rule |
| E9, E10: snap-back, confident error | unequal | wear; shaping by eye (κ = 1) | 32 (grid 4–64) | 60,000 (grid also 30,000) | skip rule; passive |

- **Items:** 600 per world. Home angle uniform; home appearance standard normal.
- **Drift:** δ = 0.0005 per step in every coordinate; angle noise 0.05° and appearance noise 0.002 per step.
- **Bodies:** 8 per world (4 of each type). A random tanh map with 32 hidden units and gain 1.5 to a 16-dimensional observation; observation noise 0.01; appearance scale λ = 0.4.
- **Agents:**
  - one tanh hidden layer (width 32 unless stated), four sigmoid outputs, zero initial output bias;
  - plain SGD on binary cross-entropy, learning rate 0.05;
  - policy: act if predicted acceptance > 2/3; within ±0.2 of 2/3 act with probability 0.6; random-act floor ε~min~ = 0.02.
- **Payoffs:** +1 accepted, −2 refused, 0 skip.
- **Building:**
  - wear η = 0.15;
  - shaping by eye: step s~u~ = 0.3 in appearance and κ·s~u~·20° in angle, along the sign of the actor's own logit gradient, with κ = 1 or 0.25;
  - six builders on one body, 30,000-step warm-up, then frozen copies keep acting with a fixed policy.
- **Natural test set:** 3,000 fresh items; decisive sets of 300 items inside and 300 outside each action's edge (within 10°).
- **Representational agreement (E~K~):** for two agents, the share of each item's 6 nearest neighbours (cosine distance on hidden-layer activations, 600 items) that are also among its 6 nearest for the other agent, as excess over the permutation-null mean K~0~: E~K~ = (K − K~0~)/(1 − K~0~). Code from an earlier project, copied unchanged (`src/metrics.py`).
- **Ensembles:** five members trained by Thompson sampling with a bootstrap mask; diverse widths 6–64; randomized priors with β = 10.

## Appendix C. Deviations

- **D1:** the legibility check was not met on every tuning seed (3-agent seed means pass on 4/5 seeds in the unequal world, 2/5 in the symmetric world). λ = 0.4 and 60k steps were kept by the investigator's decision. The width ladder is 4–64, from the narrowest width meeting the targets on seed means.
- **D2:** E2 is fitted on all 29 eligible cells, with a shaping-strength covariate and a [0.2, 0.8] sensitivity subset, because no 3 × 3 block of unsaturated cells exists. The band is diagonal, as the law predicts.
- **D3:** an added confirmatory E8 arm with fixed 60k-step pretraining, with its prediction logged before running.
- **D4:** passive-feedback ε~r~ retuned on seeds 7040–7059 (0.01). The shaping-by-eye arm was "unmatched" by the tuning rule (0.79), and both arms still gave only two qualifying bins.
- **D5 (exploratory, post hoc):** outsider-body newcomers (body 4; builders on body 0) and wild agents with the same body at the E10 primary cell, seeds 0–9, both feedback rules, wear as a control; comparison logged in DEVIATIONS.md before running, but committed together with its results, so git doesn't timestamp the order. Shortcut reading supported (section 4.4).

## Appendix D. Pilots

Six small pilots (seeds 1000–6005, never reused) shaped the design: legible structured bodies, unequal arcs, the skip rule, passive feedback, the refusal-starvation mechanism and the preregistered thresholds. They are design checks, not evidence, and are documented in notes/PILOT.md to notes/PILOT6.md in the repository.
