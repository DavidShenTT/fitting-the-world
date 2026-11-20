# Pilot 4 log and deviations

Every change below is logged before the runs it affects.

## 20 Nov, before version 5

- **Check on pilots 1–2 (Neon's round 3, question 2).** `pilot4/recheck_p12.py` re-raises width-32 wild and built (wear) agents with each pilot's own code and settings. They are not the identical agents of the original runs. It splits their edge calibration:
  - **Pilot 1:** wild false confidence on refused edges 0.306, underconfidence on accepted edges 0.357, unsigned gap 0.092. Built: 0.104, 0.687 and 0.274.
  - **Pilot 2:** wild 0.389, 0.344 and 0.117. Built: 0.207, 0.557 and 0.213.
  - So pilots 1–2's "overconfidence" was **false refusal**: built-raised agents confidently predicted refusal on accepted edge items. The unsigned gap counted that as overconfidence. The direction matches pilot 3: there was no reversal, only a measure that couldn't tell the two apart. Errata added to `PILOT.md` and `PILOT2.md`.
- **Neon's 45° midpoint mechanism, checked against pilot 3.** Neon proposes that building erases the overlap zone, so a learner interpolating between the clusters at 0° and 90° puts the boundary near the 45° midpoint. Pilot 3's edge curve (wear) crosses 0.5 about 2–3° inside the true 60° edge, not at 45°. The direction fits; the size should depend on how completely the overlap is erased (wear left 21% edge items in pilot 3 against 2% in pilot 1, where the effect was larger). Version 5 tests the mechanism directly with asymmetric arcs, which predict shifts of opposite sign.

## 20 Nov, pilot 4

- **Tests.** All nine sanity tests in `pilot4/test_world4.py` pass on the first run, including recovery of known half-widths (60° and 35°) from a hand-built predictor.
- **Tuning step 1, legibility (seed 4000, wild agents only).** Script `pilot4/tune4_legibility.py`. λ = 0.4 and noise 0.01 first; a type-1 and a type-2 map; width 32; 60,000 steps. Recorded: edge accuracy and decisive ratio for wide and narrow actions (S1), and the learned half-widths (S2). If either check fails, noise 0.005, then λ = 0.25.
- **Step 1 result.** At λ = 0.4 and noise 0.01, after 60k steps, both maps pass S1: edge accuracy wide/narrow 0.874/0.928 (type 1) and 0.935/0.885 (type 2); decisive ratios 0.686/0.777 and 0.792/0.768. Both pass S2: learned half-widths 62.0°/35.5° and 60.0°/36.2°. At 20k and 40k the narrow half-width was still 38.8–40.3° on one map, so 60k is needed. **Kept: λ = 0.4, noise 0.01.**
- **Tuning step 2, building (seed 4000; built share, edge share and appearance movement only).** Script `pilot4/tune4_build.py`. Wear η in {0.15, 0.25, 0.4}; shaping by eye s_u = 0.3 at κ = 1 and κ = 0.25 as in pilot 3. Six learning type-1 builders, 30,000 steps.
- **Step 2 result.** Under unequal arcs, wear removes nearly every edge item even at the weakest candidate: edge share 1.0% at η = 0.15, 0.5% at 0.25 and 0.2% at 0.4, against 27.2% natural. Shaping by eye at κ = 1 leaves 1.3%. At κ = 0.25 it leaves 12.8%, with u1 moving 0.69 and u2 staying in its band (0.050).
- **D1, the wear target.** S3's 5–10% came from pilot 3's symmetric world, where κ = 1 left about 6%. The brief's intent is that wear remove edges about as thoroughly as κ = 1 does, so that E8–E10 aren't diluted. Under unequal arcs both rules remove nearly all edges at pilot 3's strengths. **Kept: η = 0.15** (1.0%, against 1.3% for κ = 1). S3 fails by the letter; the intent is met.
- **Steps 3–5, design checks (seed 4000).** Script `pilot4/tune4_checks.py`:
  - **S4:** wild agents of widths 8, 32 and 128 trained 30k steps with kind-B items at 30% of offers; second-task competence at the end.
  - **S5:** standard and randomized-prior (β = 1, 3, 10) ensembles raised 60k steps in the wear world; member spread on edges vs centres.
  - **S6:** wild and built-raised (wear) width-32 agents, 60k steps, under the skip rule and under passive feedback; boundary shifts.
- **Checks S4–S6 result.**
  - **S4 fails:** second-task competence after 30k steps is −0.011 / 0.113 / 0.079 (widths 8 / 32 / 128) on one map and 0.044 / 0.007 / 0.044 on the other. No width learns the 8-sector rule, so there's no bottleneck to free. **L8c dropped from the pilot**, as pre-stated.
  - **S5:** edge-to-centre spread ratio 1.96 for standard ensembles, 2.32 at β = 1, 1.83 at β = 3 and 3.60 at β = 10. By the rule (smallest β with ratio ≥ 2), **β = 1**, though it barely differs from standard; only β = 10 clearly does.
  - **S6 fails:** single agents, built minus wild shift toward the midpoint, (wide, narrow): skip rule (−5.0, +1.5), passive feedback (+4.0, −0.2). Single agents vary by several degrees, and neither condition shows the predicted pattern (wide positive and narrow positive).
- **D2, a geometric check and a less noisy S6 (seed 4000).** Script `pilot4/diag_gap.py`.
  - **(a)** Where are the built worlds empty? Item density by angle within each quarter, for the wear and κ = 1 worlds.
  - **(b)** S6 repeated with five agents per condition, to average out agent noise.
  - The reason for (a): if building only empties a window around each pair of edges (55°–60°), an interpolating learner should put both boundaries near the window's middle. That predicts *both* arcs shrinking, not the opposite shifts Neon's 45° account predicts from tight clusters.
- **D2 result.**
  - **(a)** Built worlds are empty in a window, not down to tight clusters. Folded so that 0° is a wide centre and 90° a narrow centre, the wear world has under 1.5% of items per 5° bin from 35° to 65° and none from 45° to 60°; κ = 1 is similar (35°–65°). The window's middle is near 52.5°, between the narrow arc's edge (55°) and the 45° midpoint, so interpolating to it would shrink wide arcs by about 7.5° and stretch narrow ones by about 2.5°.
  - **(b)** Mean shift toward 45° over five agents, (wide, narrow) [sd]:
    - **skip rule:** wild (−0.1 [1.5], +2.3 [0.9]); wear (−3.1 [0.5], +6.2 [0.7]); κ = 1 (−5.7 [1.8], +6.2 [1.0]);
    - **passive feedback:** wild (−0.1 [0.9], +0.7 [1.1]); wear (+3.3 [0.4], 0.0 [0.6]); κ = 1 (+3.4 [1.2], −1.8 [1.4]).
    - Built minus wild: under the skip rule both arcs *stretch* (wide −3.0 and −5.6, narrow +3.9 and +3.9); under passive feedback both *shrink* (wide +3.4 and +3.5, narrow −0.7 and −2.5).
  - So the direction of the built-raised error depends on the feedback regime, and neither regime gives L10a's predicted pattern. What holds in every condition is that built-raised boundaries sit several degrees further from the truth, inside the emptied window, than wild agents' do.
  - The full pilot 4 still runs as specified (skip rule, β = 1, L8c dropped, η = 0.15), for the other readouts.
- **Full pilot, first run.** `pilot4/pilot4.py` as specified, three seeds, 486 s on 2 cores (about 240 s per seed). `pilot4/verify4.py` prints every figure in `PILOT4.md` from `pilot4/results/*.csv`. Added to verify4.py afterwards, computed from the same files: the L9b gap to four decimals, L8b's ratio per world, the update-count control as built minus matched wild, L10b's built-minus-wild shifts at ε_r = 0.01, and an exploratory unsigned |shift|. A reproducibility rerun (pilot4.py, diag_gap.py and tune4_checks.py) started in the background; the CSVs will be compared byte for byte, and the two diagnostics' outputs saved to `pilot4/results/`.
- **D3, the balance of labels (seed 4000).** Script `pilot4/diag_labels.py`, output to `pilot4/results/diag_labels.txt`. D2 found that built-raised boundaries stretch under the skip rule and shrink under passive feedback. A candidate explanation: under the skip rule an agent learns mostly from acts it expected to be accepted, so most of its labels are acceptances; under passive feedback it learns from every offer, and a random action accepts only about a quarter of items. Where the built world left no items, the network's guess may follow that balance. D3 records, for three wild and three built-raised (wear) agents per regime, the share of acceptances among the labels each agent learned from, for wide and narrow actions, overall and inside the emptied window (35°–65° from a wide centre), with the same agents' final shifts. It describes the label balance. It can't test the explanation, because the directions are already known.
- **Reproducibility.** The rerun's CSVs are byte-identical to the first run's, apart from the wall-clock column in `q.csv`. `diag_gap.py` and `tune4_checks.py` reproduce every figure logged above; their outputs are now saved as `pilot4/results/diag_gap.txt` and `pilot4/results/tune4_checks.txt`, and verify4.py reads D2b's built-minus-wild shifts from the first of these.
- **D3 result.** Means of three agents per row (`pilot4/results/diag_labels.txt`; built-minus-wild lines printed by verify4.py):
  - **Under the skip rule, labels are mostly acceptances:** 92% (wide) and 84% (narrow) for wild agents, 95% and 88% for built-raised (wear) agents.
  - **Under passive feedback, mostly refusals:** 35% and 18% acceptances for wild agents, 34% and 16% for built-raised ones.
  - **The built world supplies few labels inside the emptied window:** per 1,000 labels, 31 (wide) and 5 (narrow) for built-raised agents against 243 and 120 for wild ones under the skip rule; 15 and 15 against 158 and 158 under passive feedback.
  - **These agents' shifts, built minus wild:** skip (−2.8, +4.1); passive (+1.6, −0.7). Same directions as D2b.
  - So the label balance flips with the regime in the way the candidate explanation needs, and the shift flips with it. This is consistent with the explanation, not a test of it. A test would change the balance within one regime (for example, passive feedback with acceptances weighted up to half the labels) and predict the sign in advance. That is proposed for version 6, not run here.
