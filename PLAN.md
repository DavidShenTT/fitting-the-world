# PLAN.md: learning agents in a world they can shape

> **Amended by REVIEW_1.md (approved 20 Nov).** Where this plan differs from REVIEW_1 or PREREGISTRATION.md, they win. Main changes: E1, E2, E4, E5 and E6 run in the symmetric world with version 1's built share; L10-dose uses pilot 5's levels (s_u = 0.03, 0.1, 0.3); E4 uses version 1's definitions (Cohen's κ agreement, E_K on the hidden layer); the section-9 decisions are otherwise approved. PREREGISTRATION.md is the authoritative specification.

Written 20 Nov 2026 against BRIEF.md version 7.2 (frozen). This is deliverable 1; work stops here for David's review. Nothing below has been run on real seeds. Thresholds are the brief's unless a section says otherwise; every place where I propose a change or an interpretation is marked **[decision]** and collected in section 9.

## 1. Environment (done, verified 20 Nov)

Set up exactly as BRIEF.md prescribes; nothing outside this folder was written.
- `.venv` made from `C:\Users\David\AppData\Local\Programs\Python\Python312\python.exe` (Python 3.12.10).
- `.venv\Lib\site-packages\convergence_env.pth` holds the single `site.addsitedir(...)` line pointing at the Convergence venv's site-packages.
- `pip install --dry-run --no-cache-dir numba` wanted only `numba-0.68.1` and `llvmlite-0.50.0`, and reported numpy 2.5.4 as already satisfied (numba requires numpy < 2.6). It did not touch numpy, scipy, pandas or torch, so I installed it for real into `.venv`, with pip's cache in `.cache\pip`.
- `pip check`: "No broken requirements found."
- A test `@njit` function compiles and runs.
- `pilot6/test_world6.py` (which imports pilot 5's world): **16 passed** in 15.6 s.

| Package | Version | Loads from |
| --- | --- | --- |
| numpy | 2.5.4 | Convergence `.venv\Lib\site-packages\numpy\__init__.py` |
| scipy | 1.18.1 | Convergence `.venv\Lib\site-packages\scipy\__init__.py` |
| pandas | 3.1.0 | Convergence `.venv\Lib\site-packages\pandas\__init__.py` |
| torch | 2.15.0+cu134 | Convergence `.venv\Lib\site-packages\torch\__init__.py` |
| pytest | 9.2.0 | Convergence `.venv\Lib\site-packages\pytest\__init__.py` |
| numba | 0.68.1 | this folder's `.venv\Lib\site-packages\numba\__init__.py` |
| llvmlite | 0.50.0 | this folder's `.venv\Lib\site-packages\llvmlite\__init__.py` |

Every shell sets `PIP_CACHE_DIR`, `NUMBA_CACHE_DIR`, `MPLCONFIGDIR` and also `PYTHONPYCACHEPREFIX` to subfolders of `.cache\`, so no bytecode gets written into the Convergence folder. A helper `env.sh` / `env.ps1` will set these.

`src/metrics.py` is a verbatim copy of Convergence's `src/metrics.py` at commit 9c7dd7d (checked unchanged since that commit), with a two-line provenance header. E_K will be the normalised excess `excess(K, null mean)` of its `knn_overlap` statistic, with the null from its `permutation_null`. torch runs on CPU here; no GPU is used.

**Untouched file:** the folder contains `_incoming_layer2.zip`, an archive whose listing duplicates files already here (BRIEF.md, pilot logs, pilot 2 and 5 code). I haven't extracted or deleted it; it is git-ignored. Delete it if it's no longer needed.

## 2. What the pilots settle for the plan

Read PILOT.md to PILOT6.md. What carries into the real design:
- **Frozen world code:** pilot 6's `world6.py` (numba). It passes its 16 tests, matches pilot 5's world exactly when offers are unbiased, and implements unequal arcs, the symmetric world, passive feedback, biased offers, label records, refusal sources, confident error, unbuilt-edge log loss, boundary shift, ensembles with randomized priors, the matched filtered sampler, forestry entropy and passive code length.
- **Settings that worked in pilots 4–6:** λ = 0.4, observation noise 0.01, learning rate 0.05, m = 0.2, ε_m = 0.6, ε_min = 0.02; wear η = 0.15; shaping by eye s_u = 0.3; M = 6 builders on one shared map with a 30,000-step warm-up; δ = 0.0005; raising for 60,000 steps.
- **Known fragilities:** pilot 4's legibility was borderline on some seeds (wide edge accuracy 0.79 on one), so the legibility check runs on every tuning seed, not one. Wear empties the unequal world's edges (≈1% edge items left), so erasure is varied through shaping strength (s_u), not η. The matched filtered sampler must be redrawn every 50 steps. Per-world confident-error ratios vary widely (1.1–4.5), so worlds, not agents, carry the statistics.

## 3. Code layout

```
src/world.py        world + agents: pilot 6's world6.py promoted, plus E4/E5/E7 pieces (below)
src/measures.py     agreement, E_K wrapper, steady-state and plateau detectors, paired-ratio statistics
src/metrics.py      verbatim copy (E_K)
tests/test_world.py every unit test BRIEF.md section 6 lists, plus tests for the new pieces
runs/stage2_*.py    tuning checks, sanity checks, E1, E2 sweep (pilot seeds) ...
runs/stage3_*.py    E8, E9, E10 (shared raised agents)
runs/stage4_*.py    E2-E6 (E7 if approved)
runs/verify_*.py    print every reported number from results/*.csv
results/            CSV per experiment, one row per agent or world; config JSON beside each
LOG.md, STATUS.md, DEVIATIONS.md, PREREGISTRATION.md, REPORT.md
```

Additions to the world code, each unit-tested before use:
- E4/E5: building two worlds from the same home values with different builder seeds; abandonment (builders removed, drift only); behavioural agreement and representational agreement between agent pairs.
- E7: one agent with private items, encounter-frequency and noise cells, an Act / Build-and-act choice.
- A run driver that writes per-chunk checkpoints and resumes, used for every multi-worker job.

Each runner uses `multiprocessing.Pool` over seeds (16 cores; I'll use 14 workers), with entry points under `if __name__ == "__main__":`.

## 4. Exact sizes

| Quantity | Proposed value | Source |
| --- | --- | --- |
| Item pool | **600** kind-A items **[decision 1]** | pilots 1–6 |
| Drift | δ = 0.0005, angle noise 0.05°/step, appearance noise 0.002/step | pilots 4–6 |
| Bodies per world | 8 maps: 4 of type 1, 4 of type 2; observation dim 16 | pilots 4–6, brief E3 |
| λ, observation noise | 0.4, 0.01: **re-checked**, not retuned, on tuning seeds (section 5) | pilots 4–6 |
| Agent network | one hidden layer, tanh, plain SGD on BCE, zero output bias, lr 0.05 | pilots |
| Base width | 32 | pilots 5–6 |
| E8/E10 widths | narrowest width meeting legibility, then ×2 up to ×16 (expected 8, 16, 32, 64, 128) | brief E8 |
| Budgets (E10) | 30,000 and 60,000 steps | pilot 4 |
| Builders | M = 6, one shared map, 30,000-step warm-up before newborns enter | pilots |
| Wear | η = 0.15 | brief |
| Shaping by eye | s_u = 0.3, κ = 1 (E8–E10) and κ = 0.25 (E3) | pilots |
| Ensembles | K = 5; standard; diverse (widths 6, 13, 26, 45, 64 around base 32); randomized prior β = 10 | brief |
| Test sets | natural test set 3,000 items; decisive set 300 inside + 300 outside per action | pilots |
| Seeds | real: 0–999 (start at 0, 10 per condition); tuning/pilot range: **7000–7099** (never used by pilots or real runs) | brief §6 |

## 5. Stage 2: tuning checks, sanity checks, E1 (after PREREGISTRATION.md is committed)

**Tuning on seeds 7000–7004, measures the brief allows only.**
- *Legibility (wild agents only):* at λ = 0.4, noise 0.01, width 32, 60k steps, every tuning seed must give edge accuracy ≥ 0.85 and decisive ratio ≥ 0.6 for wide and narrow actions separately, and learned half-widths within ±3° of 60° and 35°. If any seed fails, step λ down (0.35, 0.3, …) and repeat; log each step. Then find the narrowest width meeting the same targets (candidates 4, 8, 16) for E8's width ladder.
- *Exploration:* confirm a lone wild agent reaches competence ≥ 0.9 within 60k steps at m = 0.2, ε_m = 0.6 on every tuning seed. Retune (competence only) only if not.
- *ε_r for L10a-matched:* home log loss only, steps of 0.01, for each built rule and main condition, chosen so built home log loss lands within ±20% of wild.
- *E2 sweep range and the L10-dose levels* (seeds 7005–7009, built share only; see E2).

**Smoke test (BRIEF §6):** one tuning seed (7010) through E1–E10 end to end with the frozen code; check every logged column is filled and no NaN. Commit the frozen code's hash in PREREGISTRATION.md before the first real seed.

**Sanity checks (verify code, no finding):** E1 under wear clusters whenever pull outpaces drift; E2 under wear the built share follows the drift–pull balance (compare with the analytic fixed point of the pull/drift recursion); E5's halving time after everyone leaves is ln 2 / δ ≈ 1,386 steps (fit an exponential to the mean |α − home| decay, within 10%).

**E1.** Natural world, M builders from random initialisation, shaping by eye; M ∈ {2, 4, 6, 12}; κ = 1 and κ = 0.25; plus the no-shaping control; 10 real seeds. Run 60,000 steps.
- *Steady state* **[decision 2]**: the built share's mean over successive 5,000-step windows changes by < 0.02; the steady-state value is the mean over the last 10,000 steps. Report the step at which it was reached; if never reached, say so.
- *L1 (κ = 1):* mean over worlds of (built share − control's built share) ≥ 0.3 for every M tested. *Against:* ≤ 0.1 for any M.
- *L1 (κ = 0.25):* builders' read appearance axis moves ≥ 0.3 from home on average (mean |u_b − home|, kind-A items, at steady state). *Against:* < 0.1.
- Also logged: builders' appearance gradient over the warm-up, the unread axis's movement (should stay at drift level), edge share.

Stop and report after E1.

## 6. Stage 3: E8, E9, E10 (shared raised agents)

**Raised agents.** In each real seed: one natural world, and built worlds under wear (η = 0.15) and shaping by eye (κ = 1, s_u = 0.3), each kept up by 6 builders still acting. Newborns enter after the builders' warm-up. Wild-raised and built-raised get equal environment steps (primary). The update-count control is a wild arm run until its acted turns equal the mean of the built-raised agents in the same seed and cell (cap 240k steps); every E8–E10 conclusion is reported both ways.

**Statistics, for every prediction.** The world (seed) is the unit. Within a world and cell, agents are averaged first. Means are reported with 95% percentile-bootstrap intervals over worlds (10,000 resamples). Criteria are judged on the point estimate exactly as the brief writes them; intervals are reported beside them, not used to pass or fail. "At least k times wild" is the geometric mean over worlds of the paired per-world ratio (built ÷ wild, same seed and condition). **[decision 3]** If a world's wild rate is exactly zero, add 0.5 counts to both numerator and denominator in that world only, and report how many worlds needed it.

### E8, forgetting
- Pretrain in the natural world until edge accuracy (on the natural test set, evaluated every 2,000 steps) changes by < 0.01 between two successive windows; cap 100k steps, report agents that hit the cap.
- Then T2 = **30,000 steps [decision 4]** in treatment (built, builders acting), filtered control (natural, offered-edge share matched to the built world's, sampler redrawn every 50 steps) and natural control.
- Widths: the five-step ladder from section 4. Wear and κ = 1. 10 seeds × 3 agents per width × arm.
- *L8a:* at the narrowest width, (fall in edge accuracy, treatment) − (fall, natural control) ≥ 0.10. *Against:* < 0.03.
- *L8a′:* |treatment − filtered| within 0.03 expected; ≥ 0.05 more fall in treatment is reported as building-specific. Offered edge shares within 0.5 points and acted turns within 10% are checked and reported.
- *L8b:* narrowest width's extra fall ≥ 2 × widest's.
- *Measure, no criterion:* population built share and individual edge accuracy over T2 on one figure.

### E9, fragility
- The width-32, 60k-step raised agents of E10 (skip rule, both built rules, plus wild) are placed in a copy of their world with every item snapped to home, builders removed, and keep learning for 2,000 steps.
- *L9:* over the first 500 steps, built-raised decisive payoff (wide and narrow pooled 50/50) at least 30% below wild-raised: (wild − built) / wild ≥ 0.30, at the world level. *Against:* gap < 10% of wild.
- *L9b:* log-loss jump at the snap (home log loss just after minus just before), built minus wild ≥ 0.1 nats.
- *Mechanism, no criterion:* act rate and refusals per act per arc type, beside each agent's shift; also under passive feedback **[decision 5]**, since the brief asks for it per feedback condition.

### E10, naive realism
- **Grid:** 10 seeds × 3 agents × {wild, wear, κ = 1} × {skip rule, passive} × 5 widths × 2 budgets. Mechanism arms (skip rule at ε_min = 0.1, 0.3), the dose arms and the ensembles at width 32 and 60k only.
- **Primary cell [decision 6]:** L10a, L10-shift, L10m and L10-dose are judged at width 32 and 60,000 steps, the cell the pilots measured. Other widths and budgets are reported in full as secondary, with the number of cells where each criterion holds. The alternative, pooling over all widths and budgets, would let the narrowest widths' legibility drive the verdict.
- **Agents:** 10 seeds × 3 agents = 30 agents per kind per condition (brief: ≥ 24 over ≥ 8 seeds; ≥ 10 worlds per condition). For L10a-matched, 4 agents per seed, so 40 per kind, enough for 3 bins of ≥ 10 each **[decision 7]**.
- *L10a (primary), each main condition and each built rule:* confident-error ratio ≥ 1.5, unbuilt-edge log-loss ratio ≥ 1.5, and home log loss built ≤ wild (world-level means). *Against:* either ratio ≤ 1, or home log loss higher.
- *L10a-matched:* built worlds get the tuned learner-only ε_r. Bin all agents by home log loss into tertiles of the pooled distribution; require ≥ 3 bins with ≥ 10 agents of each kind (if not, merge to the most bins that satisfy it and say so; fewer than 3 means the test can't be run and is reported as such). In every bin both error ratios ≥ 1.5 (bin-level, agent means, since bins cut across worlds) and built home log loss ≤ wild + 0.005 nats.
- *L10-shift (secondary, skip rule):* built minus wild unsigned boundary shift, mean of the two arc types, ≥ 2.5°. *Against:* < 1°.
- *L10-dose (skip rule):* three shaping-by-eye upkeep levels inside E2's unsaturated range (chosen on tuning seeds by built share ≈ 0.25, 0.5, 0.75). Both the pooled shift difference and the confident-error ratio rise monotonically with the share of edge items removed **[decision 8]**.
- *L10m:* mean stretch = (narrow − wide)/2 in the toward-45° convention, built minus wild; prediction: strictly falls across ε_min = 0.02 → 0.1 → 0.3. Also reported: raw stretch, near/far refusal shares, rank correlation with acceptances per far-side refusal. *Against:* not falling.
- *L10a-ens:* the three ensemble kinds under skip rule and passive, wear and κ = 1; first the randomized-prior manipulation check (edge-to-centre member-spread ratio ≥ 2 at β = 10, else the arm is dropped and reported). Readings exactly as pre-stated in the brief.
- *Exploratory:* symmetric-arc check (wild vs wear, both main conditions, with label shares, refusal sources and item-angle histograms, labelled "partial gap, 23.7% edge items"); rank correlation of home log loss and confident error within built-raised agents; biased offers restricted to items > 20° from every edge, if time allows, with nothing depending on it.

Stop and report after E8–E10.

## 7. Stage 4: E2–E6, then E7 if approved, then REPORT.md

**E2, upkeep law (shaping by eye, κ = 1).** Sweep M ∈ {2, 3, 4, 6, 8, 12, 16}, s_u ∈ {0.03, 0.06, 0.1, 0.2, 0.3} and δ ∈ {0.00025, 0.0005, 0.001, 0.002}, log-spaced, on tuning seeds; choose the test range as the cells where the built share lies in [0.1, 0.9]. Then run that range on 10 real seeds, plus three network widths (16, 32, 64).
- **Measurable form [decision 9]:** "number of built items" = count of kind-A items whose home is within 10° of a boundary and that sit > 10° from every boundary, at steady state. Fit OLS of log(count) on log M, log(1/δ) and log(width) jointly, with world-level means as data points. *L2:* slopes on M and 1/δ in [0.75, 1.25], slope on width within ±0.25. Bootstrap CIs over worlds reported. I read "width" as the agents' network width.
- *Exploratory:* the lowest M (and lowest builder competence, varied by stopping the warm-up early) at which the built share stays within 0.05 of the control.

**E3, who benefits.** Builders of one map build at κ = 0.25 (and under wear as the control); half the worlds use a type-2 builder map. Newborns: same body (builders' map), same type (the other 3 maps of that type), outsiders (all 4 maps of the other type). 10 seeds; 3 newborns per map per world.
- *L3c (primary):* forestry entropy drop ≥ 0.5 bits under the builders' body and ≤ half that under outsiders. *Against:* outsider ≥ 80% of builders'.
- *L3:* same-body gain in early learning-curve area ≥ 1.5 × outsiders'; under wear within 20%.
- *L3b:* after 2,000 steps, home log-loss drop natural → built ≥ 1.5× for same body.
- *L3d:* passive code-length saving (8,000 pairs, frozen state, paired by newborn map) ≥ 1.5× for same body.
- *Exploratory:* same-type newborns on each measure. Steps-to-criterion speed-up reported.

**E4, routes.** In each seed, two worlds built separately from the same home values (different builder seeds), plus the natural world. Pairs of newborns: same built world, different built worlds, natural world; 10 seeds × 5 pairs per arm, under wear and κ = 1.
- **Definitions [decision 10]:** BRIEF.md defers these to version 1, which isn't in this folder. My proposal: *overlap-home items* are natural-test items whose angle lies in an overlap of two arcs (the four 5° overlaps); *unambiguous-home items* are those accepted by exactly one action and > 10° from every boundary. *Behavioural agreement* of a pair is the share of items on which both agents' argmax-predicted action is the same. *Representational agreement* is E_K on the agents' hidden-layer activations over the same items (cosine distance, K at the primary density in `metrics.py`), against its permutation null.
- *L4, L5* as written in the brief.

Note: the unequal world's overlaps are only 5° wide, so overlap-home items are ≈ 5.6% of the test set (≈ 170 of 3,000). If that's too few for a stable agreement estimate I'll enlarge the test set for E4 only, not the overlaps.

**E5, decay.** Remove all agents from E4's built worlds; run drift only. Sanity check of the halving time; then newborns raised after 5/δ = 10,000 steps of abandonment. *L6:* same-world advantage (as in L4) < 0.05.

**E6.** Natural world, newborn pairs with one map of each type; agreement on overlap-home items against a chance level computed from each agent's own action frequencies on those items (permutation over items). No prediction.

**E7 (optional).** Only if David approves after stage 3. One agent, 400 private items with homes near edges, 2 × 2 cells (encounter frequency 4:1; observation noise 0.01 vs 0.05). Build-and-act costs c (set on tuning seeds so build rates are neither 0 nor 1 overall, with no cell-specific tuning) and pulls the angle toward the predicted-best action's centre by η. The choice is a per-cell-context bandit with ε-greedy exploration. *L7* as written. I expect under a day of work; I'll say so before starting if that changes.

**Prior work.** Before REPORT.md, read the abstracts the brief lists (arxiv.org, proceedings.mlr.press, openreview.net only) and position the results against them.

## 8. Cost and running

Pilot 6 took ≈ 76 µs per agent-step on one core (600 items, width 32). A rough count for stage 3 is ≈ 150 M agent-steps, wider networks costing more: about 6–10 core-hours, so **1–2 hours wall-clock on 14 workers**. Stage 2 and stage 4 are each smaller. No model downloads, no GPU, no job expected over 3 hours; the smoke test will measure real timings, and I'll stop and ask if any stage's estimate goes over 3 hours. Long jobs run in the background with logs in `logs/` and per-seed results files, so they resume by skipping finished seeds; STATUS.md will name each process ID and how to stop it.

## 9. Decisions for David before PREREGISTRATION.md

1. **Pool size: 600, not "about 2,000".** Every pilot ran 600 items, and the settings that matter (builder count, s_u dose levels, wear's edge share, warm-up length) were found at that size. At 2,000 items each item gets used about 3.3 times less often, so building would need retuning. E2's sweep covers the unsaturated regime either way. If you prefer 2,000, I'd re-establish the upkeep settings on tuning seeds first, by edge share only.
2. **E1 steady-state rule:** < 0.02 change in built share between successive 5,000-step windows; value = mean of the last 10,000 steps.
3. **Zero wild denominators** in per-world ratios: +0.5 counts in that world only, reported.
4. **E8's T2 = 30,000 steps.** The brief doesn't fix it.
5. **E9 under passive feedback too** (the mechanism is to be reported per feedback condition); L9's criterion judged under the skip rule.
6. **E10's primary cell = width 32 at 60k steps**, other widths and budgets secondary. This is the biggest interpretive choice in the plan.
7. **L10a-matched with 40 agents per kind** (4 per seed) so the binning rule can be met; tertile bins.
8. **L10-dose levels inside E2's unsaturated range** (built share ≈ 0.25 / 0.5 / 0.75). Pilot 5's strongest dose (s_u = 0.3) is saturated, so the dose test will span a smaller range of erasure than pilot 5's, and its effects may be smaller. The alternative is to use pilot 5's levels (s_u = 0.3, 0.1, 0.03) and drop "inside E2's range".
9. **E2's measurable form** (joint log–log OLS, world-level points, "width" = network width).
10. **E4's definitions** (overlap-home items, behavioural agreement as argmax match, E_K on hidden activations): my reconstruction, because version 1 of the brief isn't in this folder. If you have version 1's text, I'll use it instead.

No thresholds are changed because of a pilot.
