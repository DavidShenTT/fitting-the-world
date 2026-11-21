# LOG

## 20 Nov 2026, session 1
- Read CLAUDE.md, BRIEF.md (v7.2, frozen); no STATUS.md or REVIEW_*.md existed yet.
- Made `.venv` from the base Python 3.12; added `convergence_env.pth` (single addsitedir line).
- `pip install --dry-run --no-cache-dir numba`: would install only numba 0.68.1 and llvmlite 0.50.0; numpy 2.5.4 already satisfied. Installed for real. `pip check`: no broken requirements. njit smoke function ran.
- `pytest pilot6/test_world6.py`: 16 passed (15.6 s).
- Copied Convergence `src/metrics.py` (unchanged since 9c7dd7d) to `src/metrics.py` with provenance header.
- Read PILOT.md to PILOT6.md; extracted pilot settings from pilot4–6 runners.
- Found `_incoming_layer2.zip` (listing duplicates files already present); not extracted, not deleted, git-ignored.
- `git init`; wrote `.gitignore`, `env.sh`, PLAN.md, STATUS.md, this log. Stopped for review (deliverable 1).

## 20 Nov 2026, session 1 (continued): REVIEW_1 approved
- David approved REVIEW_1.md: symmetric world for E1/E2/E4–E6 with version 1 built share; L10-dose at pilot 5's levels; E4 per version 1 (`BRIEF_v2_OLD.md`); section-9 decisions otherwise approved.
- PLAN.md: amendment header added; hidden-layer nonlinearity corrected to tanh (as in the pilot code).
- Wrote PREREGISTRATION.md (criteria, procedures, explicit confirmations on E7 geometry, E2 saturation, E4 readout). Section 11 (frozen settings, code hash) to be filled before seed 0.
- Noticed in pilot 6 code: frozen builders inherited the learner's exploration settings (ε_min arms) and had no margin exploration in the ensemble runner. Preregistered fix: builders' policy fixed across conditions.
- Built the real-run code: `src/config.py`, `src/exps.py` (E1–E10 tasks, tuning tasks), `runs/run.py` (resumable, 14 workers). Ensembles gained passive feedback (needed for L10a-ens; test added). 26 unit tests pass; `tests/test_exps_quick.py` runs every task end to end with shrunken settings.
- **Legibility check (PREREGISTRATION §3) at λ = 0.4, seeds 7000–7004, one wild agent per seed, 60k steps: FAILED.** Unequal width 32: 2/5 seeds pass; symmetric: 0/5 (`results/tuning/leg_summary.txt`). Competence ≥ 0.9 reached within 60k on every seed in both worlds (5–35k steps).
- Followed the preregistered rule (λ down in 0.05 steps), run as one ladder λ = 0.35 … 0.10 (`tune_leg_lam`): no λ passes in the symmetric world (best 3/5); in the unequal world width 32 passes only at λ = 0.10. Pass counts don't trend with λ, so the failures look like agent-to-agent noise near the thresholds rather than illegible appearance.
- Diagnostic (`tune_leg_diag`, 3 agents per seed, λ 0.4 / 0.3 / 0.2, scored at 60k and 120k; `results/tuning/leg_diag_summary.txt`): training length matters more than λ. Seed means pass on 5/5 seeds in both worlds only at λ = 0.2 with 120k steps; at λ = 0.4 / 60k, 2/5 (symmetric) and 4/5 (unequal). Mean edge accuracy at λ = 0.4 / 60k: symmetric 0.842 / 0.867, unequal 0.871 / 0.898; decisive ratio symmetric 0.579 / 0.671, unequal 0.676 / 0.720.
- **Stopped before ε_r tuning, the E2 sweep, sanity checks, smoke test and E1**, because the preregistered tuning rule did not converge and every fix changes a preregistered setting. Question to David in STATUS.md.

## 20 Nov 2026, session 1 (continued): stage 2 after David chose option 1
- DEVIATIONS.md D1 written (legibility shortfall judged on 3-agent seed means; λ = 0.4 and 60k kept; the check guards against floor effects, mainly for E8–E10). David's PILOT6.md erratum on the builders' exploration committed.
- Width ladder: width 4 meets every target on means over tuning seeds → 4, 8, 16, 32, 64 (logged in D1).
- ε_r tuning (`tune_epsr`, seeds 7000–7004, 2 agents per value, home log loss only): wear 0.01 / 0.02, eye 0.01 / 0.02 (skip / passive), all within ±20% of wild.
- Wear sanity checks (a), (b) on seeds 7030–7032: pass (`results/tuning/sanity_wear/summary.csv`).
- E2 sweep (seeds 7020–7022, 80 cells) and the preregistered extension (s_u 0.003, δ 0.008): no full-factorial block in [0.1, 0.9] with max ≤ 0.95; eligible cells lie on a diagonal in M/δ. E2 range left open for the check-in; the code refuses real E2 seeds until set.
- Added a noise-free decay readout (signed displacement projected on its abandonment sign) before the smoke test; sanity (c), (d) pass on seed 7010.
- Smoke test, seed 7010, all experiments at full size: every column filled (`results/smoke/smoke_check.txt`). CPU per seed: E10 ≈ 1,540 s, E8 ≈ 220 s, E4/E5 ≈ 320 s.
- Froze code at de619a5; PREREGISTRATION §11 committed (032ba80) before any real seed.
- **E1 on seeds 0–9** (`python runs/run.py e1 0-9`, then `python analysis/e1.py e1`): L1 passes at every M under κ = 1 and κ = 0.25. Results in `results/e1/summary.csv`. Stopped for the E1 check-in.

## 20–21 Nov 2026: E2 decision and stage 3
- David approved E1 and chose the joint OLS on all eligible E2 cells; logged as DEVIATIONS D2 (29 cells, s_u covariate, sensitivity subset of 15 cells in [0.2, 0.8], corr(log M, log 1/δ) −0.34 overall). Config loads `results/tuning/e2/range.json`; only E2's task list changed from the frozen code.
- Wrote `analysis/common.py`, `analysis/e10.py` (E9 inside), `analysis/e8.py`; tested on the smoke data.
- Ran `bash runs/stage3.sh` (E10 then E8, seeds 0–9): finished without errors. Analysed into `results/e10/analysis_output.txt` and `results/e8/analysis_output.txt`.
- Results and surprises are in CHECKIN_3.md. Main surprise checked for bugs: E8 shows no forgetting because the preregistered plateau rule ends pretraining at a median of 11–16k steps (edge accuracy 0.70–0.80). That is the code doing what the rule says, not a bug; the pilots saw the same. Stopped for the stage 3 check-in.

## 21 Nov 2026: stage 3 decisions and stage 4
- David: add an E8 arm with fixed 60k pretraining (D3), retune passive ε_r on more seeds (D4), go ahead with stage 4. D3 and D4 were logged with their predictions before running.
- Passive ε_r retune (seeds 7040–7059): 0.01 for both rules; wear matched (0.835), eye unmatched by rule (0.791). Logged before the rerun.
- `bash runs/stage4.sh`: e10_matched_v2, e8_fixed, e4, e3, e6, e2 on seeds 0–9, no errors. Analysed into `results/*/analysis_output.txt`.
- Checked surprises for bugs:
  - L4's first clause is capped by a high natural-world κ (0.877), not a bug.
  - Sanity (d) missed in 1 of 40 worlds by 0.3 points; the built share rose, not a bug.
  - E2's slopes depend heavily on the s_u covariate, because cells were selected inside the band.
- Wrote REPORT.md (one row per prediction). E7 not run (optional, not approved). Prior-work abstracts not re-read (flagged in REPORT §6).
