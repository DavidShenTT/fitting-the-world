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
