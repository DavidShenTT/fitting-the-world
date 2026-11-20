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
