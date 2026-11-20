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
