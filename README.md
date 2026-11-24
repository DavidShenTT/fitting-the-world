# Fitting the World to the Model: What It Costs, and Who It Serves

Code, preregistration, results and paper for a small simulation study of **persistent world-fitting**: learning agents that reshape their world to suit what they already do, and what that costs in upkeep, readability and newcomers' learning.

- **Paper:** `paper/preprint.pdf` (source `paper/preprint.md`); one-page summary `paper/summary.pdf` (source `paper/summary.md`).
- **Preregistration:** `PREREGISTRATION.md`, committed before any real run (commit `063329b`); frozen settings in its section 11 (`032ba80`); frozen code `de619a5`. Deviations are logged in `DEVIATIONS.md` (D1–D5).
- **Full results:** `REPORT.md`, one row per preregistered prediction, with the command that regenerates each number.

## Requirements

Python 3.12 on any OS; CPU only, no GPU.

```
pip install -r requirements.txt
```

`torch` is used only for the representational-similarity measure (`src/metrics.py`); a CPU build is enough. `pypandoc_binary` and `typst` are needed only to rebuild the paper. `python tools/get_fonts.py` fetches the two fonts it uses (Inter and Source Serif 4, both under the SIL Open Font License) into `paper/typst/fonts/`; the figures use Inter too. `python paper/render_typst.py` then builds `preprint.pdf` and `summary.pdf`. `paper/render.py` is the HTML fallback: it writes `preprint.html`, and its PDF step (`preprint_html.pdf`) uses a local Chromium-based browser (Microsoft Edge on Windows).

## Layout

- **`src/world.py`:** the world, agents and measures (numba).
- **`src/exps.py`, `src/config.py`:** experiment tasks and frozen settings.
- **`src/metrics.py`:** representational similarity (E_K), copied unchanged from an earlier project at the commit noted in its header.
- **`runs/run.py`:** parallel, resumable runner (one result file per task, merged into CSVs).
- **`runs/*.sh`:** stage scripts.
- **`analysis/`:** one script per experiment. `analysis/figures/make_figures.py` draws every figure.
- **`results/`:** merged result tables (CSV) and analysis outputs.
- **`paper/`:**
  - `preprint.md`, `summary.md` and `figures/` (PNG, SVG);
  - `render_typst.py` builds the PDFs with Typst, using the design in `typst/` (`style.typ`, `template.typ`, and `preprint.lua`, a pandoc filter that changes layout only);
  - `render.py` builds the HTML fallback;
  - `verify_numbers.py` checks every reported number against `results/`.
- **`tests/`:** unit tests for the world (`test_world.py`) and an end-to-end smoke check of every task (`test_exps_quick.py`).
- **`pilot*/`, `PILOT*.md`:** the six design pilots (design checks, not evidence).
- **Design history:** `BRIEF.md`, `PLAN.md`, `REVIEW_1.md`, `CHECKIN_3.md`, `LOG.md`.

## Reproducing

```
bash runs/stage2_tuning.sh          # tuning and sanity checks (seeds 7000-7099)
bash runs/stage2_smoke.sh           # smoke test, seed 7010
python runs/run.py e1 0-9           # E1
bash runs/stage3.sh                 # E10 (with E9), E8
bash runs/stage4.sh                 # E2-E6, E8 fixed-pretraining arm, matched rerun
python runs/run.py e10_outsider 0-9 # D5 (exploratory); task added after the frozen code, commit 146977c
python analysis/e10.py; python analysis/e8.py; python analysis/e8.py e8_fixed
python analysis/e2.py; python analysis/e3.py; python analysis/e4.py; python analysis/e10_matched_v2.py
python analysis/e10_extra.py        # D5, eps_min extras, absolute confident-error rates (post hoc)
python tools/get_fonts.py           # fonts for the figures and the PDF build (once)
python analysis/figures/make_figures.py
python paper/verify_numbers.py
python paper/render_typst.py        # preprint.pdf, summary.pdf
python paper/render.py              # HTML fallback: preprint.html
python -m pytest tests/test_world.py
```

The whole study takes a few hours on a 16-core desktop CPU. `env.sh` sets cache directories inside the repository; adapt its interpreter path to your environment.

## Licences

- **Code:** MIT (`LICENSE`).
- **Text, figures and documents:** CC BY 4.0 (`LICENSE-text.md`).

## Citation

David Shen (2026). *Fitting the World to the Model: What It Costs, and Who It Serves.* Preprint. [DOI to be added at the Zenodo release.]
