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

## 21 Nov 2026: report revisions after David's review
- Read the listed abstracts on arxiv.org (via a web-fetch subagent): Hu et al. 2020, Treutlein et al. 2021, Cook et al. 2024, Osband et al. 2018, Hein et al. 2019, Parisi et al. 2018, Dohare et al. 2023, plus Xu et al. 2019 for stigmergy. Constant et al. 2018 isn't on arXiv, so it wasn't read.
- REPORT.md: added §3a, the prior-work section; §2a (one offer plus skip: E1/E2 building is drainage out of the overlaps, E4/E6's natural agreement mostly reflects the nearer-centre readout, so L4's failure is uninformative about arbitrary conventions); and §2b (E8 with D3: built worlds stop edge learning but don't erase it, so forgetting happens across generations). Adjusted the L4, L8a and E6 rows and §2 bullets to match.
- REPORT.md tightened per David's eight points:
  - E7 stated as not run (optional, not approved).
  - E2 slopes attributed to shaping by eye at κ = 1. The collapse without s_u is explained as selection inside the band (corr(log M, log s_u) = corr(log 1/δ, log s_u) = −0.44; `results/e2/selection_note.txt`). Width claim limited to "extra width above the competence threshold adds nothing".
  - E8 led by the D3 null for individual forgetting.
  - L9b edge concentration: untested (E9 saved only whole-world log loss).
  - L3d beside L3c as shared vs key-specific gains (93% vs 22% key-specific).
  - L10a-matched strictly descriptive.
  - Fixed builder policy highlighted in L10m.
  - PLAN.md already read "tanh" (corrected in 413663e); no change needed.
- REPORT.md: the forgetting framing changed from "generational" to "newcomers" (any agent learning inside the built world). §2b now gives three cases: fully trained (D3, nothing lost), partly trained (preregistered E8, kept what they had but gained +0.04 to +0.06 against +0.15), untrained (E9/E10, never learned).

## 22 Nov 2026: PAPER_BRIEF, up to check-in 1
- Read PAPER_BRIEF.md (approved by David): an 8–10 page preprint, figure pack and summary; no new real-seed runs; stop after the outline, figure drafts and titles.
- Read Bruineberg et al. (2018, PMC6117456) and Constant et al. (2018, PMC5938575) on PMC through a web-fetch subagent: all main sections and appendices; Constant et al.'s supplementary .docx wasn't fetched. Notes in `paper/reading_notes.md`.
- Validated the figure palette with the dataviz validator (blue, orange, aqua; CVD checks pass, aqua contrast relieved by labels).
- Wrote `analysis/figures/make_figures.py`, drawing six figures from saved results into `paper/figures/` (PNG and SVG), with `figure_numbers.csv` for every new figure number (e.g. a single-predictor E2 slope of 0.93 [0.90, 0.97]). Checked each render visually and fixed overlaps.
- Wrote `paper/OUTLINE.md` with three title options. Stopped at check-in 1.

## 22 Nov 2026: paper, check-in 1 decisions through check-in 2
- David: working title "Fitting the World to the Model: What It Costs, and Who It Serves"; author undecided; MIT (code) and CC BY 4.0 (text); a public GitHub repo with a Zenodo release (prepare the export, don't push); "newcomer" everywhere; cognitive offloading in the introduction; figure fixes.
- Figures:
  - neutral grey arcs, one title baseline, "symmetric world" label (Fig 1);
  - legend moved (Fig 3); panel titles and "untrained newcomer (60k)" (Fig 4);
  - one legend per panel (Fig 5); retitled (Fig 6);
  - SVG text saved as paths (no `<text>` elements), so it displays in any viewer;
  - "newborn" replaced by "newcomer" in the figures, outline and figure numbers.
- Read the Lean community post (21 Nov 2026) and Alpöge's note (alpo.ge/remark.pdf; text extracted with pypdf from the downloaded PDF). Notes are in `paper/reading_notes.md`.
- Wrote `paper/preprint.md` (abstract 193 words, main text about 4,300 words including captions and tables) and `paper/summary.md` (540 words).
- Installed `pypandoc_binary` 1.17 (41 MB) into `.venv` only; `pip check` clean. `paper/render.py`: pandoc to standalone HTML, then the installed Edge in headless mode (profile in `.cache/edge`) to PDF. 13 pages; main text ends on page 11.
- `paper/verify_numbers.py`: 64 checks, each recomputing a reported number from `results/` and matching it to the text. It found two rounding errors inherited from REPORT.md's 4-decimal tables:
  - L3d 1.2750 printed as 1.28; correct value 1.27.
  - D3 extra fall 0.0195 printed as 0.020, and lower bound 0.0035 as 0.004; correct values 0.019 and 0.003.
  - Corrected in the preprint and REPORT.md. All 64 checks pass.
- Clean export: README.md, LICENSE (MIT), LICENSE-text.md (CC BY 4.0), requirements.txt, `tools/make_export.py` (writes `export/`, git-ignored; scans for personal details). 266 files, 12.3 MB. Nothing pushed.

## 22 Nov: revision pass on the paper draft (David's 16 points)
- Rendering: blank line before every list (lists had rendered inline); s_u, ε_min, ε_r as subscripts in prose (pandoc `~sub~`) and in Figs 2 and 6 (mathtext). The page-3 gap came from Table 1 refusing to split; tables now break between rows. Fig. 2 floats right with the §4.1 text wrapping it, which removed a second gap on page 4.
- Clarity: new abstract opening; κ defined in §2; Fig. 4 ticks "entry" / "+30k steps"; §4.3 says why width 4; §4.2 explains wear's larger code-length saving (565, 580, 574 bits, from results/e3/summary_E3.csv) and Fig. 3B's axis reads "8,000 item–action offers" (CODE_PAIRS = 8000 passive offers in src/config.py); E_K defined at L5 and in Appendix B, with +0.34 (wear) and +0.76 (shaping by eye); §4.4 separates L10-shift (unsigned) from L10m (signed outward stretch, as PREREGISTRATION §L10m defines it), adds the passive-versus-skip bridging sentence, ensemble spread written "3.7–6.2×".
- Content: §5 answers the information-seeking objection; Hu (§4.5), Parisi and Dohare (§4.3), Cook (§5) each cited where they belong; Hein's ReLU-versus-tanh caveat; §6 opens "By analogy," and names Claude, as Alpöge's note does; Appendix A's L8a verdict as David worded it.
- Summary: "What didn't work" in plainer words.
- verify_numbers.py: strings updated for the new wording, plus 6 new checks (wear code lengths, the 1.8–2.7× and 2.7–4.0× ranges, appendix L8a values and verdict, D3 appendix values). 70 checks, 0 failed. PDF 14 pages; main text ends on page 11.

## 22 Nov: second short revision
- §4.2: "Payoff without understanding" paragraph: compression 93% key-specific, code-length saving 78% shared (416/531, results/e3/summary_E3.csv).
- §6 opens "Nothing here tests capacity; this section motivates the next experiment." and links E3's split to the Riemann Hypothesis example (verdict without understanding).
- §5: "Future work" list: contested worlds (builders with different bodies), gradual decay instead of E9's instant snap-back, newcomers as the next builders and compounding errors.
- verify_numbers.py: 2 new checks (93%, 78%); 72 checks, 0 failed. PDF 14 pages; main text ends on page 11.

## 22 Nov: third revision pass (7 points)
- §1: "What is expected, and what is new": upkeep scaling as push–decay balance; outsiders' near-zero entropy drop from their bodies; contribution = ledger, measurements, mechanism test (self-made covariate shift and selective labels, no sign in home surprise).
- §5 related ideas: Perdomo et al. 2020 and Shumailov et al. (arXiv v3, 2024) from their arXiv abstracts (read through a web-fetch subagent). Lakkaraju et al. 2017 is not on arXiv (title, author and keyword searches returned nothing), so it is positioned from general knowledge without quotation; flagged in STATUS.
- §5 information-seeking reply rewritten: removed items can't be recovered; home surprise silent; randomized-prior ensemble spread 3.7–6.2× larger at edges (built-world kinds only; wild 5.1 excluded from the check) is a signal used only through Thompson sampling, not as a warning; acting on it untested.
- §4.2 "Useful but illegible to outsiders" (93% key-specific compression, 78% shared code length). §4.3 "kept what they knew but largely stopped improving": "largely" added because they still rose 0.017 and 0.010 against the control's 0.029. D3 labelled "an added confirmatory arm" throughout. §3 notes the bootstrap intervals over 10 worlds are approximate. Appendix A's L8a split into two rows.
- §6 cut to one paragraph. Alpöge's quote re-checked against the extracted note text: the quote is exact, but the old paraphrase was off (the note says producing such a document is routine, not that a readable account is impossible); now corrected. Future work: four items.
- Layout: Fig. 1 moved after "The gate" so page 2 fills; headings and list lead-in lines kept with what follows (CSS). verify_numbers.py: 72 checks, 0 failed. PDF 15 pages; main text ends on page 12.

## 23 Nov: release preparation (author's decisions)
- §4.2: one sentence offering language or musical notation as an everyday analogy for the builders' key, marked as an analogy, not a result. Acknowledgments: the author's text.
- Citations kept (Risko & Gilbert 2016; Lakkaraju et al. 2017, KDD), confirmed by the author. Title kept. Main text of 10–12 pages accepted (now ends on page 12).
- Publishing route: the author will push this repository's real history, so the preregistration commits stay verifiable. Prepared, not pushed:
  - local paths replaced with generic placeholders in BRIEF.md, PLAN.md and the provenance comment of src/metrics.py (the copied code is unchanged);
  - CLAUDE.md, STATUS.md, PAPER_BRIEF.md, paper/OUTLINE.md, paper/reading_notes.md and logs/ untracked and git-ignored (kept on disk), matching tools/make_export.py; the tracked tree now equals the export list plus .gitignore;
  - first names left in the decision records, as decided; make_export.py now scans only for local paths and emails;
  - history check: commits use a GitHub no-reply address; no credential-like strings; earlier commits still contain local paths (2 user-folder paths, 29 project paths) and the now-untracked files, which only a history rewrite could remove.
- tools/set_author.py fills the "[Author]" placeholders (paper, summary, README, licences) and rebuilds the PDF; the name is still to be given.
- verify_numbers.py: 72 checks, 0 failed. PDF 15 pages.
