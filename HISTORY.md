# History of this repository

Before its first publication, the history of this repository was rewritten once, (a) to replace the author identity on every commit with the pen name David Shen, and (b) to remove local working files that were never meant to be published: agent instructions (CLAUDE.md), status and planning notes (STATUS.md, PAPER_BRIEF.md, paper/OUTLINE.md, paper/reading_notes.md), run logs (logs/) and intermediate result pickles (results/**/parts/*.pkl). In four early versions of tools/make_export.py, a search pattern that named the author's previous account handle was replaced by `REDACTED-HANDLE`. File contents and commit times are otherwise unchanged; no commit was dropped. Because commit hashes changed, the records written before the rewrite (PREREGISTRATION.md, LOG.md, CHECKIN_3.md, the pilot notes) cite the old hashes; the table below translates them.

| Old | New | Date | Commit subject |
| --- | --- | --- | --- |
| `877f3d6` | `19d3974` | 2026-11-20 22:28 | PLAN.md (deliverable 1), environment set-up, metrics.py copy, pilots imported |
| `063329b` | `47da90e` | 2026-11-20 22:41 | PREREGISTRATION.md (criteria before any real-run code); apply REVIEW_1 to PLAN.md |
| `413663e` | `d484373` | 2026-11-20 22:41 | PLAN.md: REVIEW_1 amendment header, tanh correction |
| `810b52e` | `91b7530` | 2026-11-20 22:45 | src/world.py from pilot 6 with builder-policy fix and E4 readouts; 25 unit tests pass |
| `31cb210` | `bb92484` | 2026-11-20 22:50 | Experiment tasks (E1-E10), tuning tasks, resumable parallel runner, quick end-to-end code check |
| `1f64aaa` | `f92dfa4` | 2026-11-20 22:55 | Stage 2: legibility tuning failed to converge; diagnostics saved; stopped for David's decision |
| `0f40b5d` | `982e28c` | 2026-11-20 22:55 | Keep per-task pickles out of git; merged CSVs stay |
| `cc029c7` | `fabd843` | 2026-11-20 23:01 | DEVIATIONS D1: legibility shortfall, lambda 0.4 / 60k kept (David); PILOT6 erratum on builders' exploration |
| `de619a5` | `404476d` | 2026-11-20 23:26 | Stage 2: frozen code for real runs (eps_r tuned, width ladder, E2 range guard); tuning, sanity and smoke results |
| `032ba80` | `1433ed6` | 2026-11-20 23:26 | PREREGISTRATION section 11: frozen settings (code 404476d), sanity and smoke results, before seed 0 |
| `0023ee5` | `c3688ed` | 2026-11-20 23:28 | E1 on seeds 0-9: L1 passes at every M (kappa 1 and 0.25); stage 2 check-in |
| `a2dc0d3` | `3d7e0fd` | 2026-11-20 23:37 | DEVIATIONS D2: E2 joint OLS on all 29 eligible cells, s_u covariate, sensitivity subset; config loads the cell list |
| `2f165dd` | `b549a20` | 2026-11-20 23:40 | Stage 3 analysis scripts (E8, E9, E10); stage 3 launched |
| `e35d97c` | `f8eb77a` | 2026-11-21 00:04 | Stage 3 results (E8, E9, E10 on seeds 0-9), analysis outputs, CHECKIN_3.md |
| `54337b9` | `396c0a9` | 2026-11-21 07:34 | DEVIATIONS D3 (E8 fixed 60k pretraining arm, prediction stated) and D4 (passive eps_r retune), before running; tasks added |
| `8fe1772` | `dc065ba` | 2026-11-21 07:40 | D4 tuning result (passive eps_r 0.01; eye arm unmatched by rule); stage 4 runner |
| `6f48c97` | `8797713` | 2026-11-21 07:42 | Stage 4 analysis scripts (E2, E3, E4-E6, matched rerun); stage 4 running |
| `0fa50bc` | `388251f` | 2026-11-21 08:11 | Stage 4: E2-E6, E8 fixed-pretraining arm (D3), passive matched rerun (D4), analyses, REPORT.md |
| `5e0b1ce` | `d63fe56` | 2026-11-21 08:55 | REPORT.md: prior-work section from abstracts; one-offer design limits on conventions (2a); forgetting across generations (2b) |
| `c605a9e` | `419eeb0` | 2026-11-21 09:11 | REPORT.md tightened: E8 led by the D3 null, E2 selection explanation, L3c/L3d shared vs key-specific, L9b edge concentration untested, matched arm descriptive, fixed builder policy in L10m |
| `ab6cb8a` | `4be1969` | 2026-11-21 09:23 | REPORT.md: forgetting reframed as non-acquisition by newcomers (fully, partly, untrained cases) |
| `e89a5bc` | `5c10afc` | 2026-11-22 15:36 | Paper check-in 1: reading notes, six figure drafts from saved results, outline and title options |
| `752d855` | `10b0cf9` | 2026-11-22 16:33 | Paper check-in 2: draft, PDF, summary, verify script (64 checks), figure fixes, clean export prep; fix two rounding errors in REPORT.md |
| `7592eec` | `4696520` | 2026-11-22 22:50 | Paper revision pass after check-in 2: rendering, clarity, citations, summary |
| `c7293c0` | `32f9e99` | 2026-11-22 23:32 | Paper: payoff without understanding (E3), section 6 framing tied to E3, future work |
| `fd03682` | `3174be1` | 2026-11-22 23:58 | Paper: section 1 framing, related work, information-seeking reply, measured E3 split, wording, one-paragraph section 6 |
| `5dd19d4` | `4d603f4` | 2026-11-23 23:36 | Release prep: analogy in 4.2, acknowledgments, generic paths, untrack internal files, author-name script |
| `8af6016` | `d45fdc0` | 2026-11-24 00:15 | Final polish: glosses in 1, analogy scoped to illegibility, veterans' null qualified, capacity future work, E_K subscript, metrics.py in 7, Fig 3 spacing, summary line |
| `146977c` | `62af047` | 2026-11-24 00:46 | Review pass: exploratory outsider-body arm (D5), L10m extras, framing, settings table, fixes |
| `62b360d` | `faf346a` | 2026-11-24 10:14 | Final consistency pass: one mechanism statement, E6 line, Table B1, D5 timing, references, Fig 6 |
| `0dffda6` | `079a9e8` | 2026-11-24 10:23 | Micro-pass: "much of this error", table captions above tables |
| `48fdd3e` | `b8a497a` | 2026-11-24 11:03 | Final pass: absolute confident-error rates, L3/L3b intervals, instant snap-back, shorter section 6 |
| `154affb` | `fb54058` | 2026-11-24 11:32 | Typst render: house style, font fetch script, restyled figures |
| `7f7b5b9` | `e332b7b` | 2026-11-24 12:12 | Typst style: centred title, stacked cost boxes, grey accents instead of blue |
| `0bdbdba` | `e4b0b62` | 2026-11-24 12:24 | Author name: David Shen |
