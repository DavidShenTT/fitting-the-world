# Pilot 6 log and deviations

Every change below is logged before the runs it affects.

## 20 Nov, before pilot 6 code

- Version 7 of the brief and `PILOT6_SPEC.md` committed together, before any pilot 6 code.

## 20 Nov, pilot 6

- **Code.** `pilot6/world6.py` is pilot 5's world with: biased offers (`offer_share`), drawn in numba so that with probability s the offered action accepts the item; the item's angle recorded on every learner turn; `refusal_sources()`; unbuilt-edge log loss in the edge scores; and folded angle histograms.
- **Tests.** All sixteen tests in `pilot6/test_world6.py` pass. They are pilot 5's thirteen (the exact-match test now compares with pilot 5's world), plus three new ones:
  - biased offers hit 0.9 overall and about 0.93 / 0.87 per action type, without changing the outcome of any recorded angle and action;
  - refusal-source counts on a hand-built record;
  - unbuilt-edge log loss on hand-built predictors, and a flat histogram for a uniform world.
- **Full run.** `pilot6/pilot6.py` as specified, no tuning; CSVs to `pilot6/results/`; `pilot6/verify6.py` prints every figure.
- **First run done.** 495 s on 2 cores (six seeds in parallel). A reproducibility rerun started in the background.
- **D1, a design flaw: biased offers move wild boundaries (Q2 fails).** Wild agents' half-widths under biased offers are 4.6° (wide) and 4.0° (narrow) wider than under passive feedback, against Q2's limit of 1.5°. Their confident error rises from 0.051 to 0.148 and their home log loss from 0.019 to 0.052. The outcome given the true angle is unchanged, but the agent sees noisy observations. Near an edge, an item just inside is offered the action about 27 times as often as one just outside, so the agent's posterior given what it sees shifts outward, much as a reweighting would. The spec flagged this possibility before the run. So every built-minus-wild comparison under biased offers carries a version of pilot 5's D2. Proposed fix: bias offers only for items more than 20° from every arc edge and offer uniformly near edges, which starves far-side refusals without biasing the edge posterior.
- **A correction to pilot 5.** Pilot 5's power estimate (agent-to-agent sd about 1.7°, about 24 agents per kind) included the broken s = 0.9 arm. Without it the sd is 0.61° for the pooled shift. Erratum lines were added to `PILOT5.md` and to `pilot5/verify5.py`. The note in version 7's E10 was corrected to cite pilots 5–6. The requirement (at least 24 agents per kind over at least 8 seeds) is unchanged.
- **Reproducibility.** The rerun's `raised.csv` and `hist.csv` are byte-identical to the first run's; `q.csv` differs only in the wall-clock rows.
