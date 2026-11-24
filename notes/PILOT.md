# Pilot report: layer 2 lifecycle experiments (20 Nov 2026)

**What this is.** A small pilot of E1, E8, E9 and E10 from `BRIEF.md`, run in a cloud workspace to check the design before CC runs it. It used 600 items, one body, wear only (shaping by eye only for E1), three seeds (1000–1002) and agent widths 8, 32 and 128. The brief's thresholds were committed to git before any pilot code; the design checks are in `PILOT_SPEC.md`, and every change made along the way is in `PILOT_LOG.md`, logged before the runs it affected. **None of this is evidence for or against a prediction.** It shows where the design works and where it breaks.

## Design checks

| Check | Result | Numbers |
| --- | --- | --- |
| C1: a lone agent reaches 95% competence within 20,000 steps | Pass, too easily | Within 500 steps at every learning rate tried, then 0.99–1.00 |
| C2: building emerges (built share ≥ 0.3 above control) | Pass | Wear 0.995–1.000 against control 0.042–0.046. Shaping by eye 0.926–0.990 |
| C3: wild agents' edge accuracy between 0.55 and 0.95 | Pass, after fix D1 | 0.666–0.728 |
| C4: built world has at most half the natural edge share | Pass | 0.020–0.030 against 0.423–0.463 |
| C5: cost | Negligible | The whole pilot took about 31 s on 2 cores |

## Design findings to take into PLAN.md

1. **The greedy policy never needed the arc edges (D1).** With 60° arcs every 90°, the action nearest an item is always accepted, with at least 15° to spare. Acceptance and the planned edge accuracy were at ceiling by construction. Fixed in the pilot two ways, and adopted in the brief:
   - Each turn an item offers two of the four actions at random.
   - Edge measures read the agent's prediction for the *boundary action* on edge items, not its choice.
2. **Competence is still trivial,** reached within 500 steps by every width. Learning cost to 95% (version 1's E3 and E4) will barely separate conditions. Use steps to a fixed edge accuracy, or to its plateau, instead.
3. **Building saturates.** At η = 0.2, 6 builders, 600 items and δ = 0.0005, the built share is 0.99–1.00. E2's upkeep law needs a regime where the pool isn't exhausted. Pick η, M and δ with E1 before running E2.
4. **E9 as written can't move.** Acceptance over all turns barely depends on edge knowledge. Both kinds are at 0.995–0.999 competence at the moment of snap-back, and their first-500 acceptance differs by 0.003 on average (range −0.058 to 0.054). Proposed: score E9 on *decisive turns*, edge items where exactly one of the two offered actions is accepted.
5. **L10b can't be run as written.** Home log loss doesn't overlap between kinds: built-raised 0.003–0.047, wild-raised 0.062–0.136. There is nothing to bin. Proposed: vary how strongly the world is built (η or the number of builders) so the ranges overlap, or match on another quantity, fixed in advance.
6. **Widths.** "Barely enough for 95%" means nothing when width 8 gets there at once. Choose widths by edge accuracy; all three plateau near 0.71 in the natural world.
7. **E8 pretraining (D2).** Pretraining to 95% competence stops long before edge knowledge settles. The pilot pretrained for 15,000 steps. Proposed: pretrain until edge accuracy levels off, with that criterion fixed in advance.

## Readouts, against the brief's thresholds

These are pilot numbers on 3 seeds. They are reported so CC knows roughly what scale to expect, not as verdicts.

| Prediction | Threshold | Pilot readout |
| --- | --- | --- |
| L8a: extra fall in edge accuracy, treatment minus control | ≥ 0.10 | 0.108 at width 8, 0.098 at 32, 0.071 at 128. Direction as predicted; one width of three clears it |
| L8b: narrowest width's extra fall vs the widest's | ≥ 2× | 1.53× |
| L9: built-raised acceptance gap in the first 500 steps after snap-back | ≥ 0.15 | 0.003. Against, but see finding 4 |
| L10a: calibration gap on natural edge items, built minus wild | ≥ 0.10 | 0.10–0.23 in 5 of 6 width × budget cells; 0.017 at width 128 after 5,000 steps |
| L10b: edge accuracy at matched home surprise | ≥ 0.10 lower | Not computable (finding 5) |

**Erratum (added after pilot 3).** The calibration gap here is unsigned: max(p, 1 − p) minus accuracy. Re-raised agents (`pilot4/recheck_p12.py`) show that the built-raised agents' larger gap came from false *refusal*, confidently predicting that accepted edge items would be refused, not from false acceptance. Read "more confident there" in that sense.

**The shape behind L10.** Built-raised agents are 4–17 times less surprised at home than wild-raised agents (home log loss), yet 0.05–0.11 worse on natural edge items in 5 of 6 cells, and more confident there. So far, low surprise at home didn't carry over to the unbuilt world, which is the pattern the naive-realism mechanism predicts. On 3 seeds with one body, this is a reason to run the real test, not a result.

**E8 in more detail.** Control agents kept improving slightly during T2: their edge accuracy fell by −0.004 to −0.033. Treatment agents fell by 0.064–0.088. Edge accuracy before T2 was 0.709–0.717 at every width.

## What the pilot doesn't cover

E2–E7, L8c (the second item type), shaping by eye in E8–E10, different bodies, and E_K. Neon's point about population versus individual retention wasn't plotted.

## Files

- `PILOT_SPEC.md`: design checks, written before code. `PILOT_LOG.md`: deviations D1 and D2 and the drift-noise fix.
- `pilot/world.py`: the world and agents (numba). `pilot/test_world.py`: sanity tests with known answers. `pilot/pilot.py`: the runner. `pilot/verify.py`: prints every figure above from `pilot/results/*.csv`.
- To regenerate: `cd pilot && python3 test_world.py && python3 pilot.py && python3 verify.py` (needs numpy, numba, pandas).
