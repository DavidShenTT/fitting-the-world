# Pilot 5 log and deviations

Every change below is logged before the runs it affects.

## 20 Nov, before pilot 5 code

- Version 6 of the brief and `PILOT5_SPEC.md` committed together, before any pilot 5 code.

## 20 Nov, pilot 5

- **Code.** `pilot5/world5.py` is pilot 4's world with: the arcs set by `set_geometry` (unequal or symmetric), passed explicitly into the numba code; reweighted passive feedback (`target_share`); per-step records of the label, whether the agent updated and the weight used; `label_share()`; confident error in the edge scores; and the shift toward 45° signed per arc (inward for arcs wider than 45°, so every arc in the symmetric world).
- **Tests.** All thirteen tests in `pilot5/test_world5.py` pass on the first run: pilot 4's nine, plus the symmetric geometry (and that the numba acceptance rule agrees with the Python one in both geometries), an exact match with pilot 4's world when no reweighting is applied, the reweighting hitting s = 0.5 and 0.9 within ±0.03 (and unweighted passive labels matching the arcs' own shares), and the confident-error measure on hand-built predictors.
- **Tuning step 1 and P6 (seed 5000; built share and edge share only).** Script `pilot5/tune5_dose.py`. Shaping by eye at κ = 1 with s_u in {0.3, 0.1, 0.03, 0.01} in the unequal world; wear at η = 0.15 in both worlds.
- **Step 1 and P6 result (seed 5000).**
  - **Unequal world:** natural edge share 28.3%. Wear (η = 0.15) leaves 1.2%. Shaping by eye at κ = 1 leaves 0.7% at s_u = 0.3, 8.3% at 0.1, 22.5% at 0.03 and 27.0% at 0.01 (built shares 0.98, 0.71, 0.21 and 0.05).
  - **Chosen dose levels: s_u = 0.1 and 0.03.** The first lands in the 5–10% aim; the second is a little above the 15–20% aim, and 0.01 is indistinguishable from natural. P4 passes: 0.7%, 8.3% and 22.5% differ by at least 3 points.
  - **P6:** in the symmetric world wear at η = 0.15 leaves 23.7% edge items (natural 49.3%), against 1.2% in the unequal world. This fits the tug-of-war account: 30°-wide overlaps hold items between two actions, while 5° overlaps let them go. Pilot 3 kept 21%. So S3's failure in pilot 4 came from the geometry, not from how wear's step is scaled.
- **Full pilot 5 run.** `pilot5/pilot5.py` as specified, with dose levels s_u = 0.1 and 0.03; CSVs to `pilot5/results/`; `pilot5/verify5.py` prints every figure.
- **First run done.** 569 s on 2 cores (three seeds in parallel). `pilot5/verify5.py` prints every figure in `PILOT5.md`. A reproducibility rerun started in the background.
- **D1, a design flaw found in the results: reweighting at s = 0.9 breaks wild learners.** The weight on acceptances is about 17 (wide) and 40 (narrow) at s = 0.9, against 1.9 and 4.5 at s = 0.5. With a learning rate of 0.05, steps that large destabilise SGD: wild agents at s = 0.9 have home log loss 0.760, against 0.016–0.031 for wild agents in the other unequal-world conditions, and narrow shifts of +16.7°. Built-raised agents, whose items sit far from the edges, aren't affected (home log loss 0.000). So every s = 0.9 comparison is uninterpretable, including "balance versus censoring". P3 passed (the shares hit their targets); the method that hit them is the flaw.
- **D2, a second flaw: reweighting moves the comparator.** Upweighting acceptances by w moves any well-trained agent's 0.5 crossing to where the true acceptance probability is 1/(1 + w). Wild agents have edge data, so their boundaries move with the weights: at s = 0.5 the wild narrow shift is +1.64°, against −0.05° under unweighted passive feedback. Built minus wild then mixes this artefact into L10s. Neither flaw is fixed in pilot 5; both go to version 7 with a proposed fix (offers biased toward accepted actions, which changes the label balance without changing any agent's conditional acceptance probability).
- **Reproducibility.** The rerun's `raised.csv` is byte-identical to the first run's; `q.csv` differs only in the wall-clock rows.
