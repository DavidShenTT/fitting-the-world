# REVIEW_1: decisions on PLAN.md (20 Nov 2026)

**Status: drafted by Claude in claude.ai for David. Not in force until David tells you it's approved.** Once approved, it overrides PLAN.md and BRIEF.md where they differ, as CLAUDE.md says.

## A. One design change: the convention experiments use the symmetric world

E1, E2, E4, E5 and E6 test conventions, which can only form in overlap zones, where the world refuses neither action. In version 1's geometry (all arcs ±60°), a third of the ring is overlap. The unequal arcs were introduced in version 5 for E10 only, and they shrink the overlap to four 5° zones (about 5.6% of items), as PLAN.md itself notes. That leaves little room for any convention to form.

- **Run E1, E2, E4, E5 and E6 in the symmetric-arc world** (`set_geometry(SYMMETRIC)`), with version 1's built share: of the items whose home is in an overlap zone, the share sitting outside every overlap zone. That is the geometry pilots 1–3 ran in. Pilot 1 found built share 0.99–1.00 against 0.04 there.
- **E3, E8, E9 and E10 stay in the unequal world,** as piloted in pilots 4–6.
- **E2's measurable form** (decision 9) then counts built items by version 1's definition (overlap-home items now outside every overlap) instead of the boundary-distance count.
- Re-check legibility on tuning seeds in the symmetric world too (pilot 5's P2 was borderline at 0.84 on one seed). Retune λ only if it fails, as for the unequal world.

## B. Answers to PLAN.md section 9

1. **Pool size 600:** approved.
2. **E1 steady-state rule:** approved.
3. **Zero wild denominators** (+0.5 counts in that world only, reported): approved.
4. **E8's T2 = 30,000 steps:** approved.
5. **E9 also reported under passive feedback,** with L9 judged under the skip rule: approved.
6. **E10's primary cell = width 32 at 60k steps,** other widths and budgets secondary: approved. Write it into PREREGISTRATION.md exactly so.
7. **L10a-matched with 4 agents per seed (40 per kind), tertile bins:** approved.
8. **L10-dose: use pilot 5's levels** (shaping by eye at κ = 1 with s_u = 0.03, 0.1, 0.3, roughly 22%, 8% and 1% edge items left), not levels inside E2's range. The test is about how much building erased, and pilot 5's levels span it. With A, E2 runs in the other geometry anyway. Record this as a change from BRIEF.md's wording.
9. **E2's measurable form:** approved (joint log–log OLS on world-level points, "width" = network width), with the built-item count from A.
10. **E4's definitions:** use version 1's text, now in `BRIEF_v2_OLD.md` (a reference copy recovered from git; it carries version 1's E1–E6 and measure definitions).
    - **Behavioural agreement is chance-corrected:** the rate of choosing the same action on the same items, corrected for chance from each agent's own action frequencies on those items (Cohen's κ). Report it separately for overlap-home and unambiguous-home items.
    - **Representational agreement:** E_K on the penultimate layer (the hidden layer here), with the permutation null.
    - In the symmetric world (A), overlap-home items are a third of the test set, so no enlargement is needed.

## C. Framing (BRIEF.md version 7.2, section 1)

Manifestation and its four-step cycle (E1–E9) lead the plan, the check-ins and the report. E10 is reported fully, but as a secondary consequence.

## D. Housekeeping

- `_incoming_layer2.zip` can be deleted; David will do it.
- Then write and commit PREREGISTRATION.md and proceed to stage 2 as planned, stopping after E1.
