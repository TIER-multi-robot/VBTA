# Check-in: PO uncapped re-run v1 — stop-hit goal met, identity check FAILED on LaCAM load flips (2026-10-28)

Commits this session: `034e0ba` (H1 script, pre-run), `573a3b8` (H1 results +
balanced report + plan check-in), `00fa639` (HVBTA_RUN_WALL_LIMIT_S env
override + rerun_uncapped.py driver), + this check-in's commit. Tests 53/53
at `00fa639`. Pre-launch smoke test: full unaffected PO group (lak102d 10x10,
all 5 conditions, 40 rows) run in-memory — 40/40 rows bit-identical to
phase4_final_po on all 82 shared non-timing columns (code + seeds + env
plumbing validated before the 5-hour launch).

## PO uncapped re-run v1 (phase4_final_po_uncapped/) — RESULTS

Launched alone, 28 workers (N_WORKERS), completed in 304 min.

- **Primary goal MET: 576/576 rows written, ZERO hit the 12,000 s stop**
  (max Execution Time 2,805 s = 23% of the limit; the pre-launch linear
  extrapolation max 4,108 s was conservative).
- **Identity check FAILED: 19 of 1,561 compared rows mismatch**
  (6 among written affected rows, 13 among discarded unaffected rows);
  1,542 identical. Verify: 291/297 written non-truncated rows identical,
  0 missing/extra, 6 mismatching.

## Diagnosis (confirmed, not a code/seed issue)

All 19 mismatches are lak307d, all 8 methods INCLUDING the optimizers,
and ALL 19 differ in Plan_Failure_Count: **LaCAM 1 s wall-clock budget
hits under the concentrated all-heavy 28-way load**. The re-run wrote 30
rows with Planner_Timeout_Count > 0 (across 21 driver groups; 24 of them
replacement rows for truncated runs, 6 among non-truncated kept rows) —
while the source phase4_final_po has ZERO failures of any kind on all
14,880 rows. The original PO sweep's 30-worker queue was mixed
(lak102d/arena/den204d/optimistic groups interleaved); this re-run put
all 28 workers on the heaviest lak307d configs for 5 h straight, and
worst-case LaCAM calls crossed the 1 s budget. This is the known
load-flip phenomenon (documented in the ablation-v2 addendum and the
2026-10-28 amendment: "nonzero [Planner_Timeout_Count] values mark
load-affected rows") — the new column makes it directly observable per
row, which is exactly why it was added.

Consequence: v1's 30 flip rows are load-contaminated (the 24 contaminated
replacement rows are analysis data, not just verification), so **v1 is
unusable as the amendment's replacement set as-is**. It is NOT deleted
(kept at phase4_final_po_uncapped/ pending the user's call). Code + seeds
are NOT in doubt: the 40/40 smoke test and 1,542/1,561 identical rows
under 28-way load confirm identical-code determinism; every mismatch is
attributable to an observed LaCAM budget hit.

## Options prepared (balanced re-run NOT started; machine idle)

Measured serial cost at 28-way pace (conservative; runs get faster at
lower load): 21 flip driver groups 72.2 h; all 46 groups 114.7 h.

| option | workers | est. wall | tradeoff |
|---|---|---|---|
| A: re-run all 46 groups, fresh dir v2 | 14 | ~8-9 h | single clean replacement dir (recommended) |
| B: re-run only the 21 flip groups | 14 | ~6-7 h | replacement = v1's 25 clean groups + v2 (two-dir merge) |
| A/B at lower load | 10 | +3 h | more LaCAM flip margin |

Verification after any re-run is exact (Planner_Timeout_Count per row +
identity check); if flips appear again, drop workers further and repeat.
Raising HVBTA_LACAM_LIMIT_S instead was considered and rejected: it
changes the planner config for the replacement rows relative to every
other row in the PO dataset.

## Note for the item-4 full-sweep estimate (later checkpoint)

The full sweep (combat/combat2, 2 maps) would concentrate 30 workers on
100-robot heavy groups for its entire late phase — the same profile that
flipped LaCAM here, on den901d-class dense maps. The full-sweep plan
should either use reduced workers or explicitly accept/detect flips via
Planner_Timeout_Count. To be raised at the item-4 checkpoint.
