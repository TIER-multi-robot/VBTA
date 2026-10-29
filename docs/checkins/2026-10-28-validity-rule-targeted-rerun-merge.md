# Check-in: validity rule confirmed + targeted re-run clean + merged files + amendment pushed + full-grid proposal (2026-10-28)

Commits this session (all pushed through `d6a959f`): `034e0ba` (H1 script,
pre-run), `573a3b8` (H1 results check-in), `00fa639` (HVBTA_RUN_WALL_LIMIT_S
+ rerun_uncapped.py), `cf4d16a` (v1 outcome check-in), `4c5a910`
(rerun_targeted.py), `5dd6262` (analysis/merge_replacements.py), `d6a959f`
(amendment, PUSHED). Tests 53/53 throughout. H1 result (preregistered): NOT
supported, direction significantly reversed - reported in the previous
check-in; no further noise analyses without approval.

## User confirmations (both clean)

(a) v1 identity cross-tab: 291/297 written compared rows identical, all
Planner_Timeout_Count = 0; 6/6 mismatching rows all PTC = 1. Discarded
compared rows: 1,251/1,264 identical (identical => Plan_Failure_Count = 0
=> PTC = 0, since PTC <= PFC and source PO has PFC = 0 on all 14,880
rows); 13 mismatching, all 13 with PFC > 0 (re-run) vs 0 (source) —
under identical code+seeds a new failure can only be load-induced
lacam_timeout. Totals: 1,542/1,561 identical (all PTC = 0), 19/19
mismatching (all PTC > 0). Zero exceptions.

(b) lacam0 exits on its FIRST solution: main.cpp `--anytime` defaults
false and the wrapper does not pass it; lacam.cpp line 134
`if (!ANYTIME) break;` on solution found. The `-t` deadline only bounds
the search for the first solution. Solved calls are machine-speed-
independent; the validity rule holds.

## Targeted re-run (rerun_targeted/) — CLEAN, first attempt everywhere

8 workers, one combined job (24 PO invalid replacement rows + 38 balanced
truncated rows + 4 fidelity controls), single-run work units, same code
and seeds, HVBTA_RUN_WALL_LIMIT_S = 12000, LaCAM 1 s limit unchanged.
Completed in 63 min (est. was ~3 h; 8-way runs are faster than the
28-way-measured costs). **66/66 rows Planner_Timeout_Count = 0 on the
first attempt — zero retries needed; 0 rows hit the stop (max 1,290 s).**
Controls: all 4 non-truncated balanced rows reproduced bit-identically
(all 82 shared non-timing columns) — the single-run driver is faithful.

## Merged analysis files (new dirs; sources untouched; v1 kept)

`phase4_final_po_merged/`: 14,880 rows; 279 truncated replaced (255 from
phase4_final_po_uncapped [PTC=0], 24 from rerun_targeted). Source_Dir
counts: original 14,601 / v1 255 / targeted 24.
`phase4_final_balanced_merged/`: 6,336 rows; 38 replaced from
rerun_targeted. Checks (analysis/merge_replacements.py), all PASS:
groups complete with all 8 methods (PO 1,860 condition groups, balanced
792); 0 rows Run_Timed_Out = 1; 0 rows PTC > 0. Original PO rows
predate the PTC column and carry 0 (justified by PFC = 0 on every source
row, verified at merge time).

## Amendment

2026-10-28 amendment (validity rule + evidence, replacement sources,
HVBTA_RUN_WALL_LIMIT_S = 12000 for re-runs and full, merge checks,
same procedure for the full sweep) committed as `d6a959f` and PUSHED
before any PO/balanced comparison numbers (none reported to date).

## Full-sweep reduced-grid proposal (HOLDING for approval; no data yet)

Estimate basis: the ONLY direct combat data is the 2026-10-25 timing
(combat 114.9 s vs den001d 63.9 s per-run mean, same code era -> 1.8x
ratio); combat/combat2 are 193x177 (~33k free cells, ~4x den001d), so
voting-per-run = 1.8x the den001d analog from the merged balanced data
(voting means 82-282 s at 50-100 robots). Sub-50-robot combos
extrapolated (fleet/pool scaling). All estimates at 8 workers (the
flip-safe count: 66/66 first-attempt clean in the targeted re-run), with
HVBTA_RUN_WALL_LIMIT_S = 12000 and the same validity rule + targeted
re-runs.

| grid | groups | serial | wall @ 8 workers | drops |
|---|---|---|---|---|
| full 36 combos (10..100 square) | 864 | ~191 h | **~24 h** | nothing |
| R1: drop size 40 (25 combos) | 600 | ~135 h | **~17 h** | the 40 interior point + its 10 asymmetric pairs |
| R2: diagonal + (50,100),(100,50) | 168 | ~64 h | **~8 h** | 29 combos (most asymmetric fleets) |

Full sits AT the ~24 h bar with genuine extrapolation risk (the 1.8x
proxy and sub-50 scaling are unvalidated on combat). Recommendation: a
calibration probe first — 2-3 direct combat voting single-runs, e.g.
(50,100) and (100,100) calm borda at 8 workers (~15-30 min total) — to
replace the 1.8x proxy with direct measurement, then launch the grid the
user picks (R1 recommended if calibration confirms; revisit if combat
comes in materially worse than modeled).

Machine idle. Next in the approved order: item 3 (PO slow-call profile,
idle, report-only) and item 8 (JV-optimum diagnostic, idle, ~2-3 h) can
run in this gap while the grid decision is pending, per the standing
machine order (item 3 -> item 8 -> full).
