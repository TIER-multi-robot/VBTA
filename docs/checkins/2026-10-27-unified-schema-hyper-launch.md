# Check-in: unified output schema, noise + hyper sweeps, items 1-4 (2026-10-27, evening)

Commits this session (all pushed unless noted): `03ffbd6` (Planner_Dead_Tick +
sweep_report planner-dead section), `977416a` (expand_results.py + noise
expansion, pushed), `0195667` (schema unification), plus this check-in's
commit. Amendment appended to `docs/ANALYSIS_PLAN.md` (2026-10-27: output-
schema unification). Tests: 53/53 pass at every step.

## Item 1 — Output-path unification (bug fix, done BEFORE the hyper launch)

**The bug:** two entry points wrote different schemas. The pool runner packed
the whole 59-item output tuple into one stringified `Output` column with no
initial-quality columns and no profiles file; only the serial runner wrote the
full 84-column schema + profiles. The serial Run ID was also a resume-order
counter, and only the FIRST run of each group appeared in the profiles file.

**The fix (commit `0195667`):** both runners now assemble rows/profiles with
the same shared builders (`build_result_row` / `build_initial_prefix` /
`_initial_quality_fields` / `build_profiles_row` / `make_run_id`, all module
level in main_simulation). The pool driver writes the full unpacked 84-column
schema + leading Map/Rep columns (86) and `phase4_profiles.csv` (one row per
group, keyed by the deterministic per-group Run ID `"{map}_{nr}x{nt}_{scenario}_rep{rep}"`
used by BOTH runners). The serial Run ID counter is replaced by the same
deterministic ID; the po_diagnostics filename now includes the condition so
filenames stay unique. `sweep_report.py` reads all three formats (new unified
pool, legacy packed noise dir, serial stamped files — also fixed a latent
early-return bug that made the serial loader read only the first file).

**Validation (`verify_unified.py`):** one small config (lak102d 10x10, rep 0,
4 scenarios, 5 sigmas, 8 methods = 160 rows) run fresh through BOTH runners:
**identical on all 80 shared non-timing columns (including Run ID); profiles
files identical (all 7 columns).** Excluded: Planning_Time_us,
total_reassignment_time, Planner_Total_ms, Execution Time (wall-clock).
Diff report: 0 mismatches. Recorded as the 2026-10-27 schema-unification
amendment, which also documents the noise sweep's seed-regenerated
initial-quality columns and their verification (item 2 below).

## Item 2 — Noise-sweep reconstruction verified on a large map (done)

`verify_large_map.py`: lak307d 50x50, rep 0, all 4 scenarios, sigma_obs = 0
(32 rows), run serially through the unified pool runner and compared against
`phase4_final_noise`'s expanded rows: **32 rows × 82 non-timing columns
identical; the 4 profiles rows identical.** Combined with the earlier
lak102d verification (160 rows × 79 columns, fresh serial run), the
reconstruction is verified on small and large maps. `expand_results.py`
stays for `phase4_final_noise` only; hyper onward writes the schema directly.
`phase4_final_noise` is untouched.

## Item 3 — Allocation-timer coverage (read BEFORE the item-6 interpretation)

**What the sweep's per-call metric covers** (CSV: (Planning_Time_us +
total_reassignment_time)/(1+reassignments)): each allocator's CORE only —
- voting: `assign_tasks_with_voting` = warm-start/hall-of-fame pool assembly
  + candidate generation (diversity filter) + ranking/anytime loop;
- Murty's: `murtys_k_best_assignments` (k=10 replan / k=1 initial);
- perturbed JV: the JV solve.
For EVERY method this EXCLUDES the per-call submatrix scoring and the
post-solve stealing loop — so the sweep metric is core-comparable across
methods as-is; no metric change needed.

**What item 6's real-time check covers:** the whole ENTRY functions
(reassign_* + initial assigns) = scoring + core + steal for every method —
the correct measure vs a 1-second tick. One coverage gap was found and fixed
before interpretation: Murty's replans route through
`reassign_robots_to_tasks_with_murtys`, which the first pass did not wrap
(its stats showed only the 4 initial solves). The patched driver was re-run
on the idle machine into `realtime_check_lak307d_v2` (366 Murty alloc calls
vs 4 before); v1 is kept on disk as the record of the gap.

**Breakdown** (`alloc_breakdown.py`, one noise group per map — lak102d 10x10,
den204d 20x20, lak307d 50x50; calm, rep 0, sigma 0; serial in-process; run
under hyper-sweep load so absolute times are inflated, shares are the point):
us/call p50 (p95, max) —
- voting (all 6 rules): scoring ~60-70 (≤2.5k); core (cand-gen+ranking)
  17k-29k p50, max up to ~3.5M; post-core (steal incl. its scorer calls +
  bookkeeping) ~330-380.
- Murty's: core 54 p50 (max 227); post 41. Perturbed JV: core 105 (max 345);
  post 334.
So: scoring and stealing are negligible for every method; allocation time IS
core time, and the sweep metric and the whole-entry metric tell the same
story (entry ≈ core + <1ms).

## Item 6 — Real-time check (patched pass, `realtime_check_lak307d_v2`)

lak307d 100x100, rep 0, all 4 scenarios, 8 methods, sigma_obs = 0, serial,
idle machine (730 s total). Allocation us/call p50/p99/max; planner ms/call
p50/p99/max; calls (alloc/planner):

| method | alloc p50 | alloc p99 | alloc max | plan p50 | plan p99 | plan max |
|---|---|---|---|---|---|---|
| borda | 106,910 | 1,057,555 | 1,502,532 | 41.7 | 62.9 | 81.1 |
| approval | 91,796 | 1,038,195 | 1,300,784 | 41.8 | 53.1 | 76.9 |
| majority judgment | 26,535 | 1,060,900 | 1,498,681 | 42.0 | 61.8 | 72.3 |
| cumulative | 96,411 | 1,059,707 | 1,452,522 | 42.2 | 56.5 | 67.1 |
| condorcet | 104,021 | 1,466,152 | 20,243,686 | 41.9 | 58.4 | 75.6 |
| range | 88,431 | 1,284,913 | 4,131,708 | 41.9 | 56.9 | 68.8 |
| Murty's | 329 | 874 | 1,026 | 40.8 | 54.8 | 88.3 |
| perturbed JV | 486 | 6,196 | 6,454 | 41.2 | 55.7 | 71.7 |

Interpretation (with the coverage statement above): the planners fit a
1-second tick everywhere (max 88 ms). The optimizers are real-time-capable
at the largest config (Murty's max ~1 ms; perturbed JV max ~6.5 ms). Voting
typically fits (~27-107 ms p50) but has a heavy tail: p99 ≈ 1.0-1.5 s and
max 1.3-20 s (condorcet) — i.e., ≥1% of voting allocation calls on 100x100
exceed the tick budget. These runs use the main config (stall rule k=5 on
urgent calls), so the tail is post-stall; the anytime rule is what keeps the
median deep below the tick.

## Sweeps — noise (pre-unification) and hyper (first unified sweep)

**noise** (`phase4_final_noise/`, packed format, expanded by
expand_results.py): 10,560 rows, 264 groups — ALL INVARIANTS PASS
(0 conflicts / violations / stuck / nudges / mover-waits / mis-assignments /
wall-stops / cap hits; planner-dead: none). Report:
`phase4_final_noise/sweep_report.txt`. Allocation us/call (core metric):
optimizers ~380-440; voting mean 315k-542k under full-load timing.

**hyper** (`phase4_final_hyper/`, first sweep in the unified format): 1,920
rows, 240 groups — ALL INVARIANTS PASS (mover-waits exempt per --pp-stress;
0 wall-stops). PP stress by design: fails/run 16.6-30.9, cap hits 36,059 on
951 rows, and the new planner-dead section: 85/240 runs went planner-dead —
per method (dead/runs, tick mean/p50, completion dead vs other):
Murty's 5/240 (155/166, 0.570 vs 0.733); perturbed JV 9/240 (106/94, 0.319
vs 0.732); voting 7-18/240 with completion 0.10-0.15 dead vs 0.57-0.59
other. Report: `phase4_final_hyper/sweep_report.txt`.

## Item 5 — anytime ablation (config finding; v2 in flight)

The first run covered only the diagonal (10x10, 20x20) lak102d pairs — but
the noise sweep runs the FULL nr×nt CROSS PRODUCT (build_queue iterates nr
then nt), so the comparison arm has 384 sigma-0 lak102d rows, not 192. Fixed
the driver to the cross product and re-launched into
`ablation_anytime_fullpool_v2` (48 groups × 8 methods, in flight; the first
dir is kept, superseded). Config-match + optimizer-rows-identical check runs
when it finishes.

## Item 7 — hyper cap sensitivity (in flight)

`phase4_hyper_500k` launched with `HVBTA_PP_MAX_EXPANSIONS=500000`,
everything else equal to the main hyper sweep (which is its 250k comparison
arm — same current code, same seeds). Report (completion,
Plan_Failure_Count, Planner_Cap_Hits by method for both settings) lands in
the next check-in.

## Transparency notes

- `verify_unified_pool` (my own just-created scratch validation dir) was
  deleted once mid-session to re-run the comparison after a script-side
  string-vs-float key bug; identical content was regenerated. No results
  directory was touched otherwise; no deletions of pre-existing data.
- New dirs this session: `verify_serial_expand_v5`, `verify_unified_serial`,
  `verify_unified_pool`, `realtime_check_lak307d` (v1, Murty-coverage gap),
  `realtime_check_lak307d_v2` (complete), `alloc_breakdown`,
  `phase4_final_hyper`, `ablation_anytime_fullpool` (superseded),
  `ablation_anytime_fullpool_v2` (in flight), `phase4_hyper_500k` (in flight).
- `verify_item2.py` is superseded by `verify_unified.py` (packed-format
  comparison no longer applies to new sweeps).

## Next

1. Ablation v2 finishes → config-match + optimizer-identity check vs the
   noise sweep's sigma-0 lak102d rows.
2. hyper-500k finishes → item-7 comparison report (vs phase4_final_hyper).
3. Launch PO (next in the approved order), then constrained, balanced, full;
   sweep_report after each.
4. Item 8 (JV-optimum diagnostic on ~2,000 non-urgent noise-sweep calls)
   still pending — driver to be written; runs alongside/after PO.

---

## ADDENDUM (same evening — items 5 and 7 completed; PO launched)

### Load-sensitivity finding (LaCAM wall-clock budget) — important

The ablation v2 (run CONCURRENTLY with the 30-worker hyper-500k sweep)
produced 95/96 bit-identical optimizer rows: ONE perturbed-JV row
(lak102d 20x20 overload rep1) diverged with Plan_Failure_Count 0 vs 3 —
a LaCAM call hit the 1-second WALL-CLOCK budget under the load burst and
failed, changing the run trajectory. An idle re-run of that group is
bit-identical to the noise sweep's row, confirming: the code is
deterministic; the known soft spot is `PLANNER_TIME_BUDGET_S = 1.0 s`
(wall-clock), which can flip marginal calls under heavy machine load.
The noise sweep's row is the canonical (idle-outcome) row. Action taken:
the ablation arm was re-run IDLE (v3); for future sweeps, avoid running
light single-process arms concurrently with 30-worker sweeps when
LaCAM-budget marginal calls are possible. (All other rows matched even
under load; normal runs fail 0-2 calls and lak307d planner p99 is ~60 ms
- far from the budget.)

### Item 5 — anytime ablation: CLEAN (v3, idle machine)

`ablation_anytime_fullpool_v3/` (48 groups = the full lak102d nr×nt cross
product, 3 reps, sigma_obs = 0, 8 methods, unified format; 384 rows):
- config columns match the noise sweep's sigma-0 lak102d rows on all 384
  rows (including Run ID, nc, rates, sigma);
- **96/96 optimizer rows bit-identical** (stall rule never touches
  optimizers — the config-match proof);
- **0/288 voting rows identical** — every voting row differs, exactly the
  stopping-rule treatment contrast the ablation needs.
- Invariants pass (sweep_report.txt in the dir). v2 is kept as the
  load-flip record; v1 (diagonal pairs only) superseded.

### Item 7 — hyper cap sensitivity (250k vs 500k): 250k stands

`phase4_hyper_500k/` (PP_MAX_EXPANSIONS=500,000, else identical to the
main hyper sweep; 1,920 rows):
- Completion by method: unchanged within ±0.01 everywhere
  (250k: voting 0.545-0.572, optimizers 0.716-0.730;
  500k: voting 0.537-0.569, optimizers 0.705-0.722).
- Plan_Failure_Count/run: marginally up (e.g., approval 27.9→29.0).
- Planner_Cap_Hits/run: marginally DOWN (expansion cap fires less with a
  2x budget; e.g., condorcet 23.7→19.1) but planner-dead runs UP
  (85→131/1,920; e.g., approval 14→20) - each failure burns longer before
  failing, exhausting the 150-failed-call budget in more runs.
- **One run hit the 600 s wall safety stop** (lak103c 15x15 overload rep1,
  approval) — the 250k calibration keeps runs far below the wall stop;
  at 500k the per-call work can double and the count-based bounds no
  longer guarantee it. sweep_report for the 500k arm flags this row.
- Conclusion: the 500k expansion cap buys no completion and breaches the
  wall-stop guarantee; the 250k setting is retained. (The wall-stop row is
  retained in the data with Run_Timed_Out=1, never dropped.)

### Sweep order status

PO launched (`phase4_final_po`, unified format, detached) — 31 map-config
cross-product combos x 4 scenarios x 3 reps x 5 mask conditions x 8
methods; sweep_report + check-in when it finishes. Then: constrained,
balanced, full. Item 8 driver next.
