# Check-in: PO + constrained sweeps, LaCAM sensitivity, item-8 pilot (2026-10-28)

Commits this session: `efadc78` → addendum `23a2e54` (pushed) → this
check-in's commit (Planner_Timeout_Count logging-only column + amendment
2026-10-28 + this file). Tests 53/53 at every step.

**Rule acknowledged (item 5 of the directive):** I deleted
`verify_unified_pool` without asking during the earlier validation. The
rule is understood to cover directories I created myself: I will ask
before deleting or modifying ANY directory. No results directory was
touched otherwise.

## PO sweep (phase4_final_po) — REPORT-ONLY FINDING: 279 wall-stop rows

14,880 rows, unified format. Collision/invariant counters all clean
(0 conflicts / violations / stuck / nudges / mover-waits / mis-assignments,
0 planner failures, 0 cap hits, 0 planner-dead). BUT:

- **279 rows hit the 600 s run wall safety stop (Run_Timed_Out = 1)** —
  the sweep's only invariant violation. Distribution: 276 on lak307d
  (+3 arena); **all under pessimistic masks** (173 low-pess, 106
  high-pess); **all 6 voting methods** (~45-48 rows each), zero optimizer
  rows; concentrated on the large fleet configs (100-robot: 180 of 279).
  Mean completion on truncated rows 0.395 (min 0.106).
- Mechanism (from the CSV): voting allocation time explodes under
  pessimistic masking on the large map — per-call p95 50-65 SECONDS
  (mean 13-14 s) vs ~1.6 s p95 in the noise sweep; optimizers are at
  ~0.2-1.3 ms/call and unaffected. The slow calls are candidate
  generation/ranking on large sparse (masked) views. The 600 s stop
  truncates the affected runs; the rows are retained (Run_Timed_Out=1),
  never dropped or re-run. No fix proposed; awaiting your call.
- sweep_report: `phase4_final_po/sweep_report.txt`.

## Constrained sweep (phase4_final_constrained) — clean

3,360 rows (420 groups x 8), ALL INVARIANTS PASS: 0 wall stops, 0
failures, 0 cap hits, 0 planner-dead. Voting alloc p95 0.44-0.64 s/call
(no masks, small maps — no sign of the PO pathology). Report:
`phase4_final_constrained/sweep_report.txt`.

## Item 2 — LaCAM 1 s wall-clock budget hits, per sweep

The simulator classifies failures internally (kind = "lacam_timeout" for
budget hits) but the CSV recorded only the aggregate Plan_Failure_Count.
Counts in the existing sweeps, via Plan_Failure_Count (a LaCAM budget hit
is always a failed call; if a row has zero failures it has zero budget
hits):

- **noise (phase4_final_noise): 0** — Plan_Failure_Count max = 0 across
  all 10,560 rows.
- **PO (phase4_final_po): 0** — max = 0 across all 14,880 rows.
- **constrained (phase4_final_constrained): 0** — max = 0 across all
  3,360 rows.
- (hyper / hyper_500k are PP sweeps: their failures are engine_unsolved
  expansion-bound outcomes, not LaCAM budget hits.)

The ONLY known LaCAM budget hits to date were in the superseded
ablation v2 arm (3 failures in one 20x20-overload row, under the
concurrent 30-worker hyper-500k load).

**Logging-only change (recorded as the 2026-10-28 amendment):** added
`Planner_Timeout_Count` — per-run count of failed calls with kind ==
"lacam_timeout" — as the last output-tuple column, produced by the shared
builders (serial 85 cols, pool 87). Verified no behavioral change by the
pool-vs-serial identity test: 160 rows x 81 shared non-timing columns
identical (the new column compared, identical), profiles identical
(fresh dirs verify_unified_serial_v2 / verify_unified_pool_v2). Full
suite 53/53. sweep_report now prints the per-sweep timeout-hit total
(reads by name; absent in pre-column dirs). Note: constrained was
launched before the edit and lacks the column (its failure count is 0,
so its budget-hit count is 0 either way); balanced and full carry it.
Counts reported above; no fix proposed.

**Correction to the 2026-10-27 check-in:** the addendum stated "The noise
sweep's row is the canonical (idle-outcome) row" — wrong. The noise
sweep ALSO ran under load (its own 30 workers). What the idle re-run
actually showed is that the noise sweep's row and the idle outcome
AGREE for that group; the noise sweep itself was not idle.

## Item 3 — JV-optimum diagnostic (item 8): what it measures + pilot

Driver: `jv_optimum_diag.py` (diagnostic only; no core-code changes; no
results dirs touched). **What it measures:** for every NON-URGENT voting
allocation call in the re-run groups (reproducing the noise sweep's call
sequences; must run idle — a LaCAM budget flip under load would change
the sequence), it records the voting winner's total suitability vs the
OPTIMAL assignment value (JV, offline) on the same unassigned x
unassigned submatrix the allocator saw (the masked/noisy view,
recomputed deterministically), plus call size (n_robots, n_tasks, pool).
Reported as winner/optimum ratio by unassigned-robot bin and per rule.

**Pilot findings (plumbing validated, small size):** non-urgent calls
are scenario-dependent: lak102d 10x10 borda — calm 0/33, task_churn
28/53, fleet_churn 34/66, overload 2/122. One task_churn group (6
rules): 187 non-urgent calls recorded; ratio p50 = p95 = 1.0, min 0.0
(degenerate empty-winner calls), 137/187 exactly at the optimum.
(Records in jv_optimum_diag_pilot/.)

**Full-size plan (HELD for your go-ahead per "report before running at
full size"):** lak102d all 4 nr x nt combos + lak307d {50, 100} combos;
4 scenarios, rep 0, sigma in {0, 0.2}; 6 voting methods; estimated
~2-3 h idle. Balanced is currently occupying the machine; the run can go
in the next idle gap.

## Item 4 — hyper planner-dead denominator + the 20 s Condorcet call

- **Hyper planner-dead: 85 of 1,920 runs** total (the sweep_report table
  shows per-method counts out of 240 runs per method = 1,920/8; the
  earlier shorthand "85/240 runs" conflated the two — the correct
  denominator is 1,920 runs).
- **The 20 s Condorcet call** (`condorcet_probe.py`, idle re-run of the
  item-6 config, wrapper-level, report-only): the probe reproduced the
  slow-call class but measured its worst at 6.8 s (vs 20.2 s max in
  realtime v2 — same config, wall-clock variance between idle runs; the
  mechanism is the same). Worst call details: scenario **calm**, tick
  **347**, **NOT urgent** — the stall rule did NOT apply (stall_k=None;
  non-urgent calls rank the full pool by design), effective pool
  **300** (nc=600 halved by 1 recent event), ~103 unassigned robots
  pre-call (the probe printed the post-call snapshot: 103r/0t — all
  tasks were paired by the call). So the tail is large NON-URGENT
  full-pool rankings, not an anytime-rule failure. No fix; report only.
  Records: `condorcet_probe/condorcet_probe.json` (400 calls).

## Launch status

- **balanced: RUNNING** (`phase4_final_balanced/`, detached; first sweep
  with the Planner_Timeout_Count column). Sweep report + next check-in
  when it finishes; **full** is last in the approved order.
- Machine policy followed: constrained ran alone; balanced runs alone;
  single-process diagnostics (identity test) only alongside multi-worker
  sweeps when their config is historically flip-free, and the item-8
  full run + any timing measurement wait for idle gaps.
