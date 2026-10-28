# Check-in: H1 analysis (preregistered), balanced report, uncapped re-run plan (2026-10-28)

Commits this session: `034e0ba` (analysis/h1_noise.py, committed BEFORE the
script was run, per directive) + this check-in's commit. Tests 53/53 at
`034e0ba`. Machine idle at session start (balanced had finished).

## Item 1 — H1 (preregistered, noise sweep): NOT SUPPORTED, direction REVERSED

Script: `analysis/h1_noise.py` (commit `034e0ba`, made before any results
were seen). Implements ANALYSIS_PLAN.md H1 exactly: per-group
Task_Completion_Fraction degradation sigma 0 -> 0.3; Murty's vs each of
Borda / Majority Judgment / Condorcet; two-sided Wilcoxon signed-rank
(zero pairs dropped, count reported); median paired difference with
percentile bootstrap 95% CI (10,000 resamples, zlib.adler32-seeded);
Holm-Bonferroni across the 3 comparisons within each reporting level;
alpha 0.05; per scenario + pooled; descriptive curve over all 8 methods.
Data: `phase4_final_noise/phase4_expanded.csv` (10,560 rows = 264 groups
x 5 sigmas x 8 methods, sanity-checked: complete cross-product, 0
Run_Timed_Out, invariants 0). Independent spot-check of 3 groups +
curve-table cross-check passed.

Results (paired difference = Murty's degradation minus the rule's;
positive = Murty's degrades more = H1 direction):

- Pooled (n=264): all three comparisons significant in the REVERSE
  direction — Murty's degrades LESS than Borda (median -0.0224, Holm
  p=3.5e-4), MJ (-0.0354, Holm p=3.8e-14), and Condorcet (-0.0170, Holm
  p=3.5e-4). Verdicts: "not supported (reverse)" x3.
- calm (n=66): all three reversed (Holm p <= 2.3e-4).
- task_churn (n=66): all three reversed (Holm p <= 1.9e-6).
- fleet_churn (n=66): MJ reversed (Holm p=4.3e-5); Borda (Holm p=0.063)
  and Condorcet (raw p=0.060, median 0.0000) not significant.
- overload (n=66): MJ reversed (Holm p=0.0040); Borda and Condorcet not
  significant (Holm p=0.70).
- Zero pairs dropped by Wilcoxon: pooled 12/14/19 (Borda/MJ/Condorcet);
  max in any scenario-level test 15 (fleet_churn, Condorcet).
- Descriptive curve (mean completion by sigma, pooled): Murty's 0.904 ->
  0.828 (drop 0.076); perturbed JV 0.901 -> 0.791 (drop 0.110); voting
  rules 0.866-0.872 -> 0.758-0.780 (drops 0.088-0.114, MJ worst).

Headline: the preregistered prediction (cardinal Murty's degrades MORE
than ordinal/median rules under observation noise) is contradicted —
Murty's has the SMALLEST completion drop of all 8 methods. Reported as
preregistered (hypothesis reported though unsupported; reverse
significance flagged per the amended verdict rule).

## Balanced sweep — DONE + sweep_report: 38 truncated rows / 11 groups

`phase4_final_balanced/`: 6,336 rows, 792 groups x 8, complete, carries
the Planner_Timeout_Count column. All collision/booking invariants 0,
0 planner failures, 0 cap hits, 0 planner-dead, 0 LaCAM timeout hits.
**VIOLATION: 38 rows hit the 600 s stop (Run_Timed_Out=1)** — 36 on
den001d + 2 on den901d, sizes (50,75)/(50,100)/(75,100), 27 of 38 in
calm, ALL 6 voting methods, ZERO optimizer rows — the same signature as
PO (no masks involved). Mean completion on truncated rows 0.848 (min
0.575). Voting alloc on balanced: mean 0.41-0.65 s/call, p95 1.5-1.9
s/call; optimizers ~0.35-0.36 ms/call. Report:
`phase4_final_balanced/sweep_report.txt` (OVERALL: INVARIANT VIOLATIONS
PRESENT — the 38 wall-stop rows only).

## Item 2 plan (CHECKPOINTED, awaiting approval)

- Affected groups: PO 72 condition-level groups (46 driver-level, all
  lak307d except 3 arena rows; all pessimistic masks); balanced 11
  groups (den001d x9, den901d x2... i.e. 9+2 by map). Re-run all 8
  methods per group; matched pairs preserved.
- New dirs: `phase4_final_po_uncapped/`, `phase4_final_balanced_uncapped/`.
- Mechanism: add env `HVBTA_RUN_WALL_LIMIT_S` (default 600, unset =
  unchanged behavior) read once at main_simulation import; new driver
  `rerun_uncapped.py` (pattern of jv_optimum_diag.py): reads the source
  CSV, builds jobs for exactly the affected driver groups, calls
  phase4_sweep.run_group unchanged (identical code + seeds; PO driver
  groups re-run all 5 mask conditions, of which only the affected
  condition rows are WRITTEN — 576 PO + 88 BAL rows; unaffected
  condition rows are re-computed and discarded), single pool, fresh dir.
- Wall-stop value proposal: 12,000 s per run (~2.9x the largest linear
  extrapolation 4,108 s from ticks: 600 s x t_ref/t_stop, ratio p50 1.48,
  max 6.85).
- Wall-time estimate (linear extrapolation for truncated rows +
  measured exec time for the rest, 28 workers): PO 46 driver groups ~127
  h serial => ~4.5 h at 28 workers, but the FLOOR is the longest driver
  group ~20,153 s (5.6 h; one worker grinds one group); expect ~6 h.
  Balanced 11 groups ~16 h serial, floor 10,945 s (~3.0 h); expect ~3 h.
  Total ~8-9 h, run alone, PO first.
- Verify (built into rerun_uncapped.py): every non-truncated row in the
  re-run groups identical to the source sweep on all shared non-timing
  columns (PO is 86-col, pre-timeout-column; shared-name comparison),
  0 rows hit the new stop, max Execution Time reported.
- Amendment (one, covering both sweeps, with the identity-check result)
  committed AND PUSHED before any PO comparison numbers are reported
  (push explicitly ordered). Sources phase4_final_po/_balanced are NOT
  modified.

## Order after approval

PO uncapped re-run (alone, ~6 h) -> balanced uncapped (~3 h) -> item 3
profile (idle, timing measurement) -> item 8 JV diagnostic (idle) ->
item 4 full-sweep wall estimate -> full -> sweep_report after each.
Item-4 note: balanced's den001d truncations mean full's dense-map
100-robot configs (combat/combat2, den901d-class) plausibly approach or
exceed 600 s per voting run under the final stopping rule — the
estimate will be made from the uncapped re-run data before full
launches, and the user will be told first if any config could hit the
stop.
