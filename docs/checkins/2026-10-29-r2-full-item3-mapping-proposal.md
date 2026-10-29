# Check-in: R2 amendment pushed, item-3 diagnosis, combat probe, full R2 DONE clean, preregistered analysis scripts + mapping proposal (2026-10-29)

Commits this session (pushed through `89a1738`): `89a1738` (R2 + launch-rule
amendment, pushed BEFORE the probe per directive), `7aaeae1` (po_slowcall +
probe_combat drivers), `0598fed` (R2 grid in phase4_sweep + PHASE4_WORKERS
override + probe floor fix), + this check-in. The three preregistered
analysis scripts (analysis/paired_stats.py, h2_recovery.py, h3_randomness.py,
hpo.py) are WRITTEN ON DISK but NOT COMMITTED and NOT RUN - the data mapping
is pending user approval, per directive. Tests 53/53 at `0598fed`.

## Item 3 — PO slow-call diagnosis (report-only; no fix proposed)

Probe: `po_slowcall_probe.py` — one in-vivo run (lak307d 100x100 calm rep0,
low/pessimistic, borda), phase-wrapped + cProfile. 45 voting calls, 622.5 s
total alloc; the two slowest calls (176.0 s, 175.5 s) account for 57%.
Records: `po_slowcall_probe/po_slowcall_probe.json`.

- **Where the time goes: ~70% candidate generation, ~27% zero-pair
  filtering, ~3% everything else.** Generator: 23,536 calls,
  435.9 s cumulative (the per-call O(nr x nt) Python loops in
  generate_high_suitability_assignments). Zero-pair filter: 6,024,893
  invocations, 167.5 s, with **4,672,759 None-returns (77.6% of all
  candidates rejected as all-infeasible)**. Ranking is negligible: 0.03 s
  total across all 45 calls; the stall rule fires at the min-pool floor
  (pool_ranked = 6), so ranking cost is tiny — the cost is entirely in
  FILLING the pool.
- Mechanism: the pessimistic masked view is 99.4-99.8% zeros with only
  ~2% of tasks visible (pos_tasks = 0.02) — the biased generator keeps
  sampling mostly-masked pairs, candidates fail the zero-pair filter,
  and the retry loop (cap 5 x nc = 3,000 batches) burns generator +
  filter work. **Zero-entry fraction: 0.995 mean on the 20 slowest calls
  vs 0.970 on fast calls** (worst calls 0.998).
- The probe's generator wrapper fired n=0 (voting.py binds the generator
  by `from .assignments import ...` at import time); the cProfile
  attribution (function identity) is authoritative and matches the
  zero_filter wrapper totals. No fix that changes results proposed.

## Combat calibration probe -> R2 LAUNCHED (per the fixed rule)

`probe_combat.py` (direct combat single-runs, calm rep0 borda, 8 workers,
HVBTA_RUN_WALL_LIMIT_S=12000): (100,100) 71 s; (50,100) 76 s; **(100,50)
613 s**. Ratios vs the den001d analogs: 0.39 / 0.27 / 4.20 (median 0.39 —
combat is mostly FASTER than den001d except the (100,50) combo).
Calibrated R2 estimate: 35.2 h serial -> **4.4 h at 8 workers <= 12 h ->
LAUNCH** (the rule fixed in the `89a1738` amendment). Heaviest single
group ~62 min ((100,50): 6 x 613 s + optimizers).

## Full R2 sweep — DONE, ALL INVARIANTS PASS

`phase4_final_full_r2/`: 168 groups x 8 methods = 1,344 rows
(combat + combat2, diagonal + (50,100),(100,50)), 190 min wall at
8 workers. sweep_report: **0 wall stops, 0 planner failures, 0 cap hits,
0 planner-dead, 0 LaCAM timeout hits (Planner_Timeout_Count = 0 on all
1,344 rows — every row valid under the 2026-10-28 rule; NO targeted
re-runs needed)**, all collision invariants 0. Voting alloc p95 up to
~2.5 s/call; optimizers ~0.16-0.19 ms/call. Report:
`phase4_final_full_r2/sweep_report.txt`. Combat planner: ~57 ms/call,
~185 calls/run.

## Preregistered analyses — scripts written, mapping PROPOSED (approval gate)

analysis/paired_stats.py (shared machinery, h1 conventions) +
analysis/h2_recovery.py, h3_randomness.py, hpo.py — embody the PROPOSED
mapping (docstrings state it and are marked PENDING APPROVAL); NOT
committed, NOT run, no results computed.

Proposed mapping (details in the check-in message to the user):
- H2 + H3: phase4_final_hyper (the only sweep with nonzero
  Plan_Failure_Count); Murty's (H2) / perturbed JV (H3) vs each of the
  6 voting rules; metrics Plan_Failure_Count, Adaptation_Latency_Mean,
  Task_Completion_Fraction; per scenario + pooled; Holm across 6 per
  (metric, level). Completion in H2 tested as "no worse" (verdict
  rule stated in the script); in H3 the standard rule reads as
  aggregation vs randomness.
- H-PO1/2/3: phase4_final_po_merged. H-PO1: D = completion(rule) -
  completion(Murty); contrast D(high) - D(none) per policy, family = 6
  rules. H-PO2: per method (8, Holm), contrasts low/high-opt vs none on
  Mis_Assignment_Rate and Wasted_Travel_Fraction, low/high-pess vs none
  on Masked_Starvation_Count - Never_Attempted_Count. H-PO3: optimistic
  groups only; Murty's vs 6 rules + pJV (7, Holm) on
  Mis_Assignment_Recovery_Latency; -1 sentinel pairs dropped (counted).
- Secondary quality/fairness (proposed, NOT written/run): per v2 sweep
  (noise, po_merged, balanced_merged, constrained, hyper, full_r2),
  paired Wilcoxon Murty's vs each voting rule on
  Task_Completion_Fraction, Avg Reassignment Jains Index,
  Workload_Variance; Holm per (sweep, metric); per scenario + pooled;
  labeled secondary. Script: analysis/secondary_quality_fairness.py.

Machine idle. Item 8 (JV-optimum diagnostic) deferred until after full
per the user's order — full is now DONE, so item 8 is next in the idle
gap once the mapping is settled.
