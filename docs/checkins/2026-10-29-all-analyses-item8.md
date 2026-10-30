# Check-in: mask check, all preregistered analyses run (H2, H3, H-PO1/2/3, secondary), item 8 full run (2026-10-29)

Commits this session: `edf5843` (all five analysis scripts + mask_check_t0 +
item-8 validity tracking, committed BEFORE running, per directive),
`7b63963` (H-PO1 driver-key bug fix, re-run clean), + this check-in.
Full outputs: h2_results.txt, h3_results.txt, hpo_results.txt,
secondary_results.txt (repo root, regenerable via the scripts).

## 1. Mask check (report-only; ran before everything)

Tick-0 allocator view, mean over 36 lak307d groups (9 sizes x 4 scenarios,
rep 0):

| condition | zero_frac | pos_task_frac | pos_robot_frac |
|---|---|---|---|
| none | 0.791 | 1.000 | 1.000 |
| low-opt | 0.728 | 1.000 | 1.000 |
| low-pess | 0.884 | 0.568 | 1.000 |
| high-opt | 0.587 | 1.000 | 1.000 |
| high-pess | 0.964 | 0.187 | 0.900 |

VERDICT: the stop-condition is NOT met - the spec itself anticipates this.
partial_observability_experiment.md §4 tabulates
P(>=1 hard-fail field masked) = 0.62 (low) / 0.95 (high); the measured
tick-0 invisible-task fractions (0.43 low-pess / 0.81 high-pess) match the
spec's own collision probabilities (slightly below them because max-value
pessimistic substitution does not hard-fail the strongest robots).
The user's accumulation expectation is PARTIALLY REFUTED: the late-run
99.5% sparsity is mostly STATIC (tick-0 high-pess is already 96.4% zeros
with 18.7% of tasks visible); accumulation adds the remainder (mean pool
19.1 -> 94.9 tasks, masked starvation 122.5/160.5 spawned - which is the
mechanism H-PO2b's pessimistic arm is about). Records:
mask_check_t0/mask_check_t0.json.

## 2. H2 (recovery, phase4_final_hyper): NOT SUPPORTED, any rule, any level

Planner-dead runs (150 failed-call budget; included in tests):
Borda 9, Approval 14, MJ 12, Cumulative 7, Condorcet 18, Range 11,
Murty's 5, pJV 9 (85 of 1,920).
- Plan_Failure_Count (n=240): all 6 significant but median 0 ->
  "not supported (median=0)" per the verdict rule (voting fails fewer
  calls in aggregate, but the typical paired difference is zero).
- Adaptation_Latency_Mean: Borda and Condorcet significant REVERSE
  (Holm p 1.2e-7 / 5.4e-1... pooled: Borda -0.058, Condorcet -0.055;
  voting has HIGHER latency); rest significant at median 0.
- Task_Completion_Fraction: all 6 significantly WORSE
  (median -0.078 to -0.098, Holm p ~1e-28) -> "not supported (worse)".
- H2 OVERALL: not supported for all 6 rules at every level. Per-scenario:
  overload is catastrophic for voting (pJV-vs-voting completion gap
  -0.21 to -0.28 there; see H3).

## 3. H3 (aggregation vs randomness, phase4_final_hyper): voting shows NO
advantage over perturbed JV anywhere

- Failures pooled: medians 0 everywhere -> not supported. In OVERLOAD,
  pJV has significantly FEWER failures than every voting rule
  (median -15.5 to -46, Holm p <= 2e-7) - significant REVERSE.
- Adaptation latency pooled: Borda/Condorcet significant REVERSE; in
  overload all 6 REVERSE (pJV lower latency by 0.07-1.25).
- Completion pooled: pJV significantly better than all 6 rules
  (median +0.073..+0.090 in pJV's favor, Holm p ~1e-26); in overload
  pJV is better by 0.21-0.32 (all Holm p ~1e-10, W=0 in places).
- Reading per the plan's logic: no aggregation story exists - voting
  never beats Murty's on hyper (H2) and does not beat the randomized
  baseline either; on the failure-stress sweep the randomized optimizer
  (pJV) is the best method overall.

## 4. H-PO (phase4_final_po_merged)

H-PO1 (advantage grows with masking, completion):
- OPTIMISTIC axis: not supported (medians 0; some significant-at-0).
- PESSIMISTIC axis: SUPPORTED for ALL 6 rules - the completion gap
  (rule - Murty's) grows by +0.007..+0.016 from none to high-pess
  (Holm p <= 7.5e-23). NOTE the honest nuance: the "advantage" D is
  NEGATIVE at no-mask (-0.031..-0.041) and shrinks toward zero under
  pessimistic masks (-0.004..-0.0008 at high-pess) - voting's
  disadvantage narrows; it never becomes a positive advantage.

H-PO2a (optimistic masking harms): SUPPORTED for every one of the 8
methods, both contrasts, both metrics (Mis_Assignment_Rate low-opt +0.012
to +0.022, high-opt +0.054 to +0.076; Wasted_Travel_Fraction similar;
all Holm p <= 2e-50).

H-PO2b (pessimistic starvation above the Never_Attempted baseline):
NOT SUPPORTED - significant REVERSE for all methods at high-pess
(median -2 to -3) and for the optimizers at low-pess. STRUCTURAL
FINDING: Masked_Starvation_Count <= Never_Attempted_Count on all 14,880
rows (masked-starved tasks are a subset of never-attempted tasks), so
the plan's derived metric is always <= 0 and cannot be positive by
construction; the test as specified can only produce not-supported or
reverse. Reported as-is per "report every result".

H-PO3 (recovery latency vs Murty's, optimistic groups): -1 semantics
reported (from finalize(): -1 = no recovery events recorded = either no
mis-assignment or mis-assignments never recovered; disambiguated via
Mis_Assignment_Count). Never-recovered-any runs flagged per method:
Borda 2, Approval 4, MJ 10, Cumulative 5, Condorcet 1, Range 4, pJV 2,
Murty's 4. TEST: significant REVERSE for all 6 voting rules - Murty's
recovers 3-10 ticks FASTER (median -4.1 to -6.8 pooled; worst in
overload, -8.4 to -10.1); pJV vs Murty's median 0 (Holm p 1.4e-5,
"not supported (median=0)"). H-PO3 not supported.

## 5. Secondary quality/fairness (labeled SECONDARY)

- Task_Completion_Fraction: Murty's significantly higher than EVERY
  voting rule in ALL SIX sweeps, pooled (medians: noise +0.006..+0.019,
  po-none +0.008..+0.020, balanced +0.009..+0.018, constrained +0.031..
  +0.045, hyper +0.078..+0.099, full R2 +0.019..+0.039; all
  Holm-significant).
- Avg Reassignment Jains Index: mixed. Murty's slightly higher (more
  fair) than Borda/Cumulative/Condorcet/Range in most sweeps; Approval
  significantly MORE fair than Murty's in balanced (+0.002),
  constrained (+0.006), hyper (+0.006) - reverse-significant.
- Workload_Variance: voting LOWER (more even) than Murty's in
  constrained (+0.09..+0.16 Murty-higher, all significant), hyper
  (+0.46..+0.62), full R2 (+0.04..+0.08 for 4 of 6 rules);
  noise/po/balanced not significant.

## 6. Item 8 (JV-optimum diagnostic, full size, exploratory)

64/64 groups serial (worker count 1, flip-safe), ZERO planner failures on
every group (validity rule clean, no re-runs needed). 11,777 non-urgent
voting call records. Winner/JV-optimum ratio by unassigned-robot bin:
p50 0.90-1.00 (33-64 bin: 0.984, n=3023), p95 = 1.0 in every bin; ~48%
of calls exactly at the optimum. Per-method p50: Range 1.000, Condorcet
0.994, Borda 0.979, Approval 0.954, Cumulative 0.880, MJ 0.820.
Records: jv_optimum_diag/jv_optimum_records.json.

## Overall picture (every hypothesis reported as preregistered)

H1 not supported (reversed). H2 not supported. H3 not supported (pJV
dominates voting on the stress sweep). H-PO1 supported on the
pessimistic axis only (gap narrows, never positive). H-PO2a supported
universally; H-PO2b not supported (structurally untestable as
specified). H-PO3 not supported (reverse). Secondary: Murty's wins
completion everywhere; fairness split (voting better on workload
variance, Approval better on Jain's index; Murty's better on Jain's
vs most rules).

Machine idle. Awaiting the user's call on next steps.
