# Analysis Plan

Written and committed before running the sweeps it covers. Changes go in a
dated Amendments section at the end; the original text is never edited.

## Data used

Only the v2 sweeps count toward these tests: the noise experiment,
hyper-constrained, the PO sweep, constrained, maps_balanced2, and Full-Sweep,
each re-run into a new `<name>_v2_<timestamp>/` directory. Pilot runs are for
validation only.

## Methods compared

Six voting rules (Borda, Approval, Majority Judgment, Cumulative, Condorcet,
Range), Murty's K-best (built on JV), and perturbed JV.

Perturbed JV's noise level σ_pjv is chosen by this rule, fixed now: on the
pilot map, try σ_pjv ∈ {0.02, 0.05, 0.1, 0.2} and use the smallest value whose
duplicate-retry rate is below 10%. Completion results are not looked at
during selection. The chosen value and the duplicate-retry rates are added in
an Amendment before any sweep counted here is run.

Ordinal/median rules: Borda, Majority Judgment, Condorcet.
Cardinal methods: Range, Murty's, perturbed JV.

## Hypotheses and predictions

H1, noise robustness (primary). As observation noise σ_obs increases,
completion degrades faster for cardinal methods than for ordinal/median
rules. Prediction: the drop in Task_Completion_Fraction from σ_obs = 0 to
σ_obs = 0.3 is larger for Murty's than for each of Borda, Majority Judgment,
and Condorcet.

H2, recovery. Voting has fewer plan failures and lower adaptation latency
than Murty's, with completion no worse than Murty's.

H3, aggregation vs. randomness. Voting's recovery advantage over Murty's also
holds against perturbed JV. If perturbed JV matches voting, the advantage
comes from randomness, not aggregation.

H-PO1. Voting's advantage over Murty's grows as masking increases.
H-PO2. Optimistic masking raises mis-assignment rate and wasted travel;
pessimistic masking raises starvation above the Never_Attempted baseline.
H-PO3. Voting and perturbed JV have lower Mis_Assignment_Recovery_Latency
than Murty's.

Quality and fairness claims (completion, Jain's index, Workload_Variance) are
re-tested on the v2 data as secondary results.

## Metrics

Primary: Task_Completion_Fraction, Adaptation_Latency_Mean,
Plan_Failure_Count.
Secondary: Mis_Assignment_Rate, Wasted_Travel_Fraction,
Masked_Starvation_Count minus Never_Attempted_Count,
Mis_Assignment_Recovery_Latency, True_Assignment_Suitability, Jain's index,
Workload_Variance, Throughput_Mean.

## Statistical tests

- All method comparisons are paired. A pair is two methods on the same group
  (same map, robot count, task count, rep, scenario, and condition), which
  share robots, tasks, schedule, masks, and noise.
- Test: Wilcoxon signed-rank on per-group differences.
- Effect size: median paired difference with a bootstrap 95% confidence
  interval (10,000 resamples).
- H1 is tested on the per-group degradation (value at σ_obs = 0 minus value
  at σ_obs = 0.3), compared between Murty's and each ordinal rule.
- Multiple comparisons: Holm-Bonferroni within each hypothesis family.
  α = 0.05.
- Results are reported per scenario and pooled.

## Reporting

Every hypothesis above is reported, whether supported or not. No hypothesis
is dropped or replaced after the data is seen. Exploratory analyses are
labeled as exploratory.

## Amendments

(none yet)

### 2026-10-23: suitability scoring change (distance and proximity factor)

The balanced suitability scores saturated: over 90% of feasible pairs
scored at or above 0.95 and the median was exactly 1.0 (measured on the
lak102d P2 pilot, all scenarios; pooled unclipped median 1.69). The cause
was twofold: the proximity factor multiplied scores by up to 2 for nearby
robots (2 / (1 + d/map_size)), pushing most feasible pairs past the 1.0
cap, and the distance it used was the robot's stale planned-path length
(or an un-absoluted Manhattan fallback that could even go negative)
rather than the actual robot-task distance.

Change: distance is now the true shortest-path (BFS) distance between the
robot's current location and the task location, computed once per task at
creation, with an absolute-value Manhattan fallback when no map is
available; the proximity factor is now 1 / (1 + d/map_size), bounded in
(0.5, 1]. After the change, unclipped feasible scores spread across
(0.3, 0.95) with a median near 0.6 and nothing above 1.0.

This matters for H1: the noise experiment assumes scores are graded
(continuous, spread out) so observation noise has something to perturb.
With scores piling at the 1.0 cap, noise of any size would barely change
the ranking of pairs and H1 could not be meaningfully tested. Because the
scoring changed, all pilots run before 2026-10-23 (pilot_po_P1_*,
pilot_po_P2_all_voting_fixes, the partial pilot_po_P2_threshold_off) used
the saturated scorer and will be re-run; the pilot repetition count was
also reduced to 1 for faster iteration. Full sweeps keep 3 repetitions.

### 2026-10-23: unreachable robot-task pairs score 0

Follow-up to the scoring amendment above, which did not spell this out:
when the BFS distance map says the robot's map region has no path to the
task at all, the pair scores 0 - the robot can never reach it, so the
pair is treated as infeasible for allocation (the same effect as a
hard-fail). At execution time this case cannot arise: the feasibility
check runs when the robot is standing on the task (distance 0).

### 2026-10-25: execution-layer collision safety (tick order through replan on deviation)

This amendment records every execution-layer change since the plan was
written, so the provenance of all post-v9 sweep data is explicit. None of
these changes alter allocation objectives, event schedules, seeds, or
metrics; they fix physical-collision correctness in the simulator.

1. **Tick order**: dynamic events, then replan, then the MAPF planner call,
   then movement - movement is always last, on a plan that already accounts
   for this tick's events.
2. **Idle agents**: every unassigned robot joins every planner call as a
   stay-put agent (never a hard obstacle); LaCAM may displace them.
3. **Step-aside goals**: an idle agent standing on a mover's goal cell gets
   a nearest-free-cell goal searched over the whole reachable map.
4. **Two-phase synchronous movement**: all next cells are computed before
   anyone moves; vertex/swap conflicts in the intended set are counted
   (Planned_Conflict_Count) as an invariant detector, not patched.
5. **Transitive waiting (v7)**: robots without a usable path stay put; any
   robot whose next cell is occupied by a staying robot also stays put
   (fixed point). Forced waits are counted in Propagated_Wait_Count.
6. **Three movement bugs fixed (v7)**: (a) planned waits were never
   consumed, freezing robots mid-route and desynchronizing the joint plan;
   (b) a remaining_distance force-zero made robots work tasks one cell
   early and teleport onto task cells without updating the occupied set;
   (c) pathless reassigned robots (remaining_distance 0 after unassign)
   passed the work gate and teleported onto new tasks after failed replans.
   Work now requires the robot to be physically at the task location.
7. **Mover-mover resolution + replan on deviation (v8/v9)**: two movers
   targeting one cell, or exchanging cells, are resolved deterministically
   (lower robot ID keeps its move; Mover_Conflict_Waits counts the other).
   Any off-plan wait or plan-less assigned robot sets a deviation flag
   that forces a planner call at the top of the next tick, before movement.
8. **Occupied-set race fixed (v9)**: the occupied-cell set is updated
   two-pass (discard all vacated, then add all entered) so a robot
   entering a cell vacated in the same synchronous move no longer loses
   it; the set previously decayed over a run and arrival nudges placed
   new robots onto occupied cells.
9. **Safety nudge removed (v9)**: after the fixes above, the pilot shows
   0 planned conflicts, 0 nudges, 0 violations, and 0 plan failures over
   1680 runs; the last-resort nudge never fired and was masking-only.
   The end-of-tick MAPF-validity check remains as a hard invariant
   (Collision_Violation_Count must be 0 in every sweep used for analysis).

CSV: two columns were added for these mechanisms
(Propagated_Wait_Count, Mover_Conflict_Waits; see CSV_COLUMNS.md 12c-2).
All result directories at pilot_po_P3_collision_v6 and earlier predate
these fixes and must not be used for collision-sensitive analysis; the
final ablation arms (pilot_po_P3_collision_v9 = zero_pair on,
pilot_po_final_v9_zero_pair_off, pilot_po_final_v9_threshold_off) are the
first fully clean data.

### 2026-10-25: deterministic anytime budget (replaces the 20 ms wall-clock budget)

The urgent-path anytime budget in voting (20 ms wall-clock, in effect since
the anytime design landed) made every urgent voting call load-dependent:
candidate pools ranked 1-120 candidates depending on machine speed, so
voting rows were not bit-reproducible across runs, process counts, or
parallel load (measured: 42/140 rows differing between two same-code serial
runs). Replaced by a deterministic candidate-count cap,
VOTING_URGENT_CANDIDATE_BUDGET = 3, chosen as the MEDIAN ranked-candidate
count over 20,303 urgent calls measured serially on the v9 pilot config
(distribution: p25 1, p50 3, p75 8, p95 17, mean 5.6). The score threshold
(0.7, coverage-normalized) was already deterministic and is unchanged.
After the change, pool-vs-serial output is bit-identical on all rows
(voting and Murty's), excluding wall-clock timing columns. LaCAM budget:
verified ZERO failures of any kind (in particular no lacam_timeout) with 30
concurrent workers on the largest configs (combat, combat2, den901d at 100
robots) at the 1 s limit, so the limit stays 1.0 s. All sweeps run under
this deterministic regime; pre-2026-10-25 voting rows carry anytime-budget
noise and are not bit-comparable to later rows.

### 2026-10-26: perturbed JV baseline's sigma_pjv = 0.05 (chosen by the duplicate-retry rule)

Step 8.2 requires choosing sigma_pjv before seeing any completion results,
as the smallest candidate in {0.02, 0.05, 0.1, 0.2} whose duplicate-retry
rate is below 10%. Measured on the pilot map (lak102d, 16 group matrices:
[10,20]x[10,20] x 4 scenarios, rep 0): for each matrix, two independent
noise draws were solved with JV; a retry is a duplicate when both draws
yield the identical assignment. Rates: sigma 0.02 -> 4/16 = 25.0%,
sigma 0.05 -> 1/16 = 6.2%, sigma 0.1 -> 1/16 = 6.2%, sigma 0.2 -> 0/16 =
0.0%. CHOSEN: sigma_pjv = 0.05 (smallest below 10%). No completion
numbers were read for this choice. sigma_pjv is the baseline's own
internal randomness, seeded per (run seed, replan index) with fresh draws
on every retry; it is never mixed with sigma_obs in code or columns.

### 2026-10-26: deterministic planner-work bounds for the hyper-constrained PP sweep (replaces wall-clock exposure)

The hyper-constrained PP pilot (pilot_hyper_pp_v9, 768 runs) hit the 600 s
per-run safety stop on 45 runs (by method: borda 8, MJ 6, approval 5,
condorcet 6, cumulative 3, range 4, Murty's 4, perturbed JV 9;
concentrated in overload/task_churn on 15-robot configs). Measurement of
the capped runs (per-call planner timing + A* expansion instrumentation,
re-running the capped configs): successful PP plan() calls cost < 25 ms
and need at most ~178k total A* expansions, while FAILING calls exhaust
the time-expanded search space - up to 3.3M expansions, 1-10 s each
(lak103c 15x10: 80 failed calls averaging 7.7 s; hrt003d 15x15: 492
failures x ~1 s). PP itself has no wall-clock limit (the wrapper threads
time limits only to LaCAM), so run time was unbounded in both per-call
work and per-run call count.

Bounds adopted (all COUNT-based, deterministic, applied identically to
every allocation method; every firing is counted per run in the new
Planner_Cap_Hits CSV column; Planner_Call_Count and Planner_Total_ms
columns record per-run planner-call accounting):
- PP_MAX_EXPANSIONS = 250,000 A* expansions per plan() call, shared
  across all priority orders and agent searches of that call (1.4x the
  largest expansion count observed for a successful call, so it only
  converts slow failures into fast ones);
- PP_RUN_MAX_EXPANSIONS = 35,000,000 A* expansions per simulation run
  (the hard bound: at the measured ~100k expansions/s under 30-worker
  pool load this caps a run's planner time near 350 s; normal heavy
  hyper runs use ~15-25M);
- PLANNER_FAILED_CALL_BUDGET = 150 failed planner calls per run, after
  which the run is planner-dead (further planner calls skipped, stale
  paths accepted);
- PLANNER_CALLS_PER_TICK = 4 planner invocations per tick (the replan
  call plus replan-on-failure alternative retries).
The 600 s RUN_WALL_LIMIT_S stays as a SAFETY STOP only. Pilot re-run on
the bounded code (pilot_hyper_pp_v11, same 768-run config): ZERO runs hit
the wall stop (max 372 s, p99 280 s), all collision invariants 0, and
the planner-failure stress phenomenon is preserved (15,173 failures,
mean 19.8/run; completion profile matches the unbounded pilot: Murty's
0.736, perturbed JV 0.724, voting 0.615-0.645).

### 2026-10-26: urgent-call stopping rule (candidate cap removed; threshold to become f * UB)

The anytime behavior of voting under urgency is changed before the Phase-4
sweeps, in two steps:
1. EFFECTIVE NOW: the urgent candidate cap (VOTING_URGENT_CANDIDATE_BUDGET
   = 3, the 2026-10-25 amendment) is REMOVED from the main configuration.
   Under urgency, voting stops when the good-enough threshold fires (with
   the MIN_POOL_BEFORE_THRESHOLD = 5 floor) or when the pool is exhausted.
   The capped arm remains available for the cap ablation only
   (HVBTA_URGENT_CAP=1): six voting methods, lak102d, 3 reps,
   sigma_obs = 0, compared against the main noise sweep's sigma_obs = 0
   rows (configs identical except the cap).
2. PENDING USER CHOICE (f): the good-enough threshold becomes
   stop when a candidate's coverage-normalized quality >= f * UB, where
   UB = min(UB_robot, UB_task) is computed per allocation call from the
   scores the allocator sees (UB_robot = mean of the top max_feasible
   values among each robot's best feasible score; UB_task likewise over
   tasks; max_feasible = min(#robots with >= 1 positive task, #tasks with
   >= 1 positive robot)). f is a single value or a function of the call's
   robot/task counts, chosen from the calibration report (v9 pilot
   config + one rep of den901d 100x100; NO completion numbers were read
   for the calibration). f is set via HVBTA_UB_F. No Phase-4 sweep runs
   until f is set.
Every sweep's report includes mean allocation time per call by method:
(Planning_Time_us + total_reassignment_time) / (1 + total_reassignments).

### 2026-10-27: anytime stopping rule finalized as a stall rule (k = 5); tiered f * UB rejected

This closes part 2 of the 2026-10-26 stopping-rule amendment. The f * UB
good-enough threshold is NOT adopted. The 4c calibration showed the
coverage-normalized winner/UB ratio is ~1.0 (median) for small calls but
0.4-0.6 for large calls, so any f that matters would have to be tiered by
call size — several fitted constants with thin calibration support in the
large-call regime (few large calls per sweep, and the calibration for them
came from a single den901d 100x100 rep). The tiered f is rejected as
over-fitted.

INSTEAD, the main configuration's urgent stopping rule is a STALL rule:
under urgency, voting ranks candidates in pool order and stops once the
best-so-far coverage-normalized candidate quality has not improved for
k = 5 consecutive candidates, but never before MIN_POOL_BEFORE_THRESHOLD
= 5 candidates have been ranked (k equals the existing minimum-pool
floor — one constant, no fitting), or when the pool is exhausted,
whichever comes first. Non-urgent calls are unchanged (full pool). The
UB/f * UB code remains in the codebase but unused by the main
configuration. HVBTA_STALL=0 disables the rule (the anytime-ablation
arm, item 5 of the final scope: full pool on urgent calls).

Also in this amendment:

1. Planner-dead behavior (documenting what the 2026-10-26 bounds do after
   PLANNER_FAILED_CALL_BUDGET = 150 failed planner calls is exhausted):
   the run becomes planner-dead — every further planner invocation (the
   tick's replan call AND all replan-on-failure alternative retries) is
   skipped and counted in Planner_Cap_Hits. Allocation is NOT skipped:
   robots keep being (re)assigned every tick as usual. Robots keep moving
   along the stale paths from the last successful plan (they do not stop
   and are not teleported); robots with no path stay put. Work still
   requires physical presence at the task location, so robots whose
   stale path reaches their assigned task still complete it. The run ends
   only at the tick horizon or when all tasks complete, as in every other
   run.

2. Driver-consistency bug fix (justified as a bug): the serial __main__
   driver defaulted PERTURBED_JV_SIGMA to None (JV unperturbed) unless
   HVBTA_PJV_SIGMA was set, while the pool driver (phase4_sweep.run_group)
   hard-sets 0.05 per the 2026-10-26 sigma_pjv amendment. The serial
   default is now 0.05, matching the pool driver and the amendment; the
   env override remains. (Found by the item-2 pool-vs-serial check: all
   perturbed_jv rows mismatched with the old default; after the fix all
   160 rows x 2 paths match bit-for-bit, excluding timing columns.)

