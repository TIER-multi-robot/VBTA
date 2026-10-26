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
