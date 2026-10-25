# Simulation CSV output reference

Every row in a simulation-results CSV corresponds to a single
`(map, num_robots, num_tasks, num_candidates, suitability_method, planner,
scenario, rep, method, mask_condition)` run. This document describes the
76-column CSV that `main_simulation.py` produces (v7: 51 legacy + 20
partial-observability/pool-diagnostic columns + 5 collision/limit columns).

For each metric we note: **what it measures**, **whether high or low is
better**, and **what claim it backs** (see `CLAIMS_AND_EVIDENCE.md`).

---

## Column order (matches the CSV header exactly)

The row is assembled as:
`row_prefix + list(output_tuple) + config_suffix + [map_size]`

| Component | Items | Source |
|---|---|---|
| `row_prefix` | 14 | `_record_assignment` (13 items: identity + initial quality + initial fairness) with `Planner` injected at position 4 |
| `output_tuple` | 53 | `benchmark_simulation` = `main_simulation` return (52: 48 pre-collision + 4 collision/limit incl. `Propagated_Wait_Count`) + `execution_time` (1) |
| `config_suffix` | 8 | Scenario + 4 raw rates + 3 PO config columns, appended at the writer callsite |
| `map_size` | 1 | Appended at the writer callsite |
| **Total** | **76** | |

---

## 1. Row identity (7 columns)

| # | Column | What it measures | Direction | Notes |
|---|---|---|---|---|
| 1 | `Run ID` | Sequential per-repetition ID | — | Grouping key |
| 2 | `Method` | Allocation method under test | — | Voting rule, optimizer, or direct-LLM |
| 3 | `Suitability Method` | Scorer that built the suitability matrix | — | balanced / loose / strict / LLM |
| 4 | `Planner` | MAPF planner for this run | — | cbs / pp / pibt — config axis |
| 5 | `Num Robots` | Initial robot count | — | Swept axis |
| 6 | `Num Tasks` | Initial task count | — | Swept axis |
| 7 | `Num Candidates` | Candidate pool size for voting | — | `max(robots, tasks) * 5` |

## 2. Initial allocation quality (3 columns)

Computed once, before the simulation starts.

| # | Column | What it measures | Direction | Claim backed |
|---|---|---|---|---|
| 8 | `Total Score` | Sum of suitability over initial (robot, task) pairs | **Higher** = better assignments | Claim 1 (voting ≈ optimization on quality) |
| 9 | `Mean_Assignment_Suitability` | `Total Score / assigned_count` — mean per-pair quality | **Higher** = better | Claim 1 |
| 10 | `Planning_Time_us` | Wall-clock microseconds to produce the initial allocation | **Lower** = faster | Claim 4 (voting scales better) |

## 3. Initial fairness snapshot (4 columns)

Computed once from the per-agent suitability scores of the initial assignment.

| # | Column | What it measures | Direction | Claim backed |
|---|---|---|---|---|
| 11 | `Init_Jains_Index` | Jain's fairness index (1.0 = perfectly equal, 1/n = maximally unequal) | **Higher** = more fair | Claim 2 (voting is more fair) |
| 12 | `Init_Gini` | Gini coefficient (0 = equal, 1 = maximally unequal) | **Lower** = more fair | Claim 2 |
| 13 | `Init_Median` | Median suitability score | Contextual | Robust central tendency; not directional |
| 14 | `Init_IQR` | Interquartile range Q3−Q1 | **Lower** = more consistent | Claim 2 (less spread = more fair) |

## 4. Aggregate simulation outcome (6 columns)

| # | Column | What it measures | Direction | Claim backed |
|---|---|---|---|---|
| 15 | `Task_Completion_Fraction` | `total_success / total_tasks` — fraction of all spawned tasks that completed | **Higher** = better | Headline outcome metric |
| 16 | `total_reassignment_time` | Sum of planning_time across every replan (microseconds) | **Lower** = faster adaptation | Claim 4 (voting scales better) |
| 17 | `Cumulative_Reassignment_Quality` | Sum of suitability totals across every replan | **Higher** = better lifelong quality | Claim 1 (lifelong quality) |
| 18 | `total_reassignments` | Count of replan events | Contextual | More events = more dynamic stress; not directional |
| 19 | `Total Time Steps` | Timesteps the loop actually ran (may be < max_time_steps on early exit) | Contextual | Shorter runs with high completion = efficient; shorter with low completion = failure |
| 20 | `Avg Reassignment Jains Index` | Mean Jain's index across every replan's post-assignment fairness snapshot | **Higher** = more fair over time | Claim 2 (lifelong fairness) |

## 5. Lifelong-operation metrics (4 columns)

| # | Column | What it measures | Direction | Claim backed |
|---|---|---|---|---|
| 21 | `Attempted_Completion_Rate` | `total_success / len(tasks_ever_assigned)` — fraction of tasks the method *tried* that completed | **Higher** = better | Insulates from late-arriving tasks |
| 22 | `Tasks_Attempted` | Count of unique tasks ever bound to a robot | Contextual | Shows engagement; low = method left tasks unassigned |
| 23 | `Tasks_Spawned` | `total_tasks` at run end (initial + dynamic additions) | Contextual | Constant across methods in a group |
| 24 | `Completion_Step` | First timestep at which the task pool hit 0; −1 if never | Contextual | Early = cleared fast; −1 = never cleared (overload or slow method) |

## 6. Primary comparative metrics (5 columns)

These are the metrics most likely to expose voting's advantages over optimization.

| # | Column | What it measures | Direction | Claim backed |
|---|---|---|---|---|
| 25 | `Throughput_Mean` | Mean task completions per elapsed timestep | **Higher** = more productive | Claim 1 (competitiveness) |
| 26 | `Throughput_Var` | Variance of per-step completions | **Lower** = steadier | Claim 3 (voting adapts faster — steady throughput under churn) |
| 27 | `Adaptation_Latency_Mean` | Mean timesteps from each dynamic event to the next successful planner resolution | **Lower** = adapts faster | Claim 3 (voting adapts faster under churn). **CONFOUND**: includes planner wall-time — see note below |
| 28 | `Adaptation_Latency_Max` | Worst-case latency across the run | **Lower** = better worst case | Claim 3 |
| 29 | `Workload_Variance` | `np.var` of per-robot task-success counts at run end | **Lower** = more evenly distributed | Claim 2 (voting is more fair) |

**Latency confound**: `Adaptation_Latency` currently includes planner wall-time
because the planner runs inside the replan tick before `pending_event_steps` is
drained. PP/PIBT are microsecond-fast; CBS can take seconds. So "voting adapts
faster" may partly be "voting's planner happens to be faster." When comparing
voting vs optimization, hold the planner fixed.

## 7. Fair-comparison denominators (2 columns)

Constant across every method + planner in a group. Use these as common
denominators when comparing methods that ran for different numbers of steps.

| # | Column | What it measures | Direction | Notes |
|---|---|---|---|---|
| 30 | `Full_Schedule_Tasks_Spawned` | Initial `num_tasks` + every `add_tasks` in the full event schedule | — | Constant per group |
| 31 | `Full_Schedule_Robots_Max` | Initial `num_robots` + every `add_robots` in the full event schedule | — | Constant per group |

## 8. Urgency + pressure bucketing (3 columns)

Recorded per replan and summarized at run end. Lets you slice results by
operating pressure: *"under what pressure did each method perform how?"*

| # | Column | What it measures | Direction | Notes |
|---|---|---|---|---|
| 32 | `Percentage_Urgent_Mean` | Mean fraction of unassigned tasks that were urgent at each replan | Contextual | Range [0, 1]; higher = more pressure |
| 33 | `Percentage_Urgent_Max` | Worst-case urgent fraction across the run | Contextual | Range [0, 1] |
| 34 | `Robot_Pressure_Fraction` | Fraction of replans that fired in the same tick as a new-robot event | Contextual | Range [0, 1]; higher = more fleet churn |

**Bucketing recipe:**
- `Percentage_Urgent_Mean == 0 AND Robot_Pressure_Fraction == 0` → *"none urgent, no pressure"*
- `Percentage_Urgent_Mean > 0 AND Robot_Pressure_Fraction == 0` → *"some urgent, no pressure"*
- `Percentage_Urgent_Mean == 0 AND Robot_Pressure_Fraction > 0` → *"none urgent, robot pressure"*
- `Percentage_Urgent_Mean == 1 AND Robot_Pressure_Fraction > 0` → *"all urgent, robot pressure"*

Voting uses these signals live: if either urgency or robot pressure is nonzero
at a given replan, voting gets `time_budget_ms=20` and `score_threshold=0.7`
(anytime termination). Optimization methods ignore them.

## 9. Reassignment fairness snapshot (3 columns)

Same as Section 3 but averaged across every replan event instead of computed
once at initial assignment. Jain's for the reassignment block is already
exposed as `Avg Reassignment Jains Index` (column 20).

| # | Column | What it measures | Direction | Claim backed |
|---|---|---|---|---|
| 35 | `Avg_Reass_Gini` | Mean Gini across every post-replan snapshot | **Lower** = more fair over time | Claim 2 (lifelong fairness) |
| 36 | `Avg_Reass_Median` | Mean of medians across replans | Contextual | Robust central tendency |
| 37 | `Avg_Reass_IQR` | Mean IQR across replans | **Lower** = more consistent | Claim 2 |

## 10. Empirical regime classification (2 columns)

Computed per run from the per-timestep pool-size log. Lets you verify the regime
actually behaved as intended, not just as configured.

| # | Column | What it measures | Direction | Notes |
|---|---|---|---|---|
| 38 | `Mean_Task_Pool_Size` | Mean of `len(tasks)` across every timestep the sim ran | Contextual | High = backlog (overload); low = keeping up |
| 39 | `Pool_Size_Trend` | Mean pool size in last quartile − mean in first quartile | **Near 0** = steady state; **strongly positive** = overload (backlog growing); **negative** = draining | Most important regime-validation metric |

**How to use for tuning:** if `Pool_Size_Trend` is ~0 for "calm" but >0 for
"overload," the regimes are working. If "calm" is also trending up, the calm
rates are too hot. If "overload" is flat, it isn't overloading — raise
`task_arrival_rate` in the `REGIMES` dict.

## 11. Horizon provenance (2 columns)

So the horizon choice is auditable when a reviewer asks why horizons differ
across regimes.

| # | Column | What it measures | Direction | Notes |
|---|---|---|---|---|
| 40 | `Scheduled_Event_Count` | Number of scheduled events (arrivals + departures + cancellations) actually in the truncated schedule for this run | Contextual | Should be ≈ K (50) for calm; higher for churn |
| 41 | `Horizon_Binding_Term` | Which of the three floors set the horizon | — | `"transient"`, `"events"`, or `"travel"` |

**How to use:** calm should bind on `"events"` (needs more steps to reach K
events). Large maps should bind on `"travel"`. If calm binds on `"transient"`,
the transient floor is too high relative to the event count.

## 12. Overhead (1 column)

| # | Column | What it measures | Direction | Claim backed |
|---|---|---|---|---|
| 62 | `Execution Time` | Wall-clock nanoseconds for the whole `main_simulation` call | **Lower** = faster | Claim 4 (voting scales better) |

## 12b. Partial-observability outcome metrics (12 columns)

From the partially-observable task-requirements experiment
(`partial_observability_experiment.md`). These columns always exist; when
the PO machinery is inactive (legacy sweeps) counts are 0 and rates/means
are **-1** (N/A sentinel).

On arrival at a task the simulator reveals the task's mask (once) and runs
balanced's hard-fail checks (`balanced_hard_fail`) against the TRUE
attributes. A failed check is a **mis-assignment**: the robot is freed at
its current location, the task returns to the pool unmasked, and a replan
event is raised through `pending_event_steps`.

| # | Column | What it measures | Direction | Notes |
|---|---|---|---|---|
| 45 | `Mis_Assignment_Count` | Arrivals where the execution-time feasibility check failed | **Lower** = better | Must be exactly 0 at `Mask_Level = none` (sanity check, §6.4) |
| 46 | `Mis_Assignment_Rate` | `Mis_Assignment_Count / arrivals` | **Lower** = better | -1 when no arrivals |
| 47 | `Wasted_Travel_Steps` | Travel steps spent on assignments that failed at arrival | **Lower** = better | Counted from assignment to arrival |
| 48 | `Wasted_Travel_Fraction` | `Wasted_Travel_Steps / total robot travel steps` | **Lower** = better | -1 when no travel |
| 49 | `Masked_Starvation_Count` | Tasks still masked at run end that were never assigned | **Lower** = better | The pessimistic-policy cost (H-PO2) |
| 50 | `Mean_Masked_Wait` | Mean ticks a task spent in the pool while masked | **Lower** = better | Over all ever-masked tasks; -1 when none |
| 51 | `Reveal_Events` | Number of masks removed | Contextual | Arrivals at still-masked tasks |
| 52 | `Mis_Assignment_Recovery_Latency` | Mean ticks from a mis-assignment to the task's next reassignment | **Lower** = recovers faster | H-PO3; -1 when no mis-assignments |
| 53 | `True_Assignment_Suitability` | Mean balanced suitability of executed assignments on TRUE attributes | **Higher** = better | How soft-field masking degrades quality (H-PO1) |
| 54 | `Masked_View_Suitability` | Same assignments scored on the masked view | Contextual | Gap vs column 53 shows how much the policy misjudged |
| 55 | `Nav_Drop_Count` | Unassignments from the pre-existing navigation runtime check | Contextual | §6.5; counted at `none` too (pre-existing baseline) |
| 56 | `Nav_Drop_Wasted_Travel` | Travel steps spent on pairings dropped by the nav check | Contextual | Kept separate from mis-assignment waste |

`Mean_Assignment_Suitability` (column 9) is reported on TRUE attributes in
PO runs — under optimistic masking the masked view inflates scores, so
reporting it would make high masking look better. Masked-view values are
available separately as `Masked_View_Suitability` (the score the allocator
saw at assignment time; pairs are not re-scored after a reveal, so a task
revealed mid-run keeps its assignment-time masked score). Per-field
mis-assignment breakdowns (which masked fields drove each failure) go to
the per-run JSON in `<results_dir>/po_diagnostics/`, not this CSV.

## 12c. PO baseline + invariant diagnostics (3 columns)

| # | Column | What it measures | Direction | Notes |
|---|---|---|---|---|
| 57 | `Never_Attempted_Count` | Tasks spawned but never assigned to any robot, regardless of masking | **Lower** = better | Capacity baseline: `Masked_Starvation_Count` must be read against this (even at `Mask_Level=none` some tasks are never attempted when the fleet is saturated) |
| 58 | `All_Zero_Skip_Ticks` | Ticks where allocation was skipped because the unassigned × unassigned (masked) submatrix was all zero | Contextual | The no-bounce guard: no random fallback, no zero-suitability pairs executed; infeasible-for-all tasks starve instead |
| 59 | `Stuck_Assigned_Robot_Ticks` | Ticks (after a SUCCESSFUL plan call) where some robot had an assigned task but an EMPTY `current_path` (can neither move nor work) | **Must be 0** | Invariant per spec §1c. Ticks following planner FAILURES are excluded: the documented accept-stale-paths behavior legitimately leaves newly-assigned robots pathless until the next successful plan (verified: no stuck robot survives a successful plan call) |

## 12c-2. Collision handling diagnostics (v7, 4 columns)

Emitted by the two-phase timestep (`timestep.py`); counts are per-run totals.
Column numbers below are the v7 CSV positions; sections 12d/13 shift accordingly
(the doc pre-dates the collision block, which was added in v6 without a
renumbering pass).

| # | Column | What it measures | Direction | Notes |
|---|---|---|---|---|
| 60 | `Planned_Conflict_Count` | Vertex + swap conflicts in the tick's intended next-position set (after transitive waiting) | **Must be ~0** | With jointly planned paths this is 0; a nonzero value marks a mixed-vintage plan bug |
| 61 | `Safety_Nudge_Count` | Safety-net fires: a parked robot sharing a cell was nudged to a free neighbor | **Must be 0** | Last resort; each fire also leaves `occupied_locations` stale |
| 62 | `Collision_Violation_Count` | End-of-tick invariant breaches (co-located robots or same-tick swaps) | **Must be 0** | A violation means the nets failed to prevent physical overlap |
| 63 | `Propagated_Wait_Count` | Forced waits from transitive waiting: robots whose planned next step was blocked by a staying robot (no usable path / mid-route unassign / newly arrived), so they held position | Contextual | The cost of the stale-path design: robots that wait instead of colliding. 0 in runs with no planner failures and no mid-route stops |

## 12d. Anytime pool diagnostics (2 columns)

| # | Column | What it measures | Direction | Notes |
|---|---|---|---|---|
| 60 | `Threshold_Fire_Rate` | Share of threshold-active voting calls where the anytime early termination cut the pool short of its full size | Contextual | -1 when the run made no voting allocations (e.g. Murty's rows). Under the coverage threshold (2.8) a healthy value means the anytime path is exploring at least `MIN_POOL_BEFORE_THRESHOLD` candidates before stopping |
| 61 | `Mean_Pool_Size_At_Rank` | Mean ranked-pool size across ALL voting calls in the run | **Higher** = more exploration | -1 when no voting calls. Pools of ~1 indicate degenerate anytime behavior (pre-2.8 bug where the first candidate always cleared the mean-per-pair threshold) |

## 13. Config covariates (8 columns)

Constant per run — the raw configuration that produced this row.

| # | Column | What it measures | Notes |
|---|---|---|---|
| 63 | `Scenario` | Regime name (`calm` / `task_churn` / `fleet_churn` / `overload`) | For readable tables/plots |
| 64 | `Task_Arrival_Rate` | Per-timestep probability of a task-arrival check | Self-describing for reanalysis |
| 65 | `Task_Cancellation_Rate` | Per-timestep probability of a task-cancellation check | |
| 66 | `Robot_Arrival_Rate` | Per-timestep probability of a robot-arrival check | |
| 67 | `Robot_Departure_Rate` | Per-timestep probability of a robot-departure check | |
| 68 | `Mask_Level` | `none` / `low` / `high` | Empty string when PO inactive |
| 69 | `Mask_Count` | Fields masked per task (0 / 2 / 5) | -1 when PO inactive |
| 70 | `Mask_Policy` | `optimistic` / `pessimistic` / `none` (when level is none) | Empty string when PO inactive |

Mask conditions combine into 5 groups: `none`, `low-opt`, `low-pess`,
`high-opt`, `high-pess`. Masks are a pure function of
`(schedule_seed, task_id, mask_count)`; the condition is NOT in the schedule
seed, so all conditions replay identical schedules and a task's `low` mask
is a subset of its `high` mask.

## 14. Map size (1 column)

| # | Column | What it measures | Notes |
|---|---|---|---|
| 71 | `Map Size` | Small / Medium / Large based on grid hypotenuse | Small < 55, Medium < 105, Large ≥ 105 |

---

## Quick-reference: which columns to compare for each claim

| Claim | Compare these columns | Group by |
|---|---|---|
| **1. Voting ≈ optimization on quality** | `Total Score`, `Cumulative_Reassignment_Quality`, `Mean_Assignment_Suitability` | Method (within same scenario/planner) |
| **2. Voting is more fair** | `Init_Jains_Index`, `Avg Reassignment Jains Index`, `Avg_Reass_Gini`, `Workload_Variance` | Method |
| **3. Voting adapts faster** | `Adaptation_Latency_Mean`, `Adaptation_Latency_Max`, `Throughput_Var` | Method (hold Planner fixed) |
| **4. Voting scales better** | `Planning_Time_us`, `total_reassignment_time`, `Execution Time` | Method × `Num Robots` |
| **5. Planner choice matters** | `Task_Completion_Fraction`, `Throughput_Mean`, `Adaptation_Latency_Mean` | Planner (within same method/scenario) |
| **6. Regimes expose conditional advantages** | `Task_Completion_Fraction`, `Workload_Variance`, `Throughput_Var` | Scenario × Method |

---

## Removed columns (with recovery hints)

| Removed | Recovery |
|---|---|
| `Population_Coverage_Score` | `Total Score / (Num Robots × Num Tasks)` |
| `total_reward` | Only meaningful for economic analysis of reward accumulation |
| `total_success` | `Task_Completion_Fraction × Tasks_Spawned` |
| `Average Reassignment Score` | `Cumulative_Reassignment_Quality / total_reassignments` |
| `Average Path Length` | Only meaningful for map-difficulty comparisons |
| `Initial Jains Index (sim)` | Duplicate of `Init_Jains_Index` |
| `All_Tasks_Completed` | Equivalent to `Completion_Step != -1` |
| `Workload_Mean` | `total_success / Num Robots` |
| `Init_Below_GE_Frac`, etc. | Threshold/deficit metrics from `calculate_threshold_metrics` |
| `Init_Score_Range`, `Init_Min_Max_Ratio`, `Init_CV` | Alternative spread measures; IQR + Gini cover this |
| `Init_Mean` | Same as `Mean_Assignment_Suitability` |
| `Init_Med_Mean_Gap` | `|Init_Median - Init_Mean|` |
| `Avg_Reass_*` counterparts | Same treatment as the Init_ block |
| `CPU Usage`, `Memory Usage` | psutil snapshots, not portable across machines |
