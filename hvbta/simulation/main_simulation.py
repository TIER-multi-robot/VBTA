import time
import random
import glob
import re
import zlib
import numpy as np
import csv
import os
import json
import copy
import datetime
from collections import deque
from typing import List

from hvbta.simulation.timestep import simulate_time_step
import hvbta.allocators.voting as V
import hvbta.allocators.optimizers as O
import hvbta.suitability as S
import hvbta.generation as G
from hvbta.metrics import (
    calculate_jains_index,
    calculate_inequality_metrics,
    calculate_robustness_metrics,
)
from hvbta.pathfinding.CBS_helper import load_map, create_obstacle_list
from hvbta.pathfinding.MAPF_wrapper import run_planner
import hvbta.pathfinding.MAPF_wrapper as MW
from hvbta.pathfinding import prioritized_planner as _PP
from hvbta.allocators.misc_assignment import (
    add_new_tasks, add_new_robots, remove_random_robots, remove_random_tasks,
    remove_task_by_id, remove_robot_by_id, unassign_task_from_robot,
)
from hvbta.models import CapabilityProfile, TaskDescription
from hvbta.partial_observability import PartialObservabilityState


# ---------------------------------------------------------------------------
# Named regimes (TODO Part 2 Phase 1). Replaces the four independent rate knobs
# with a single `scenario` selector so the sweep is tractable and tables are
# readable. Every method + planner in a (map, robots, tasks, nc, sm, rep,
# scenario) group replays the SAME event schedule; the scenario sets the four
# rates that build that schedule.
#
# These are HYPOTHESES tuned to a 10-robot / 10-task baseline. The overload
# rate must exceed the measured completion capacity or it isn't overload; the
# calm rate must be low enough to sit near-static. Adjust after Phase 0 data.
# ---------------------------------------------------------------------------
REGIMES = {
    "calm":       dict(task_arrival_rate=0.10, task_cancellation_rate=0.01,
                       robot_arrival_rate=0.01, robot_departure_rate=0.01),
    "task_churn": dict(task_arrival_rate=0.15, task_cancellation_rate=0.08,
                       robot_arrival_rate=0.01, robot_departure_rate=0.01),
    "fleet_churn":dict(task_arrival_rate=0.05, task_cancellation_rate=0.02,
                       robot_arrival_rate=0.15, robot_departure_rate=0.15),
    "overload":   dict(task_arrival_rate=0.60, task_cancellation_rate=0.02,
                       robot_arrival_rate=0.03, robot_departure_rate=0.03),
    }


def compute_all_fairness_metrics(scores):
    """
    Trimmed fairness bundle - four canonical numbers per snapshot:
      Jains_Index  - overall equity (1 = perfectly equal)
      Gini         - canonical inequality (0 = equal, 1 = maximally unequal)
      Median       - robust central tendency
      IQR          - robust spread
    The dropped fields (thresholds, deficits, range, min/max ratio, CV, mean,
    med-mean gap) are all derivable from these plus the per-run raw scores if a
    downstream analysis ever needs them.
    """
    jains = calculate_jains_index(scores)
    inequality = calculate_inequality_metrics(scores)
    robustness = calculate_robustness_metrics(scores)
    return (
        jains,
        inequality["gini"],
        robustness["median"],
        robustness["iqr"],
    )


FAIRNESS_METRIC_NAMES = ["Jains_Index", "Gini", "Median", "IQR"]


def func_name(f):
    return getattr(f, "__name__", str(f))


# Per-run wall-clock limit (seconds). A run that hits it stops early and is
# written as a row with Run_Timed_Out = 1 (metrics computed over the ticks
# executed), never dropped. Configurable.
RUN_WALL_LIMIT_S = 600

# Deterministic planner-work bounds (2026-10-26, hyper-PP pilot fix): count-
# based caps that bound per-run planner work independently of machine load,
# applied identically to every allocation method (they live in the planner /
# replan layer, not the allocator). The 600 s wall cap above stays as a
# SAFETY STOP only - with these bounds no run should ever reach it.
#   PLANNER_CALLS_PER_TICK   - max run_planner invocations per tick (the
#       replan call + replan-on-failure alternative retries).
#   PLANNER_FAILED_CALL_BUDGET - max FAILED planner calls per run; once
#       exhausted the run goes planner-dead (further planner calls are
#       skipped, stale paths accepted) instead of burning the wall clock.
#       Normal runs fail 0-2 calls; the hyper-PP pilot's 600 s-capped runs
#       failed 100-1000+.
# Every cap firing is counted per run in the Planner_Cap_Hits column
# (together with PP's per-call expansion-budget hits).
PLANNER_CALLS_PER_TICK = 4
PLANNER_FAILED_CALL_BUDGET = 150


def _derived_seed(*parts) -> int:
    """
    Deterministic seed derivation from arbitrary key parts (agent
    instructions: use zlib.adler32 everywhere, never hash()). Returns an
    unsigned 32-bit seed.
    """
    return zlib.adler32(repr(tuple(parts)).encode()) & 0xFFFFFFFF


def _seed_rngs(seed: int):
    """
    Seed BOTH RNGs the simulator uses: Python's module-level `random` and
    numpy's global `np.random`. Called at every isolation scope (per-rep
    entity generation, per-method initial assignment, per-method run) so
    results are reproducible across processes and one method's random
    draws can never change another method's results.
    """
    random.seed(seed)
    np.random.seed(seed)


def suitability_all_zero(suitability_matrix):
    return all(value == 0 for row in suitability_matrix for value in row)


# ---------------------------------------------------------------------------
# Phase 2: Event-exposure-based horizon. Walks the pre-materialized schedule
# accumulating scheduled events; returns the step at which the running total
# first reaches K. This makes cross-regime comparison fair: every regime sees
# a comparable NUMBER OF EVENTS, not a comparable number of steps.
# ---------------------------------------------------------------------------
def horizon_for_event_count(schedule, K):
    """Return the step index at which cumulative scheduled events reach K.

    Events counted per entry: len(add_tasks) + len(add_robots) +
    len(remove_robot_ids) + len(remove_task_ids). Returns len(schedule) if K
    is never reached (the schedule is long enough that all events fire).
    """
    total = 0
    for i, entry in enumerate(schedule):
        total += (
            len(entry["add_tasks"])
            + len(entry["add_robots"])
            + len(entry["remove_robot_ids"])
            + len(entry["remove_task_ids"])
        )
        if total >= K:
            return i + 1
    return len(schedule)


def state_check(robots: List[CapabilityProfile], unassigned_robots: List[str], unassigned_tasks: List[str]):
    """State signature for deciding whether to re-plan."""
    active, goals = [], []
    for r in robots:
        if r.assigned and r.current_task:
            active.append(r.robot_id)
            goals.append((r.robot_id, tuple(r.current_task.location), r.current_task.task_id))
    return (
        tuple(sorted(active)),
        tuple(sorted(goals)),
        tuple(sorted(unassigned_robots)),
        tuple(sorted(unassigned_tasks)),
    )


def _with_seeded_random(seed, fn, *args, **kwargs):
    """
    Call `fn` with the module-level `random` RNG deterministically seeded, then
    restore the previous RNG state. Used by build_event_schedule so generation
    functions (which use module `random`) produce identical entities across
    every method's replay.
    """
    saved = random.getstate()
    random.seed(seed)
    try:
        return fn(*args, **kwargs)
    finally:
        random.setstate(saved)


def build_event_schedule(
        initial_robots: List[CapabilityProfile],
        initial_tasks: List[TaskDescription],
        grid: List[List[int]],
        max_time_steps: int,
        *,
        add_tasks: bool = False,
        add_robots: bool = False,
        remove_robots_flag: bool = False,
        remove_tasks_flag: bool = False,
        task_arrival_rate: float = 0.0,
        robot_arrival_rate: float = 0.0,
        robot_departure_rate: float = 0.0,
        task_cancellation_rate: float = 0.0,
        tasks_to_add: int = 1,
        robots_to_add: int = 1,
        robots_to_remove: int = 1,
        tasks_to_remove: int = 1,
        robot_generation_strict: bool = True,
        task_generation_strict: bool = True,
        seed: int = 0) -> list:
    """
    Pre-materialize a deterministic per-timestep event script that every method
    in a (rep, sm) group replays identically.

    Each entry is a dict with pre-built entity objects for additions and IDs
    for removals:
        {
            "add_tasks":       [TaskDescription, ...],
            "add_robots":      [CapabilityProfile, ...],
            "remove_task_ids":  [str, ...],
            "remove_robot_ids": [str, ...],
        }

    Fairness properties:
      - Two methods pointed at the same schedule see identical events at
        identical timesteps with identical entity capabilities/task_types.
      - A method that terminates early (all tasks completed, or hits max_time_steps)
        simply consumes a prefix of the schedule.
      - Removal targets are picked from a shadow "alive" list at schedule-build
        time. If a scheduled removal references an entity that method X has already
        completed / lost, replay no-ops for that method - the method just got there
        faster.

    Locations for added entities are picked at build time with an empty occupied
    set. At replay time a fresh copy of each entity is inserted; minor location
    conflicts are tolerated because runtime add_new_* was already permissive.
    """
    ev_rng = random.Random(seed)
    shadow_robot_ids = [r.robot_id for r in initial_robots]
    shadow_task_ids = [t.task_id for t in initial_tasks]
    # Track occupied locations for dynamically added tasks/robots so they
    # don't overlap with each other. Without this, two tasks can spawn at
    # the same cell, giving two robots the same goal — an unsolvable MAPF
    # instance that causes the planner to time out.
    occupied_task_locs = set(t.location for t in initial_tasks)
    occupied_robot_locs = set(r.location for r in initial_robots)
    next_robot_id = len(initial_robots) + 1
    next_task_id = len(initial_tasks) + 1

    schedule = []
    for _t in range(max_time_steps):
        entry = {
            "add_tasks": [],
            "add_robots": [],
            "remove_task_ids": [],
            "remove_robot_ids": [],
        }

        # task arrivals
        if add_tasks and ev_rng.random() < task_arrival_rate:
            n = ev_rng.randint(1, max(1, tasks_to_add))
            for _ in range(n):
                tid = f"T{next_task_id}"
                next_task_id += 1
                gen_seed = ev_rng.randint(0, 2**32 - 1)
                # Strict-only generation; the *_generation_strict flags are vestigial.
                # Pass occupied_task_locs so new tasks don't overlap existing ones.
                new_t = _with_seeded_random(
                    gen_seed,
                    G.generate_random_task_description_strict, tid, grid, occupied_task_locs, [],
                )
                occupied_task_locs.add(new_t.location)
                entry["add_tasks"].append(new_t)
                shadow_task_ids.append(tid)

        # robot arrivals
        if add_robots and ev_rng.random() < robot_arrival_rate:
            n = ev_rng.randint(1, max(1, robots_to_add))
            for _ in range(n):
                rid = f"R{next_robot_id}"
                next_robot_id += 1
                gen_seed = ev_rng.randint(0, 2**32 - 1)
                new_r = _with_seeded_random(
                    gen_seed,
                    G.generate_random_robot_profile_strict, rid, grid, occupied_robot_locs,
                )
                occupied_robot_locs.add(new_r.location)
                entry["add_robots"].append(new_r)
                shadow_robot_ids.append(rid)

        # robot departures - pick from currently-alive shadow list
        if remove_robots_flag and ev_rng.random() < robot_departure_rate:
            n = ev_rng.randint(1, max(1, robots_to_remove))
            if len(shadow_robot_ids) > 1:
                n = min(n, len(shadow_robot_ids) - 1)  # never drain to zero
                victims = ev_rng.sample(shadow_robot_ids, n)
                entry["remove_robot_ids"].extend(victims)
                for v in victims:
                    shadow_robot_ids.remove(v)

        # task cancellations
        if remove_tasks_flag and ev_rng.random() < task_cancellation_rate:
            n = ev_rng.randint(1, max(1, tasks_to_remove))
            if shadow_task_ids:
                n = min(n, len(shadow_task_ids))
                victims = ev_rng.sample(shadow_task_ids, n)
                entry["remove_task_ids"].extend(victims)
                for v in victims:
                    shadow_task_ids.remove(v)

        schedule.append(entry)
    return schedule


def reassign_robots_to_tasks_direct(
        robots: List[CapabilityProfile],
        tasks: List[TaskDescription],
        bypass_fn,
        unassigned_robots: List[str],
        unassigned_tasks: List[str],
        start_positions: dict,
        goal_positions: dict,
        llm_cache: dict = None):
    """
    Replan via a direct-assignment LLM (e.g. bypass_suitability_from_names_with_llm).
    Mirrors the return shape of V.reassign_robots_to_tasks /
    O.reassign_robots_to_tasks_with_method so the main loop can dispatch uniformly:
        return_assignments, unassigned_robots, unassigned_tasks, score, length, per_agent_scores
    The bypass operates only on the unassigned subset, so it never steals tasks from
    busy robots. If a matrix cache is supplied, per-pair scores are read out of it for
    the fairness metrics; otherwise per-pair scores default to 0.5 sentinel.
    """
    urobots = [r for r in robots if not r.assigned]
    utasks = [t for t in tasks if not t.assigned]
    if not urobots or not utasks:
        return {}, unassigned_robots, unassigned_tasks, 0.0, 0.0, []

    t0 = time.perf_counter_ns()
    (sub_pairs, _unr_sub, _unt_sub), parse_failed = bypass_fn(urobots, utasks)
    length = (time.perf_counter_ns() - t0) / 1000.0

    if parse_failed:
        print("[direct LLM replan] parse failed - leaving unassigned this tick")
        return {}, unassigned_robots, unassigned_tasks, 0.0, length, []

    cache_matrix = llm_cache.get("matrix") if llm_cache else None
    r_to_idx = llm_cache.get("robot_id_to_idx") if llm_cache else None
    t_to_idx = llm_cache.get("task_id_to_idx") if llm_cache else None

    return_assignments = {}
    score = 0.0
    for ri, ti in sub_pairs:
        r = urobots[ri]
        t = utasks[ti]
        pair_score = 0.5
        if (cache_matrix is not None and r_to_idx and t_to_idx
                and r.robot_id in r_to_idx and t.task_id in t_to_idx):
            pair_score = float(cache_matrix[r_to_idx[r.robot_id], t_to_idx[t.task_id]])

        r.current_task = t
        r.tasks_attempted += 1
        r.assigned = True
        t.assigned_robot = r
        t.assigned = True
        r.current_task_suitability = pair_score
        t.current_suitability = pair_score
        start_positions[r.robot_id] = r.location
        goal_positions[r.robot_id] = t.location
        return_assignments[r.robot_id] = t.task_id
        score += pair_score

    new_unassigned_robots = [r.robot_id for r in robots if not r.assigned]
    new_unassigned_tasks = [t.task_id for t in tasks if not t.assigned]
    per_agent_scores = [
        float(r.current_task_suitability)
        for r in robots
        if r.assigned and r.current_task is not None and r.current_task_suitability is not None
    ]
    return return_assignments, new_unassigned_robots, new_unassigned_tasks, score, length, per_agent_scores


def main_simulation(
        output: tuple,
        robots: List[CapabilityProfile],
        tasks: List[TaskDescription],
        num_candidates: int,
        voting_method: callable,
        grid: List[List[int]],
        map_dict: dict,
        suitability_method: callable,
        suitability_matrix: np.ndarray,
        max_time_steps: int,
        add_tasks: bool,
        add_robots: bool,
        remove_robots: bool,
        tasks_to_add: int = 1,
        robots_to_add: int = 1,
        robots_to_remove: int = 1,
        robot_generation_strict: bool = True,
        task_generation_strict: bool = True,
        initial_jains_index: float = 0.0,
        # Lifelong event rates: per-timestep probability of an arrival/removal check firing.
        # Old <=2 / <=4 hard-caps are gone; set 0.0 to disable a channel.
        task_arrival_rate: float = 0.05,
        robot_arrival_rate: float = 0.02,
        robot_departure_rate: float = 0.02,
        task_cancellation_rate: float = 0.02,
        remove_tasks: bool = False,
        tasks_to_remove: int = 1,
        # When provided, dynamic events are replayed from this schedule instead of
        # drawn from the random-rate gates. Enables fair cross-method comparison
        # within a (rep, sm) group.
        event_schedule: list = None,
        # Partial-observability experiment (see partial_observability_experiment.md).
        # mask_level: "none" / "low" / "high" - activates the masked-view allocator
        #   + execution-time feasibility check. None (default) disables the
        #   machinery entirely (legacy behavior; PO CSV columns get sentinels).
        # mask_policy: "optimistic" / "pessimistic" (only meaningful when
        #   mask_level is "low"/"high").
        # schedule_seed: the group's schedule seed (zlib.adler32 of the group
        #   key). Masks are derived from it + task_id, so it must be passed for
        #   masks to be reproducible.
        mask_level: str = None,
        mask_policy: str = None,
        # 8.1 score-noise experiment: Gaussian observation noise (sigma_obs)
        # on FEASIBLE entries only, deterministic per (schedule_seed,
        # robot_id, task_id, tick), shared across methods. None = off.
        # Execution always uses true scores; only the allocator's
        # suitability_source sees the noisy view.
        noise_sigma: float = None,
        schedule_seed: int = None,
        # Per-run RNG seed keyed by (group, method): derived by the sweep via
        # _derived_seed(schedule_seed, "run", method). Deliberately does NOT
        # include mask_level/mask_policy (or any future observation-noise
        # sigma): conditions that share robots/tasks/schedule must share the
        # run seed, so e.g. sigma_obs = 0 reproduces the no-noise results
        # exactly. Seeding both `random` and `np.random` with it at entry
        # makes each method's run reproducible regardless of execution order
        # or what any other method consumed before it.
        run_seed: int = None,
        # Optional path for the per-run PO diagnostic JSON (masks, per-field
        # mis-assignment breakdown). Only written when mask_level is set.
        po_diag_path: str = None,
        # MAPF planner selector ("cbs" | "pp" | "pibt" | "lacam"). Routes through
        # MAPF_wrapper.run_planner so the outer loop is untouched. Default "cbs"
        # during the refactor so partial progress never breaks a run.
        planner: str = "lacam",
        # Phase 4 CSV-provenance fields, passed through from the sweep so the
        # row is self-describing. These are constant per run, not computed
        # inside the sim — they're just threaded to the return tuple.
        scenario: str = "",
        scheduled_event_count: int = 0,
        horizon_binding_term: str = ""):
    print(f"SUITABILITY METHOD: {suitability_method}")

    # Per-run RNG isolation (agent instructions Step 1.2): seed both RNGs so
    # this method's draws depend only on its own (group, method) key.
    if run_seed is not None:
        _seed_rngs(run_seed)

    voting_methods = {
        V.rank_assignments_borda: "Borda Count",
        V.rank_assignments_approval: "Approval Voting",
        V.rank_assignments_majority_judgment: "Majority Judgment",
        V.rank_assignments_cumulative_voting: "Cumulative Voting",
        V.rank_assignments_condorcet_method: "Condorcet Method",
        V.rank_assignments_range: "Range Voting",
    }
    optimization_methods = {
        O.cbba_task_allocation: "CBBA",
        O.ssia_task_allocation: "SSIA",
        O.ilp_task_allocation: "ILP",
        O.jv_task_allocation: "JV",
        O.murtys_task_allocation: "Murty's K-Best",
        O.perturbed_jv_task_allocation: "Perturbed JV",
    }
    voting_method_name = voting_methods.get(voting_method, "Unknown Method")
    optimization_method_name = optimization_methods.get(voting_method, "Unknown Method")

    HYPOTENUSE = (len(grid) ** 2 + len(grid[0]) ** 2) ** 0.5
    total_reward = 0.0
    total_success = 0.0
    total_tasks = len(tasks)  # tasks_spawned - grows with dynamic additions
    # Set of every task_id that was ever assigned to a robot. Denominator for
    # attempted_completion_rate, which excludes late-arriving tasks that never
    # got a chance to be attempted before max_time_steps.
    tasks_ever_assigned = set()
    # Records the timestep at which len(tasks) first hit 0. None if never fully
    # cleared. Lets downstream analysis compute variable-length metrics
    # (throughput, adaptation latency) against a real completion horizon.
    all_tasks_completed_at_step = None

    # Primary comparative-metric trackers.
    # completed_per_step: one entry per elapsed timestep with # of task completions
    #     that step. Mean = throughput; variance = burstiness.
    # pending_event_steps: FIFO of timesteps at which a dynamic event fired without
    #     a subsequent successful PLANNER resolution yet. Drained when run_planner succeeds.
    # adaptation_latencies: timesteps elapsed between each drained event and the
    #     replan that resolved it.
    completed_per_step = []
    # Phase 0 diagnostics: per-timestep task-pool and fleet-size logs. Used to
    # diagnose the regime (steady state vs. overload) and to confirm the
    # fleet-size random walk isn't drifting when arrival == departure rate.
    # Surfaced via a single print at run end so a pilot run's stdout can be
    # scraped for plotting without shifting the metric return tuple (CSV
    # plumbing for these is deferred to Phase 4).
    task_pool_size_per_step = []
    fleet_size_per_step = []
    pending_event_steps = []
    adaptation_latencies = []
    # Urgency accounting. At each replan tick we record:
    #   percentage_urgent_per_replan: fraction of currently-unassigned tasks
    #     whose priority_level == "urgent". Cross-run comparable in [0, 1].
    #   robot_pressure_per_replan:    1 if the "add robots" event fired in the
    #     same tick as this replan, else 0. Cross-run comparable in [0, 1] when
    #     averaged.
    # Report mean + max of each at run end so runs can be bucketed by
    # ("none urgent, no pressure") ... ("all urgent, high pressure").
    percentage_urgent_per_replan = []
    robot_pressure_per_replan = []
    # Count of planner failures (ok=False) across the entire run. This
    # measures how often the MAPF planner couldn't find a plan, which
    # is the key metric for the planner-allocation interaction (replan-
    # on-failure tries alternatives before accepting stale paths).
    plan_failure_count = 0
    # Total count of alternative assignments tried during replan-on-failure
    # across the entire run. Each retry attempt (whether it succeeds or fails)
    # increments this. Used to report the ratio of alternatives tried to
    # failures, showing how many alternatives each method attempts before
    # recovering or accepting stale paths.
    alternatives_tried = 0
    # Invariant-check accumulator: ticks where some robot is assigned to a
    # task but has an empty current_path (can neither move nor work).
    stuck_assigned_robot_ticks = 0
    # Phase 3v2 collision accounting: safety-net fires (swap/nudge) and
    # end-of-tick invariant violations, counted by the timestep.
    collision_counters = {"swap": 0, "nudge": 0, "violations": 0, "planned_conflicts": 0,
                          "propagated_waits": 0, "mover_conflict_waits": 0,
                          "plan_deviation": 0}
    # Phase 3 cascade diagnostics: one entry per failed planner call,
    # {tick, kind, ...} (kind = duplicate_start / duplicate_goal /
    # lacam_no_solution / lacam_timeout / engine_unsolved). Goes into the
    # per-run diagnostic JSON.
    plan_failure_log = []
    # Planner-call wall-time accounting (reported per run in the diag JSON;
    # planning idle agents as stay-put agents costs time, so measure it).
    MW.reset_planner_timing()
    planner_timing = MW.PLANNER_TIMING
    # Deterministic planner-work bound state: failed-call budget + cap-hit
    # counter (see PLANNER_CALLS_PER_TICK / PLANNER_FAILED_CALL_BUDGET).
    # _pp_cap_hits_start snapshots PP's cumulative expansion-budget counter
    # so the per-run delta can be added to Planner_Cap_Hits.
    planner_failed_calls = 0
    planner_cap_hits = 0
    # Per-run count of failed planner calls that hit the wall-clock time
    # budget (kind == "lacam_timeout"; MAPF_wrapper classifies every
    # failure). Logging-only column (Planner_Timeout_Count, 2026-10-27):
    # distinguishes LaCAM budget hits from no-solution failures in the
    # CSV. Deterministic given the (load-sensitive) planner outcomes.
    lacam_timeout_count = 0
    # First tick at which planner_failed_calls reached
    # PLANNER_FAILED_CALL_BUDGET (planner-dead mode begins; reported as
    # the Planner_Dead_Tick column, -1 = never). Deterministic.
    planner_dead_tick = -1
    _pp_cap_hits_start = _PP.PP_EXPANSION_CAP_HITS
    # Ticks where allocation was skipped because the unassigned x unassigned
    # submatrix was all zero (2.7 guard; applies to every method and PO and
    # non-PO runs alike).
    all_zero_skip_ticks = 0
    # Pool diagnostics per run (2.8): V.POOL_STATS collects one dict per
    # assign_tasks_with_voting call; aggregated into the CSV columns
    # Threshold_Fire_Rate and Mean_Pool_Size_At_Rank at run end.
    V.POOL_STATS = []
    murtys_alternatives = []
    # Warm-start seed for the next voting replan. Persists across the loop as
    # a list of (robot_id, task_id) tuples (ID-based so it survives adds /
    # removes / completions). Updated after every successful voting replan.
    # None until the first voting round produces something.
    previous_best_id_pairs = None

    # Adaptive candidate count. Rolling window of per-timestep event counts over
    # the last 5 steps; recent churn shrinks the voting candidate pool so voting
    # responds fast under stress and explores broadly when calm.
    ADAPTIVE_WINDOW = 5
    MIN_CANDIDATES = 5
    recent_events_window = deque(maxlen=ADAPTIVE_WINDOW)

    # Hall of fame candidate cache. List of ID-pair-list candidates (each a list
    # of (robot_id, task_id)) carried across timesteps. After each voting round
    # its top-K survivors are merged in; at the next round they are translated to
    # local indices (dropping invalid pairs) and injected alongside fresh ones.
    HALL_OF_FAME_MAX = 20
    HALL_OF_FAME_K = 3
    candidate_cache = []

    # Full-schedule denominators. Constant across all methods in a (rep, sm) group
    # regardless of which method terminated early. Use these as denominators when
    # comparing across methods; use per-method total_tasks / len(robots) when you
    # want "what this method actually experienced".
    if event_schedule is not None:
        full_schedule_tasks_added = sum(len(e["add_tasks"]) for e in event_schedule)
        full_schedule_robots_added = sum(len(e["add_robots"]) for e in event_schedule)
    else:
        full_schedule_tasks_added = 0
        full_schedule_robots_added = 0
    full_schedule_tasks_spawned = total_tasks + full_schedule_tasks_added
    full_schedule_robots_max = len(robots) + full_schedule_robots_added
    total_reassignment_time = 0.0
    cumulative_reassignment_quality = 0.0
    total_reassignments = 0
    total_time_steps = max_time_steps
    reassignment_jains_indices = []
    reassignment_fairness_metrics = []
    reassign_score = 0.0
    reassign_length = 0.0

    # LLM suitability cache - reuse matrix unless NEW entities are added
    llm_cache = {
        "matrix": suitability_matrix if getattr(suitability_method, "_is_llm_batch", False) else None,
        "robot_id_to_idx": {r.robot_id: i for i, r in enumerate(robots)} if getattr(suitability_method, "_is_llm_batch", False) else None,
        "task_id_to_idx": {t.task_id: j for j, t in enumerate(tasks)} if getattr(suitability_method, "_is_llm_batch", False) else None,
    }

    occupied_positions = set(r.location for r in robots)
    # suitability_source is set inside the should_replan block; the
    # replan-on-failure block below can run on path-only replan ticks
    # (should_replan False), so pre-bind the plain scorer as the default.
    suitability_source = suitability_method
    # Frozen-cell obstacle set for the timestep's post-movement
    # reconciliation (Phase 3 cascade fix).
    _obstacle_set = frozenset(map_dict.get("obstacles") or [])

    # Partial-observability state. Active whenever mask_level is set
    # (including "none", which runs the execution-time feasibility check
    # with no masked attributes - the required control condition).
    po = None
    if mask_level is not None:
        po = PartialObservabilityState(schedule_seed, mask_level, mask_policy)

    # apply initial assignment
    assigned_pairs = output[0]
    for robot_idx, task_idx in assigned_pairs:
        r = robots[robot_idx]
        t = tasks[task_idx]
        if po is not None:
            # Record the masked-view pair score (allocator's own view) so
            # note_round's Masked_View_Suitability is correct for the initial
            # assignment too, then scrub zero-suitability pairs (PO never
            # executes a pairing the allocator itself scored 0 - spec §6.3).
            r.current_task_suitability = float(
                po.masked_scorer(suitability_method)([r], [t], map_size=HYPOTENUSE)[0, 0])
            if float(r.current_task_suitability) == 0.0:
                continue
        r.current_task = t
        r.assigned = True
        r.tasks_attempted = 1
        t.assigned_robot = r
        t.assigned = True
        tasks_ever_assigned.add(t.task_id)

    unassigned_tasks = [t.task_id for t in tasks if not t.assigned]
    unassigned_robots = [r.robot_id for r in robots if not r.assigned]
    start_positions = {r.robot_id: r.location for r in robots if r.assigned and r.current_task}
    goal_positions = {r.robot_id: r.current_task.location for r in robots if r.assigned and r.current_task}

    if po is not None:
        # The initial assignment counts as the first allocation round for the
        # true-vs-masked suitability metrics.
        po.note_round(robots, HYPOTENUSE)

    avg_path_length, _initial_planner_okay = run_planner(planner, robots, start_positions, goal_positions, map_dict)
    # The initial call's outcome was previously ignored entirely - a
    # failure left every robot pathless but invisible to
    # Plan_Failure_Count, contaminating "runs with no planner failure"
    # analyses. Count it like any other failed call.
    if not _initial_planner_okay:
        plan_failure_count += 1
        planner_failed_calls += 1
        if planner_dead_tick < 0 and planner_failed_calls >= PLANNER_FAILED_CALL_BUDGET:
            planner_dead_tick = 0
        _entry = {"tick": 0, "phase": "initial"}
        _entry.update(MW.LAST_PLAN_FAILURE or {"kind": "unknown"})
        if _entry.get("kind") == "lacam_timeout":
            lacam_timeout_count += 1
        plan_failure_log.append(_entry)

    # create the initial state
    previous_active, previous_goals, previous_unassigned_robots, previous_unassigned_tasks = state_check(robots, unassigned_robots, unassigned_tasks)
    events = {"new_tasks": 0, "new_robots": 0, "completed_tasks": 0}
    idle_steps = {r.robot_id: 0 for r in robots}
    robot_max_id = len(robots) + 1
    task_max_id = len(tasks) + 1

    run_wall_start = time.perf_counter()
    run_timed_out = False
    # Tick-order restructure (Phase 3v3): the timestep is the LAST thing in
    # each tick. Completions / battery deaths / reveals from tick N's
    # timestep are processed at the START of tick N+1, before any new
    # movement. No robot moves on a plan made stale by an event in the
    # same tick.
    completed_prev_timestep = 0
    # v8 replan-on-deviation: set by the timestep when any robot's
    # movement deviates from its plan (off-plan wait, mover-conflict
    # wait, or an assigned robot with no usable path). Forces a planner
    # call at the top of the next tick, before movement, so plans never
    # stay stale after a deviation.
    deviation_replan_pending = False
    # 8.1 noise experiment: the noisy scorer reads the current replan tick
    # from this holder so noise is a pure function of (schedule_seed,
    # robot_id, task_id, tick). Updated at the top of every replan tick.
    noise_tick_holder = [0]
    for time_step in range(max_time_steps):
        # Per-run wall-clock limit: stop early, flag the row, never drop it.
        if time.perf_counter() - run_wall_start > RUN_WALL_LIMIT_S:
            print(f"RUN WALL LIMIT ({RUN_WALL_LIMIT_S}s) hit at tick {time_step} - stopping early.")
            run_timed_out = True
            total_time_steps = time_step
            break
        # update unassigned robots and tasks
        unassigned_robots = [r.robot_id for r in robots if not r.assigned]
        unassigned_tasks = [t.task_id for t in tasks if not t.assigned]

        # Tick order (Phase 3v3): process completions from the PREVIOUS
        # tick's timestep BEFORE any new movement or event application.
        # No robot moves on a plan made stale by an event in the same tick.
        events["completed_tasks"] += completed_prev_timestep
        should_replan_paths = completed_prev_timestep > 0
        # Replan on deviation (v8): the previous tick's timestep flagged
        # that some robot ended off-plan; replan before any movement.
        if deviation_replan_pending:
            should_replan_paths = True
            deviation_replan_pending = False

        # Snapshot pool sizes so we can measure add/remove churn from the dynamic
        # block below. Combined with completions, this feeds the adaptive
        # candidate-count rolling window.
        n_tasks_before_events = len(tasks)
        n_robots_before_events = len(robots)

        # Dynamic events. When an event_schedule is supplied, replay it
        # deterministically so every method in a (rep, sm) group sees the same
        # arrivals/departures at the same timesteps. Otherwise fall back to
        # the legacy random-rate gates.
        if event_schedule is not None and time_step < len(event_schedule):
            entry = event_schedule[time_step]

            # scheduled task arrivals
            if entry["add_tasks"]:
                print(f"ADDING NEW TASKS AT TIME STEP {time_step + 1} (scheduled)")
                for new_t in entry["add_tasks"]:
                    t_copy = copy.deepcopy(new_t)
                    tasks.append(t_copy)
                    unassigned_tasks.append(t_copy.task_id)
                    total_tasks += 1
                    events["new_tasks"] += 1
                pending_event_steps.append(time_step)

            # scheduled robot arrivals
            if entry["add_robots"]:
                print(f"ADDING NEW ROBOTS AT TIME STEP {time_step + 1} (scheduled)")
                for new_r in entry["add_robots"]:
                    r_copy = copy.deepcopy(new_r)
                    # The schedule picked this location at build time (using
                    # the INITIAL robot positions). By replay time robots have
                    # moved, so the cell may be occupied. Nudge the arrival
                    # to the nearest free cell to avoid spawning on top of a
                    # standing robot (the duplicate-start trigger).
                    if r_copy.location in occupied_positions:
                        _dims = map_dict.get("dimension", (0, 0))
                        from hvbta.pathfinding.planner_common import _nearest_free_cell
                        _free = _nearest_free_cell(
                            r_copy.location, _obstacle_set, occupied_positions,
                            _dims[0], _dims[1])
                        if _free is not None:
                            print(f"  [arrival] {r_copy.robot_id} nudged from "
                                  f"{r_copy.location} (occupied) to {_free}")
                            r_copy.location = _free
                    robots.append(r_copy)
                    unassigned_robots.append(r_copy.robot_id)
                    occupied_positions.add(r_copy.location)
                    idle_steps.setdefault(r_copy.robot_id, 0)
                    events["new_robots"] += 1
                pending_event_steps.append(time_step)

            # scheduled robot departures (targeted, id-based)
            if entry["remove_robot_ids"]:
                departed_here = 0
                for rid in entry["remove_robot_ids"]:
                    if remove_robot_by_id(
                        robots, tasks, unassigned_robots, unassigned_tasks,
                        rid, occupied_positions, start_positions, goal_positions,
                    ):
                        idle_steps.pop(rid, None)
                        departed_here += 1
                if departed_here:
                    pending_event_steps.append(time_step)

            # scheduled task cancellations (targeted, id-based). Treat as a
            # completion-shaped event so the replan logic kicks in.
            if entry["remove_task_ids"]:
                cancelled_here = 0
                for tid in entry["remove_task_ids"]:
                    if remove_task_by_id(tasks, unassigned_tasks, tid, robots, unassigned_robots):
                        cancelled_here += 1
                events["completed_tasks"] += cancelled_here
                if cancelled_here:
                    pending_event_steps.append(time_step)
        else:
            # legacy random-rate path (unchanged from prior implementation) NOT USED, MUST USE SCHEDULE FOR FAIR COMPARISON
            if add_tasks and random.random() < task_arrival_rate:
                print(f"ADDING NEW TASKS AT TIME STEP {time_step + 1}")
                n = random.randint(1, max(1, tasks_to_add))
                task_max_id, total_tasks = add_new_tasks(
                    tasks, unassigned_tasks, task_max_id, n, total_tasks,
                    grid, occupied_positions, task_generation_strict,
                )
                events["new_tasks"] += n
                pending_event_steps.append(time_step)

            if add_robots and random.random() < robot_arrival_rate:
                print(f"ADDING NEW ROBOTS AT TIME STEP {time_step + 1}")
                n = random.randint(1, max(1, robots_to_add))
                robot_max_id = add_new_robots(
                    robots, unassigned_robots, robot_max_id, n,
                    grid, occupied_positions, robot_generation_strict,
                )
                events["new_robots"] += n
                for r in robots:
                    idle_steps.setdefault(r.robot_id, 0)
                pending_event_steps.append(time_step)

            if remove_robots and random.random() < robot_departure_rate:
                if len(robots) > 1:
                    print(f"REMOVING RANDOM ROBOTS AT TIME STEP {time_step + 1}")
                    removed = remove_random_robots(
                        robots, tasks, unassigned_robots, unassigned_tasks,
                        random.randint(1, max(1, robots_to_remove)),
                        occupied_positions, start_positions, goal_positions,
                    )
                    for removed_robot in removed:
                        idle_steps.pop(removed_robot.robot_id, None)
                    if removed:
                        pending_event_steps.append(time_step)

            if remove_tasks and random.random() < task_cancellation_rate:
                if len(tasks) > 0:
                    print(f"CANCELLING RANDOM TASKS AT TIME STEP {time_step + 1}")
                    removed_t = remove_random_tasks(
                        tasks, unassigned_tasks,
                        random.randint(1, max(1, tasks_to_remove)),
                        robots, unassigned_robots,
                    )
                    events["completed_tasks"] += len(removed_t)
                    if removed_t:
                        pending_event_steps.append(time_step)

        # Record this step's churn for the adaptive candidate window. Sum of
        # completions (from the PREVIOUS tick's timestep) + |task pool delta|
        # + |robot pool delta| approximates total activity this step (a
        # heuristic; exactness is not required).
        events_this_step = (
            completed_prev_timestep
            + abs(len(tasks) - n_tasks_before_events)
            + abs(len(robots) - n_robots_before_events)
        )
        recent_events_window.append(events_this_step)

        # increment or reset idle steps for idle robots
        for r in robots:
            if not r.assigned:
                idle_steps[r.robot_id] = idle_steps.get(r.robot_id, 0) + 1
            else:
                idle_steps[r.robot_id] = 0

        # refresh start/goal
        for robot in robots:
            if robot.assigned and robot.current_task:
                start_positions[robot.robot_id] = robot.location
                goal_positions[robot.robot_id] = robot.current_task.location
            else:
                start_positions.pop(robot.robot_id, None)
                goal_positions.pop(robot.robot_id, None)

        # update state
        current_active, current_goals, current_unassigned_robots, current_unassigned_tasks = state_check(robots, unassigned_robots, unassigned_tasks)

        # check if there are any new events
        should_replan = False
        if unassigned_robots and unassigned_tasks:
            if events["new_tasks"] or events["new_robots"] or events["completed_tasks"]:
                should_replan = True
            # check if there is a state change
            elif (
                current_active != previous_active
                or current_goals != previous_goals
                or current_unassigned_robots != previous_unassigned_robots
                or current_unassigned_tasks != previous_unassigned_tasks
            ):
                should_replan = True
        if should_replan:
            should_replan_paths = True

        # top_k_out_ids is only populated on voting allocation rounds; the
        # replan-on-failure block below reads it on ANY tick (including
        # path-only replan ticks where should_replan is False and no
        # allocation ran), so it must ALWAYS exist.
        top_k_out_ids = []

        if should_replan:
            # 8.1: noise for this allocation round is keyed to this tick.
            noise_tick_holder[0] = time_step
            # ---- urgency + pressure snapshot for this replan ----
            # Compute once so voting/optimization/direct-LLM all see the same
            # numbers and downstream analysis can bucket runs by these dimensions.
            unassigned_task_objs = [t for t in tasks if not t.assigned]
            num_unassigned = len(unassigned_task_objs)
            num_urgent = sum(
                1 for t in unassigned_task_objs
                if getattr(t, "priority_level", "medium") == "urgent"
            )
            percentage_urgent = (num_urgent / num_unassigned) if num_unassigned else 0.0
            robot_pressure = 1.0 if events["new_robots"] > 0 else 0.0
            percentage_urgent_per_replan.append(percentage_urgent)
            robot_pressure_per_replan.append(robot_pressure)

            # determine what to pass to the reassignment functions
            if getattr(suitability_method, "_is_llm_batch", False):
                current_robot_ids = {r.robot_id for r in robots}
                current_task_ids = {t.task_id for t in tasks}
                # check the LLM cache is valid
                cache_valid = (
                    llm_cache["matrix"] is not None
                    and llm_cache["robot_id_to_idx"] is not None
                    and llm_cache["task_id_to_idx"] is not None
                    and current_robot_ids.issubset(llm_cache["robot_id_to_idx"].keys())
                    and current_task_ids.issubset(llm_cache["task_id_to_idx"].keys())
                )
                # update it if not
                if not cache_valid:
                    print(f"LLM cache miss - rebuilding matrix for {len(robots)} robots, {len(tasks)} tasks")
                    result = suitability_method(robots, tasks)
                    # Unwrap (matrix, parse_failed) tuples so the cache stores a bare ndarray.
                    suitability_matrix = result[0] if isinstance(result, tuple) else result
                    llm_cache["matrix"] = suitability_matrix
                    llm_cache["robot_id_to_idx"] = {r.robot_id: i for i, r in enumerate(robots)}
                    llm_cache["task_id_to_idx"] = {t.task_id: j for j, t in enumerate(tasks)}
                # use the updated cache as suitability source if LLM is suitibility method
                suitability_source = (llm_cache["matrix"], llm_cache["robot_id_to_idx"], llm_cache["task_id_to_idx"])
            else:
                # PO: wrap the scorer so the allocator only ever scores masked
                # views. Every allocator decision (candidate generation,
                # voting, Murty's, stealing, warm-start) flows through this
                # call, so true values of masked fields never reach them.
                if po is not None:
                    suitability_source = po.masked_scorer(suitability_method)
                else:
                    suitability_source = suitability_method
                # 8.1 noise experiment: wrap the (possibly masked) scorer
                # with observation noise. sigma=0 rows pass through
                # unchanged, reproducing the no-noise results exactly.
                if noise_sigma is not None:
                    from hvbta.partial_observability import noisy_scorer
                    suitability_source = noisy_scorer(
                        suitability_source, noise_sigma, schedule_seed,
                        noise_tick_holder)

            # All-zero guard (2.7): when the unassigned x unassigned submatrix
            # is all zero, skip the allocation ENTIRELY - for voting AND
            # Murty's, in ALL runs (PO and non-PO). The allocators' legacy
            # random fallbacks would otherwise assign infeasible pairs (zero
            # hard-fail-checked suitability) that mis-fire at arrival and
            # bounce forever; infeasible-for-all tasks starve instead.
            po_all_zero = False
            if callable(suitability_source):
                _ur = [r for r in robots if not r.assigned]
                _ut = [t for t in tasks if not t.assigned]
                if _ur and _ut:
                    _m = np.asarray(suitability_source(_ur, _ut, map_size=HYPOTENUSE))
                    po_all_zero = not np.any(_m > 0)
                if po_all_zero:
                    # Counted per tick in the All_Zero_Skip_Ticks column (no
                    # per-tick print - too noisy at sweep scale).
                    all_zero_skip_ticks += 1
                    if po is not None:
                        po.all_zero_skip_ticks += 1  # mirrored in the diag JSON

            # top_k_out_ids is only populated on voting rounds; the
            # replan-on-failure block reads it on ANY tick, so it must
            # exist even when allocation was skipped this tick (all-zero
            # guard) or the method is an optimizer.
            top_k_out_ids = []

            # assign with voting
            if not po_all_zero and voting_method in voting_methods:
                # Anytime pressure: any urgent unassigned task or a new-robot
                # event this tick triggers a tight budget so voting bails on
                # the first good-enough candidate.
                urgent_pressure = (percentage_urgent > 0.0) or (robot_pressure > 0.0) # NOTE: here I can take out robot pressure
                # 4a (2026-10-26): the urgent candidate cap is OFF in the
                # main configuration - HVBTA_URGENT_CAP=1 re-enables it for
                # the cap ablation (VOTING_URGENT_CANDIDATE_BUDGET=3).
                _urgent_cap_arm = os.environ.get("HVBTA_URGENT_CAP", "0") == "1"
                voting_candidate_budget = (V.VOTING_URGENT_CANDIDATE_BUDGET
                                          if (urgent_pressure and _urgent_cap_arm) else None)
                # Anytime stopping rule (2026-10-27; final scope): under
                # urgency, voting ranks candidates in pool order and stops
                # once the best-so-far candidate (by coverage-normalized
                # quality) has not improved for k =
                # V.VOTING_URGENT_STALL_K = 5 consecutive candidates - but
                # only after at least MIN_POOL_BEFORE_THRESHOLD = 5
                # candidates - or when the pool is exhausted. Replaces the
                # f*UB threshold (tiered f rejected: too many fitted
                # constants with thin support in large calls; the UB code
                # stays in place but is unused by the main config).
                # HVBTA_STALL=0 disables the rule (the anytime ablation
                # arm: full pool on urgent calls). Non-urgent calls always
                # use the full pool.
                _stall_enabled = os.environ.get("HVBTA_STALL", "1") == "1"
                voting_stall_k = (V.VOTING_URGENT_STALL_K
                                  if (urgent_pressure and _stall_enabled) else None)
                voting_score_threshold = None

                # Adaptive candidate count: recent churn shrinks the pool so
                # voting responds fast under stress and explores broadly when
                # calm. Floors at MIN_CANDIDATES.
                recent_event_count = sum(recent_events_window)
                effective_candidates = max(MIN_CANDIDATES, num_candidates // (1 + recent_event_count))

                # Hall of fame: hand the surviving cached candidates to the pool
                # and collect this round's top-K back out.
                top_k_out_ids = []

                print(f"REASSIGNING WITH VOTING METHOD: {voting_method_name}"
                      f"{f' (URGENT: pct_urgent={percentage_urgent:.0%}, robot_pressure={int(robot_pressure)}, stall_k={voting_stall_k})' if urgent_pressure else ''}"
                      f"{f' [warm-start seed: {len(previous_best_id_pairs)} pairs]' if previous_best_id_pairs else ''}"
                      f" [candidates: {effective_candidates}/{num_candidates}, recent_events={recent_event_count}]"
                      f"{f' [hall-of-fame: {len(candidate_cache)} cached]' if candidate_cache else ''}")
                # 2.5: tie-break RNG seeded from schedule seed + replan index
                # (total_reassignments before this call = the replan index).
                voting_tie_seed = _derived_seed(schedule_seed, "ties", total_reassignments)
                total_reassignments += 1
                _, unassigned_robots, unassigned_tasks, reassign_score, reassign_length, reassign_per_agent_scores = V.reassign_robots_to_tasks(
                    robots, tasks, effective_candidates, voting_method, suitability_source,
                    unassigned_robots, unassigned_tasks, start_positions, goal_positions, HYPOTENUSE,
                    candidate_budget=voting_candidate_budget,
                    score_threshold=voting_score_threshold,
                    stall_k=voting_stall_k,
                    seed_assignment_ids=previous_best_id_pairs,
                    cached_candidate_ids=(candidate_cache or None),
                    top_k_out_ids=top_k_out_ids,
                    hall_of_fame_k=HALL_OF_FAME_K,
                )
                if reassign_per_agent_scores:
                    reassignment_jains_indices.append(calculate_jains_index(reassign_per_agent_scores))
                    reassignment_fairness_metrics.append(compute_all_fairness_metrics(reassign_per_agent_scores))
                # Capture this round's winner as ID pairs so the next voting
                # round can warm-start from it. The stealing block in
                # reassign_robots_to_tasks may have added more pairs after
                # return_assignments was populated, so read state off the
                # robot objects instead of the returned dict.
                previous_best_id_pairs = [
                    (r.robot_id, r.current_task.task_id)
                    for r in robots
                    if r.assigned and r.current_task is not None
                ] or None

                # Merge this round's top-K into the hall of fame (dedup by
                # canonical pair-set, keep most-recent HALL_OF_FAME_MAX).
                if top_k_out_ids:
                    existing_keys = {frozenset(c) for c in candidate_cache}
                    for cand in top_k_out_ids:
                        key = frozenset(cand)
                        if key not in existing_keys:
                            candidate_cache.append(cand)
                            existing_keys.add(key)
                    if len(candidate_cache) > HALL_OF_FAME_MAX:
                        candidate_cache = candidate_cache[-HALL_OF_FAME_MAX:]
            
            # assign with optimization
            elif not po_all_zero and voting_method in optimization_methods:
                # Murty's uses a special reassign function that generates
                # K-best alternatives for the replan-on-failure mechanism.
                if voting_method == O.murtys_task_allocation:
                    print(f"REASSIGNING WITH OPTIMIZATION METHOD: Murty's K-Best")
                    total_reassignments += 1
                    murtys_alternatives.clear()
                    _, unassigned_robots, unassigned_tasks, reassign_score, reassign_length, reassign_per_agent_scores = O.reassign_robots_to_tasks_with_murtys(
                        robots, tasks, num_candidates, voting_method, suitability_source,
                        unassigned_robots, unassigned_tasks, voting_method, start_positions, goal_positions, HYPOTENUSE,
                        k_alternatives=10, murtys_alternatives_out=murtys_alternatives,
                    )
                else:
                    print(f"REASSIGNING WITH OPTIMIZATION METHOD: {optimization_method_name}")
                    total_reassignments += 1
                    # 8.2: perturbed JV draws fresh noise per replan (the
                    # replan index IS total_reassignments after increment).
                    alloc_fn = voting_method
                    if voting_method is O.perturbed_jv_task_allocation:
                        alloc_fn = O.perturbed_jv_factory(
                            _derived_seed(run_seed, "pjv", total_reassignments))
                    _, unassigned_robots, unassigned_tasks, reassign_score, reassign_length, reassign_per_agent_scores = O.reassign_robots_to_tasks_with_method(
                        robots, tasks, num_candidates, voting_method, suitability_source,
                        unassigned_robots, unassigned_tasks, alloc_fn, start_positions, goal_positions, HYPOTENUSE,
                    )
                if reassign_per_agent_scores:
                    reassignment_jains_indices.append(calculate_jains_index(reassign_per_agent_scores))
                    reassignment_fairness_metrics.append(compute_all_fairness_metrics(reassign_per_agent_scores))

            # assign with direct LLM (bypass the matrix)
            elif not po_all_zero and getattr(voting_method, "_is_llm_direct", False):
                print("REASSIGNING WITH DIRECT LLM ASSIGNMENT")
                total_reassignments += 1
                _, unassigned_robots, unassigned_tasks, reassign_score, reassign_length, reassign_per_agent_scores = reassign_robots_to_tasks_direct(
                    robots, tasks, voting_method,
                    unassigned_robots, unassigned_tasks,
                    start_positions, goal_positions,
                    llm_cache=llm_cache,
                )
                if reassign_per_agent_scores:
                    reassignment_jains_indices.append(calculate_jains_index(reassign_per_agent_scores))
                    reassignment_fairness_metrics.append(compute_all_fairness_metrics(reassign_per_agent_scores))
            total_reassignment_time += reassign_length
            cumulative_reassignment_quality += reassign_score

            if po is not None:
                # PO safety net (spec §6.3/§6.4): never execute a pair the
                # allocator itself scored 0. Voting's internal all-zero-
                # candidate fallback can apply infeasible pairs; scrub them
                # here so they never travel and mis-fire at arrival. The
                # scrub uses the allocator's own view (masked under PO), so
                # optimistic masking still lets genuinely-misjudged pairs
                # through - those are the experiment's mis-assignments.
                for r in robots:
                    if (r.assigned and r.current_task is not None
                            and (r.current_task_suitability is None
                                 or float(r.current_task_suitability) == 0.0)):
                        unassign_task_from_robot(
                            r, r.current_task,
                            unassigned_robots=unassigned_robots,
                            unassigned_tasks=unassigned_tasks,
                        )

            if po is not None and not po_all_zero:
                # Post-allocation snapshot of true-vs-masked suitability over
                # the currently assigned pairs.
                po.note_round(robots, HYPOTENUSE)

            # update location dicts
            for robot in robots:
                if robot.assigned and robot.current_task:
                    start_positions[robot.robot_id] = tuple(robot.location)
                    goal_positions[robot.robot_id] = tuple(robot.current_task.location)
                else:
                    start_positions.pop(robot.robot_id, None)
                    goal_positions.pop(robot.robot_id, None)

        # Record any task that is currently assigned - this is the "attempted" set.
        # We do this every timestep (cheap) so cancellations of never-attempted
        # tasks do not incorrectly count them.
        for t in tasks:
            if t.assigned:
                tasks_ever_assigned.add(t.task_id)

        if po is not None:
            # End-of-tick PO bookkeeping: assignment-change detection (travel
            # resets, recovery-latency closeout) and masked-wait accumulation.
            po.end_of_tick(robots, tasks, tasks_ever_assigned, time_step)

        # rerun planner if needed, update state, and clear events.
        # On planner failure, try alternative assignments from the voting
        # candidate pool (for voting methods) or a perturbed re-solve (for
        # optimization methods). Cap at MAX_REPLAN_RETRIES alternatives; if
        # all fail, accept stale paths (no freeze — the stale paths are from
        # the last successful replan and may have minor collision risk, but
        # freezing would lose progress entirely).
        # Deterministic planner-work bounds (2026-10-26): the retries are
        # additionally bounded by PLANNER_CALLS_PER_TICK (calls per tick)
        # and PLANNER_FAILED_CALL_BUDGET (failed calls per run; after it the
        # run is planner-dead - no further planner calls). Count-based,
        # identical for every method, counted in Planner_Cap_Hits.
        MAX_REPLAN_RETRIES = 10
        # None = planner not called this tick; True/False = outcome.
        planner_okay = None
        # Deterministic per-tick planner-call counter (see
        # PLANNER_CALLS_PER_TICK): the replan call counts as call #1; every
        # replan-on-failure retry increments it.
        planner_calls_this_tick = 0
        if should_replan_paths and start_positions and goal_positions:
            if planner_failed_calls >= PLANNER_FAILED_CALL_BUDGET:
                # Per-run failed-call budget exhausted: planner-dead mode.
                # Skip the planner call entirely (stale paths stand) and
                # count the suppressed call. Deterministic and identical
                # for every method.
                planner_cap_hits += 1
            else:
                print("\nRERUNNING PLANNER\n")
                planner_calls_this_tick = 1
                new_len, planner_okay = run_planner(planner, robots, start_positions, goal_positions, map_dict)
                avg_path_length = new_len or avg_path_length

                if not planner_okay:
                    plan_failure_count += 1
                    planner_failed_calls += 1
                    if planner_dead_tick < 0 and planner_failed_calls >= PLANNER_FAILED_CALL_BUDGET:
                        planner_dead_tick = time_step
                    _entry = {"tick": time_step}
                    _entry.update(MW.LAST_PLAN_FAILURE or {"kind": "unknown"})
                    if _entry.get("kind") == "lacam_timeout":
                        lacam_timeout_count += 1
                    plan_failure_log.append(_entry)

                # ---- Replanning on plan failure ----
                if not planner_okay:
                    # Collect alternative assignments to try.
                    alternatives = []
                    if voting_method in voting_methods and top_k_out_ids:
                        # voting: top_k_out_ids has the ranked candidates from
                        # this round (index 0 = winner, already tried). The rest
                        # are alternatives.
                        for cand in top_k_out_ids[1:]:
                            alternatives.append(cand)

                    # Also try the hall-of-fame cache (older candidates that
                    # survived from previous rounds).
                    if voting_method in voting_methods and candidate_cache:
                        for cand in candidate_cache:
                            if cand not in alternatives:
                                alternatives.append(cand)

                    # For Murty's: use the K-best alternatives generated during
                    # the assignment. These are near-optimal perturbations that
                    # differ from the winner by ~1 pair.
                    if voting_method == O.murtys_task_allocation and murtys_alternatives:
                        for cand in murtys_alternatives:
                            alternatives.append(cand)

                    # For standard optimization (CBBA/SSIA/ILP/JV): re-solve with
                    # a shuffled unassigned list (perturbation). These optimizers
                    # are deterministic, so perturbation may produce the same
                    # solution — this is a known limitation documented in the paper.
                    if voting_method in optimization_methods and voting_method != O.murtys_task_allocation:
                        for _ in range(MAX_REPLAN_RETRIES):
                            alternatives.append("PERTURB")

                    print(f"  {len(alternatives)} alternatives available for replan (cap={MAX_REPLAN_RETRIES})")

                    # Try each alternative: unassign current pairs, apply the
                    # alternative, update positions, retry planner.
                    retried = 0
                    while not planner_okay and retried < min(len(alternatives), MAX_REPLAN_RETRIES):
                        # Deterministic planner-work bounds: stop retrying
                        # when the per-run failed-call budget or the
                        # per-tick planner-call cap is reached (counted in
                        # Planner_Cap_Hits; identical for every method).
                        if planner_failed_calls >= PLANNER_FAILED_CALL_BUDGET:
                            planner_cap_hits += 1
                            break
                        if planner_calls_this_tick >= PLANNER_CALLS_PER_TICK:
                            planner_cap_hits += 1
                            break
                        alt = alternatives[retried]
                        retried += 1
                        alternatives_tried += 1
                        print(f"  PLANNER FAILED — trying alternative assignment {retried}/{min(len(alternatives), MAX_REPLAN_RETRIES)}")

                        # Unassign all currently-assigned pairs that were set
                        # by this replan (i.e. all robots that have a task
                        # assigned). We need to be careful not to unassign
                        # robots that were already assigned BEFORE this replan
                        # tick — but in practice, reassign_robots_to_tasks only
                        # touches unassigned robots, so all current assignments
                        # were either pre-existing (untouched) or just set.
                        # We unassign ALL assigned robots that have a task,
                        # then re-apply the alternative. This is safe because
                        # the alternative covers the same robots/tasks.
                        from hvbta.allocators.misc_assignment import unassign_task_from_robot as _unassign
                        for r in robots:
                            if r.assigned and r.current_task:
                                t = r.current_task
                                _unassign(r, t, unassigned_robots, unassigned_tasks)

                        # Apply the alternative assignment.
                        if alt == "PERTURB":
                            # Re-solve with optimization (shuffled order)
                            import random as _rng
                            _rng.shuffle(unassigned_robots)
                            _rng.shuffle(unassigned_tasks)
                            # 8.2: each retry draws FRESH perturbed-JV noise
                            # (the increment below happens first in the normal
                            # path; here the noise seed must incorporate this
                            # retry, so derive from the NEXT reassignment index).
                            total_reassignments += 1
                            alloc_fn = voting_method
                            if voting_method is O.perturbed_jv_task_allocation:
                                alloc_fn = O.perturbed_jv_factory(
                                    _derived_seed(run_seed, "pjv", total_reassignments))
                            _, unassigned_robots, unassigned_tasks, reassign_score, reassign_length, reassign_per_agent_scores = O.reassign_robots_to_tasks_with_method(
                                robots, tasks, num_candidates, voting_method, suitability_source,
                                unassigned_robots, unassigned_tasks, alloc_fn, start_positions, goal_positions, HYPOTENUSE,
                            )
                            total_reassignment_time += reassign_length
                            cumulative_reassignment_quality += reassign_score
                        elif isinstance(alt, list):
                            # Voting alternative: list of (robot_id, task_id) pairs
                            for rid, tid in alt:
                                robot = next((r for r in robots if r.robot_id == rid), None)
                                task = next((t for t in tasks if t.task_id == tid), None)
                                if robot and task and not robot.assigned and not task.assigned:
                                    robot.current_task = task
                                    robot.assigned = True
                                    robot.tasks_attempted += 1
                                    task.assigned_robot = robot
                                    task.assigned = True
                                    if rid in unassigned_robots:
                                        unassigned_robots.remove(rid)
                                    if tid in unassigned_tasks:
                                        unassigned_tasks.remove(tid)
                                    # Use matrix lookup for suitability
                                    if isinstance(suitability_source, tuple) and len(suitability_source) == 3:
                                        full_matrix, r_idx_map, t_idx_map = suitability_source
                                        pair_score = float(full_matrix[r_idx_map[rid], t_idx_map[tid]])
                                    else:
                                        pair_score = float(suitability_source([robot], [task], map_size=HYPOTENUSE)[0, 0])
                                    robot.current_task_suitability = pair_score
                                    task.current_suitability = pair_score

                        # PO: alternatives are applied without a suitability filter,
                        # so scrub zero-suitability pairs here too - PO never
                        # executes a pairing the allocator itself scored 0 (the
                        # regular post-allocation scrub runs before this replan-
                        # on-failure section, so it misses these).
                        if po is not None:
                            for r in robots:
                                if (r.assigned and r.current_task is not None
                                        and (r.current_task_suitability is None
                                             or float(r.current_task_suitability) == 0.0)):
                                    unassign_task_from_robot(
                                        r, r.current_task,
                                        unassigned_robots=unassigned_robots,
                                        unassigned_tasks=unassigned_tasks,
                                    )

                        # Update positions for the new assignment.
                        for robot in robots:
                            if robot.assigned and robot.current_task:
                                start_positions[robot.robot_id] = tuple(robot.location)
                                goal_positions[robot.robot_id] = tuple(robot.current_task.location)
                            else:
                                start_positions.pop(robot.robot_id, None)
                                goal_positions.pop(robot.robot_id, None)

                    # Retry planner.
                    if start_positions and goal_positions:
                        planner_calls_this_tick += 1
                        new_len, planner_okay = run_planner(planner, robots, start_positions, goal_positions, map_dict)
                        avg_path_length = new_len or avg_path_length
                        if not planner_okay:
                            plan_failure_count += 1
                            planner_failed_calls += 1
                            if planner_dead_tick < 0 and planner_failed_calls >= PLANNER_FAILED_CALL_BUDGET:
                                planner_dead_tick = time_step
                            _entry = {"tick": time_step}
                            _entry.update(MW.LAST_PLAN_FAILURE or {"kind": "unknown"})
                            if _entry.get("kind") == "lacam_timeout":
                                lacam_timeout_count += 1
                            plan_failure_log.append(_entry)
                        if planner_okay:
                            print(f"  ALTERNATIVE {retried} SUCCEEDED — new plan accepted.")

                    if not planner_okay:
                        print("  ALL ALTERNATIVES FAILED — accepting stale paths.")

            if planner_okay and pending_event_steps:
                # Adaptation latency: elapsed timesteps from each event to this
                # successful resolution. Drain the queue.
                for ev_step in pending_event_steps:
                    adaptation_latencies.append(time_step - ev_step)
                pending_event_steps.clear()
            previous_active, previous_goals, previous_unassigned_robots, previous_unassigned_tasks = state_check(robots, unassigned_robots, unassigned_tasks)
            events = {k: 0 for k in events}

        # Invariant check: a robot with an assigned task and an EMPTY
        # current_path after a SUCCESSFUL plan call is stuck - it can
        # neither move nor work. Counted per tick in
        # Stuck_Assigned_Robot_Ticks; must be 0. (Ticks after planner
        # FAILURES legitimately leave newly-assigned robots pathless -
        # the documented accept-stale-paths behavior - so they are not
        # counted; verified empirically that no stuck robot survives a
        # successful plan call.)
        if planner_okay:
            stuck_assigned_robot_ticks += sum(
                1 for r in robots
                if r.assigned and r.current_task is not None and not r.current_path
            )

        # ---- SIMULATE TIMESTEP (the LAST thing in each tick) ----
        # Robots move on the plan computed above, which accounts for all
        # events applied at the top of this tick. Completions, battery
        # deaths, reveals, and nav drops happen here; they are processed
        # at the START of the next tick (completed_prev_timestep).
        completed_this_step, _unassigned_count, total_reward, total_success = simulate_time_step(
            robots, tasks, unassigned_robots, unassigned_tasks,
            suitability_method, occupied_positions, start_positions,
            goal_positions, 1.0, total_reward, total_success,
            po_state=po, current_tick=time_step, pending_event_steps=pending_event_steps,
            obstacles=_obstacle_set,
            collision_counters=collision_counters,
        )
        completed_prev_timestep = completed_this_step
        # Replan-on-deviation: the timestep flagged an off-plan wait or
        # a plan-less assigned robot; the top of the next tick replans.
        if collision_counters.get("plan_deviation"):
            deviation_replan_pending = True

        # Per-step throughput signal (used for mean + variance at end).
        completed_per_step.append(completed_this_step)
        # Phase 0: log pool/fleet size every timestep, right after the step.
        task_pool_size_per_step.append(len(tasks))
        fleet_size_per_step.append(len(robots))

        # Record the first moment the task pool hits 0 — useful METRIC even
        # when we don't terminate (the pool may refill from future arrivals).
        if len(tasks) == 0 and all_tasks_completed_at_step is None:
            all_tasks_completed_at_step = time_step + 1

        # Termination: pool empty AND no future task arrivals in the schedule.
        no_future_tasks = (
            event_schedule is None
            or all(len(e["add_tasks"]) == 0 for e in event_schedule[time_step + 1:])
        )
        if len(tasks) == 0 and no_future_tasks:
            print(f"All tasks completed at time step {time_step + 1} (no future arrivals).")
            total_time_steps = time_step + 1
            break

    task_completion_fraction = total_success / total_tasks if total_tasks else 0.0
    attempted_completion_rate = total_success / len(tasks_ever_assigned) if tasks_ever_assigned else 0.0
    avg_reassignment_jains_index = (sum(reassignment_jains_indices) / len(reassignment_jains_indices)) if reassignment_jains_indices else 0.0

    if reassignment_fairness_metrics:
        num_metrics = len(FAIRNESS_METRIC_NAMES)
        avg_reassign_metrics = tuple(
            sum(m[i] for m in reassignment_fairness_metrics) / len(reassignment_fairness_metrics)
            for i in range(num_metrics)
        )
    else:
        avg_reassign_metrics = tuple(0.0 for _ in FAIRNESS_METRIC_NAMES)

    completion_step_out = all_tasks_completed_at_step if all_tasks_completed_at_step is not None else -1

    # Primary comparative metrics: throughput, adaptation latency, workload variance.
    throughput_mean = float(np.mean(completed_per_step)) if completed_per_step else 0.0
    throughput_var = float(np.var(completed_per_step)) if completed_per_step else 0.0
    adaptation_latency_mean = float(np.mean(adaptation_latencies)) if adaptation_latencies else 0.0
    adaptation_latency_max = float(max(adaptation_latencies)) if adaptation_latencies else 0.0
    task_success_counts = [r.tasks_successful for r in robots]
    task_attempted_counts = [r.tasks_attempted for r in robots]
    workload_variance = float(np.var(task_success_counts)) if task_success_counts else 0.0
    attempted_workload_variance = float(np.var(task_attempted_counts)) if task_attempted_counts else 0.0
    # Urgency + pressure summaries. Bucketing key for downstream analysis:
    #   "none urgent, no pressure"    -> Percentage_Urgent_Mean == 0 AND Robot_Pressure_Fraction == 0
    #   "some urgent, no pressure"    -> Percentage_Urgent_Mean > 0  AND Robot_Pressure_Fraction == 0
    #   "none urgent, some pressure"  -> Percentage_Urgent_Mean == 0 AND Robot_Pressure_Fraction > 0
    #   "all urgent, high pressure"   -> Percentage_Urgent_Mean == 1 AND Robot_Pressure_Fraction > 0
    percentage_urgent_mean = float(np.mean(percentage_urgent_per_replan)) if percentage_urgent_per_replan else 0.0
    percentage_urgent_max = float(max(percentage_urgent_per_replan)) if percentage_urgent_per_replan else 0.0
    robot_pressure_fraction = float(np.mean(robot_pressure_per_replan)) if robot_pressure_per_replan else 0.0

    print(f"Voting: "
          f"Task completion fraction: {task_completion_fraction:.2%} ({total_success}/{total_tasks} spawned), "
          f"Attempted completion rate: {attempted_completion_rate:.2%} ({total_success}/{len(tasks_ever_assigned)} attempted), "
          f"Completion step: {completion_step_out}, "
          f"Throughput: mean={throughput_mean:.3f}/step var={throughput_var:.3f}, "
          f"Adaptation latency: mean={adaptation_latency_mean:.2f} max={adaptation_latency_max:.0f} (over {len(adaptation_latencies)} events), "
          f"Workload variance: {workload_variance:.3f}, "
          f"Attempted workload variance: {attempted_workload_variance:.3f}, "
          f"Urgency mean={percentage_urgent_mean:.2%} max={percentage_urgent_max:.2%}, "
          f"Robot-pressure fraction={robot_pressure_fraction:.2%}, "
          f"Reassignment Time: {total_reassignment_time}, "
          f"Reassignment quality: {cumulative_reassignment_quality}, reassignments: {total_reassignments}, "
          f"Robots: {len(robots)}")

    # Phase 0: emit the per-step pool/fleet series on a single line each so a
    # pilot run's stdout can be scraped to plot pool size vs. timestep, measure
    # the transient length, and check for fleet-size drift. Tagged prefixes
    # (POOL_SERIES / FLEET_SERIES) make them easy to grep out of a log.
    print(f"POOL_SERIES\t" + ",".join(str(n) for n in task_pool_size_per_step))
    print(f"FLEET_SERIES\t" + ",".join(str(n) for n in fleet_size_per_step))

    # Phase 4: empirical regime-classification metrics computed from the
    # pool-size log. Mean_Task_Pool_Size = average pool size across the run.
    # Pool_Size_Trend = mean(last-quartile) - mean(first-quartile); near 0 =>
    # steady state, strongly positive => overload.
    mean_task_pool_size = float(np.mean(task_pool_size_per_step)) if task_pool_size_per_step else 0.0
    if len(task_pool_size_per_step) >= 4:
        q = len(task_pool_size_per_step) // 4
        first_q = float(np.mean(task_pool_size_per_step[:q]))
        last_q = float(np.mean(task_pool_size_per_step[-q:]))
        pool_size_trend = last_q - first_q
    else:
        pool_size_trend = 0.0

    # Partial-observability outcome metrics (spec §7). Sentinels when the
    # machinery is inactive: counts 0, rates/means -1.
    if po is not None:
        po_metrics = po.finalize(tasks, tasks_ever_assigned)
        if po_diag_path:
                planner_calls = MW.PLANNER_TIMING["calls"]
                planner_ns = MW.PLANNER_TIMING["ns"]
                planner_agents = MW.PLANNER_TIMING["agents_sum"]
                po.dump_diagnostics(
                    po_diag_path,
                    scenario=scenario,
                    planner=planner,
                    mask_level=mask_level,
                    mask_policy=mask_policy,
                    method=func_name(voting_method),
                    plan_failure_log=plan_failure_log,
                    collision_counters=dict(collision_counters),
                    run_timed_out=run_timed_out,
                    planner_calls=planner_calls,
                    planner_total_ms=planner_ns / 1e6,
                    planner_mean_ms_per_call=(planner_ns / 1e6 / planner_calls) if planner_calls else 0.0,
                    planner_mean_agents_per_call=(planner_agents / planner_calls) if planner_calls else 0.0,
                )
    else:
        po_metrics = {
            "mis_assignment_count": 0,
            "mis_assignment_rate": -1.0,
            "wasted_travel_steps": 0,
            "wasted_travel_fraction": -1.0,
            "masked_starvation_count": 0,
            "mean_masked_wait": -1.0,
            "reveal_events": 0,
            "recovery_latency_mean": -1.0,
            "true_assignment_suitability": -1.0,
            "masked_view_suitability": -1.0,
            "nav_drop_count": 0,
            "nav_drop_wasted_travel": 0,
            "all_zero_skip_ticks": 0,
        }

    # Tasks spawned but never assigned to any robot, regardless of masking -
    # the capacity baseline Masked_Starvation_Count must be compared against
    # (spec §3a: ~45 tasks are never attempted even at Mask_Level=none).
    never_attempted_count = total_tasks - len(tasks_ever_assigned)

    # Pool diagnostics (2.8): aggregate V.POOL_STATS into the two CSV columns.
    # Threshold_Fire_Rate = share of anytime-active calls (legacy threshold
    # OR the 2026-10-27 stall rule) where early termination cut the pool
    # short of its full size; Mean_Pool_Size_At_Rank = mean ranked-pool size
    # across all voting calls this run. -1 marks N/A (no voting calls -
    # e.g. Murty's rows, or runs with no voting allocation).
    if V.POOL_STATS:
        _th_calls = [s for s in V.POOL_STATS
                     if s["threshold"] is not None or s.get("stall_k") is not None]
        _fired = sum(1 for s in _th_calls if s["ranked"] < s["full"])
        threshold_fire_rate = (_fired / len(_th_calls)) if _th_calls else -1.0
        mean_pool_size_at_rank = float(np.mean([s["ranked"] for s in V.POOL_STATS]))
    else:
        threshold_fire_rate = -1.0
        mean_pool_size_at_rank = -1.0
    # Diagnostic archive hook: hand this run's per-call pool stats to the
    # sink before clearing them (stall-rule report driver only; inert in
    # sweeps).
    if V.POOL_STATS and getattr(V, "POOL_STATS_SINK", None) is not None:
        V.POOL_STATS_SINK.extend(V.POOL_STATS)
    V.POOL_STATS = None

    return (
        # Section 4 - aggregate outcome (kept only)
        task_completion_fraction,
        total_reassignment_time, cumulative_reassignment_quality, total_reassignments,
        min(total_time_steps, max_time_steps),
        avg_reassignment_jains_index,
        # Section 5 - lifelong metrics (kept only)
        attempted_completion_rate, len(tasks_ever_assigned), total_tasks,
        completion_step_out,
        # Section 6 - primary comparative metrics (kept only)
        throughput_mean, throughput_var,
        adaptation_latency_mean, adaptation_latency_max,
        workload_variance,
        attempted_workload_variance,
        # Section 7 - full-schedule denominators, constant across methods in a (rep, sm) group.
        full_schedule_tasks_spawned, full_schedule_robots_max,
        # Section 8 - urgency + pressure summaries.
        percentage_urgent_mean, percentage_urgent_max, robot_pressure_fraction,
    ) + avg_reassign_metrics[1:] + (  # skip first (Jains) - already exposed above
        # Section 11 - empirical regime classification + horizon provenance
        mean_task_pool_size, pool_size_trend,
        scheduled_event_count, horizon_binding_term,
        # Section 12 - planner failure count + alternatives tried
        plan_failure_count,
        alternatives_tried,
        # Section 15 - partial-observability outcome metrics (spec §7).
        # Order matches the CSV header; -1 marks N/A (PO machinery inactive
        # or empty denominators).
        po_metrics["mis_assignment_count"],
        po_metrics["mis_assignment_rate"],
        po_metrics["wasted_travel_steps"],
        po_metrics["wasted_travel_fraction"],
        po_metrics["masked_starvation_count"],
        po_metrics["mean_masked_wait"],
        po_metrics["reveal_events"],
        po_metrics["recovery_latency_mean"],
        po_metrics["true_assignment_suitability"],
        po_metrics["masked_view_suitability"],
        # 8.1: Observed_View_Suitability - mean pair score on the
        # allocator's ACTUAL view (noisy under the score-noise
        # experiment, masked under PO). Identical to
        # Masked_View_Suitability by construction; kept as a separate
        # column so the noise experiment's observed-view metric is never
        # confused with the PO mask metric. -1 = N/A.
        (po_metrics["masked_view_suitability"] if noise_sigma is not None else -1.0),
        po_metrics["nav_drop_count"],
        po_metrics["nav_drop_wasted_travel"],
        # Section 15b - PO baseline + invariant diagnostics
        never_attempted_count,
        all_zero_skip_ticks,
        stuck_assigned_robot_ticks,
        # Section 15b-2 - collision handling diagnostics (Phase 3v6)
        collision_counters["planned_conflicts"],
        collision_counters["nudge"],
        collision_counters["violations"],
        collision_counters.get("propagated_waits", 0),
        collision_counters.get("mover_conflict_waits", 0),
        # Run-level wall-clock limit flag (1 = hit the limit, stopped early)
        1 if run_timed_out else 0,
        # Section 15c - anytime pool diagnostics (2.8)
        threshold_fire_rate,
        mean_pool_size_at_rank,
        # Section 15d - planner-call accounting (PP-pilot diagnostics,
        # 2026-10-26). Planner_Call_Count is a deterministic per-run count
        # (initial call + replan calls + replan-on-failure retries);
        # Planner_Total_ms is wall-clock and excluded from comparisons;
        # Planner_Cap_Hits counts every deterministic planner-work cap
        # firing (per-tick planner-call cap, per-run failed-call budget,
        # PP per-call expansion budget) - must be 0 in runs that stay
        # within normal planner behavior.
        planner_timing["calls"],
        planner_timing["ns"] / 1e6,
        planner_cap_hits + (_PP.PP_EXPANSION_CAP_HITS - _pp_cap_hits_start),
        # First tick at which the run went planner-dead (the per-run
        # failed-call budget was exhausted); -1 = never went dead.
        # Deterministic; reported per method by sweep_report.py.
        planner_dead_tick,
        # Failed planner calls that hit the wall-clock time budget
        # (kind == "lacam_timeout"). Logging-only (2026-10-27): lets the
        # CSV distinguish LaCAM budget hits from no-solution failures.
        lacam_timeout_count,
    )


def benchmark_simulation(
        output: tuple,
        robots: List[CapabilityProfile],
        tasks: List[TaskDescription],
        num_candidates: int,
        voting_method: callable,
        grid: List[List[int]],
        map_dict: dict,
        suitability_method: callable,
        suitability_matrix: np.ndarray,
        max_time_steps: int,
        add_tasks: bool,
        add_robots: bool,
        remove_robots: bool,
        tasks_to_add: int = 1,
        robots_to_add: int = 1,
        robots_to_remove: int = 1,
        robot_generation_strict: bool = True,
        task_generation_strict: bool = True,
        initial_jains_index: float = 0.0,
        task_arrival_rate: float = 0.05,
        robot_arrival_rate: float = 0.02,
        robot_departure_rate: float = 0.02,
        task_cancellation_rate: float = 0.02,
        remove_tasks: bool = False,
        tasks_to_remove: int = 1,
        event_schedule: list = None,
        mask_level: str = None,
        mask_policy: str = None,
        # 8.1 score-noise experiment sigma (None = off).
        noise_sigma: float = None,
        schedule_seed: int = None,
        # Per-run RNG seed keyed by (group, method): derived by the sweep via
        # _derived_seed(schedule_seed, "run", method). Deliberately does NOT
        # include mask_level/mask_policy (or any future observation-noise
        # sigma): conditions that share robots/tasks/schedule must share the
        # run seed, so e.g. sigma_obs = 0 reproduces the no-noise results
        # exactly. Seeding both `random` and `np.random` with it at entry
        # makes each method's run reproducible regardless of execution order
        # or what any other method consumed before it.
        run_seed: int = None,
        po_diag_path: str = None,
        planner: str = "lacam",
        scenario: str = "",
        scheduled_event_count: int = 0,
        horizon_binding_term: str = ""):
    start_time = time.perf_counter_ns()
    output_tuple = main_simulation(
        output, robots, tasks, num_candidates, voting_method,
        grid, map_dict, suitability_method, suitability_matrix,
        max_time_steps, add_tasks, add_robots, remove_robots,
        tasks_to_add, robots_to_add, robots_to_remove,
        robot_generation_strict, task_generation_strict,
        initial_jains_index=initial_jains_index,
        task_arrival_rate=task_arrival_rate,
        robot_arrival_rate=robot_arrival_rate,
        robot_departure_rate=robot_departure_rate,
        task_cancellation_rate=task_cancellation_rate,
        remove_tasks=remove_tasks,
        tasks_to_remove=tasks_to_remove,
        event_schedule=event_schedule,
        mask_level=mask_level,
        mask_policy=mask_policy,
        noise_sigma=noise_sigma,
        schedule_seed=schedule_seed,
        run_seed=run_seed,
        po_diag_path=po_diag_path,
        planner=planner,
        scenario=scenario,
        scheduled_event_count=scheduled_event_count,
        horizon_binding_term=horizon_binding_term,
    )
    execution_time = time.perf_counter_ns() - start_time
    print(f"Simulation completed in {execution_time:.5f} nanoseconds.")
    return output_tuple + (execution_time,)


def build_initial_prefix(run_id, method_name, sm_name, num_robots, num_tasks, nc,
                         score, planning_time_us, per_agent_scores):
    """Initial-quality row prefix (13 items, BEFORE planner injection) -
    the shared builder used by BOTH entry points (serial __main__ and the
    phase-4 pool driver) so their CSV schemas are identical by
    construction (schema unification, 2026-10-27)."""
    metrics = compute_all_fairness_metrics(per_agent_scores)
    assigned_count = len(per_agent_scores) if per_agent_scores else 0
    # arithmetic mean suitability across assigned pairs (score / assigned_count).
    mean_assignment_suitability = (score / assigned_count) if assigned_count > 0 else 0.0
    return [run_id, method_name, sm_name, num_robots, num_tasks, nc, score,
            mean_assignment_suitability, planning_time_us] + list(metrics)


def _record_assignment(assignment_infos, run_id, method_name, sm_name, num_robots,
                       num_tasks, nc, score, planning_time_us, per_agent_scores):
    """Compute initial fairness metrics, derived score fields, and append a row."""
    assignment_infos.append(
        build_initial_prefix(run_id, method_name, sm_name, num_robots, num_tasks,
                             nc, score, planning_time_us, per_agent_scores))
    return assignment_infos[-1][9]  # initial jains


def make_run_id(map_name, num_robots, num_tasks, scenario, rep):
    """Deterministic per-group Run ID: same config -> same ID, independent
    of run order or resume state (the serial writer's old on-disk counter
    was order-dependent; unified 2026-10-27). Shared by both entry points
    and the profiles files so (Run ID, Map) joins cleanly."""
    return f"{map_name}_{num_robots}x{num_tasks}_{scenario}_rep{rep}"


def _initial_quality_fields(po_cond, out, score, per_agent_scores,
                            robots, tasks, hypotenuse):
    """PO true-score rule (the serial writer's _record_initial): under PO
    conditions the REPORTED initial quality is computed on TRUE
    (un-masked / un-noised) pair scores, not the allocator's view.
    Returns (total_score, per_agent_scores)."""
    if po_cond is None or not out:
        return score, per_agent_scores
    pairs = out[0] if isinstance(out, tuple) else out
    true_per_agent = [
        S._balanced_pair(robots[r], tasks[t], map_size=hypotenuse)
        for r, t in pairs
    ]
    return sum(true_per_agent), true_per_agent


def build_result_row(row_prefix, output_tuple, scenario_name, regime,
                     mask_level, mask_count, mask_policy, noise_sigma, map_class):
    """Assemble the full 85-column result row (serial schema):
    row_prefix (14, with planner injected) + output_tuple (60) +
    config suffix (10) + map class (1). Shared by both entry points."""
    return (list(row_prefix) + list(output_tuple) + [
        scenario_name,
        regime["task_arrival_rate"], regime["task_cancellation_rate"],
        regime["robot_arrival_rate"], regime["robot_departure_rate"],
        mask_level, mask_count, mask_policy,
        noise_sigma if noise_sigma is not None else "",
        1 if noise_sigma is not None else 0,
    ] + [map_class])


def build_profiles_row(run_id, map_name, num_robots, num_tasks, sm_name,
                       robot_profiles, task_profiles):
    """Profiles-file row (7 columns), shared by both entry points."""
    return [run_id, map_name, num_robots, num_tasks, sm_name,
            robot_profiles, task_profiles]


# Canonical result-row header (84 columns; the serial writer's exact
# schema). Shared with the phase-4 pool driver so both entry points write
# identical schemas (unification, 2026-10-27).
RESULT_ROW_PREFIX_HEADER = [
    'Run ID', 'Method', 'Suitability Method', 'Planner',
    'Num Robots', 'Num Tasks', 'Num Candidates',
    'Total Score', 'Mean_Assignment_Suitability', 'Planning_Time_us',
    'Init_Jains_Index', 'Init_Gini', 'Init_Median', 'Init_IQR',
]
RESULT_ROW_OUTPUT_HEADER = [
    # Aggregate simulation outcome
    'Task_Completion_Fraction',
    'total_reassignment_time', 'Cumulative_Reassignment_Quality', 'total_reassignments',
    'Total Time Steps', 'Avg Reassignment Jains Index',
    # Lifelong-operation metrics
    'Attempted_Completion_Rate', 'Tasks_Attempted', 'Tasks_Spawned', 'Completion_Step',
    # Primary comparative metrics
    'Throughput_Mean', 'Throughput_Var',
    'Adaptation_Latency_Mean', 'Adaptation_Latency_Max',
    'Workload_Variance', 'Attempted_Workload_Variance',
    # Full-schedule denominators
    'Full_Schedule_Tasks_Spawned', 'Full_Schedule_Robots_Max',
    # Urgency + pressure
    'Percentage_Urgent_Mean', 'Percentage_Urgent_Max', 'Robot_Pressure_Fraction',
    # Reassignment fairness (Jains already above)
    'Avg_Reass_Gini', 'Avg_Reass_Median', 'Avg_Reass_IQR',
    # Empirical regime classification
    'Mean_Task_Pool_Size', 'Pool_Size_Trend',
    # Horizon provenance
    'Scheduled_Event_Count', 'Horizon_Binding_Term',
    # Planner failures + alternatives
    'Plan_Failure_Count', 'Alternatives_Tried',
    # Partial-observability outcome metrics (-1 = N/A)
    'Mis_Assignment_Count', 'Mis_Assignment_Rate',
    'Wasted_Travel_Steps', 'Wasted_Travel_Fraction',
    'Masked_Starvation_Count', 'Mean_Masked_Wait', 'Reveal_Events',
    'Mis_Assignment_Recovery_Latency',
    'True_Assignment_Suitability', 'Masked_View_Suitability',
    'Observed_View_Suitability',
    'Nav_Drop_Count', 'Nav_Drop_Wasted_Travel',
    # PO baseline + invariant diagnostics
    'Never_Attempted_Count', 'All_Zero_Skip_Ticks', 'Stuck_Assigned_Robot_Ticks',
    # Collision handling diagnostics
    'Planned_Conflict_Count', 'Safety_Nudge_Count', 'Collision_Violation_Count',
    'Propagated_Wait_Count', 'Mover_Conflict_Waits',
    # Run wall-clock limit flag
    'Run_Timed_Out',
    # Anytime pool diagnostics
    'Threshold_Fire_Rate', 'Mean_Pool_Size_At_Rank',
    # Planner-call accounting
    'Planner_Call_Count', 'Planner_Total_ms', 'Planner_Cap_Hits',
    'Planner_Dead_Tick',
    # Failed planner calls that hit the wall-clock time budget
    'Planner_Timeout_Count',
    # Overhead
    'Execution Time',
]
RESULT_ROW_SUFFIX_HEADER = [
    'Scenario', 'Task_Arrival_Rate', 'Task_Cancellation_Rate',
    'Robot_Arrival_Rate', 'Robot_Departure_Rate',
    'Mask_Level', 'Mask_Count', 'Mask_Policy',
    'Obs_Noise_Sigma', 'Noise_Experiment',
    'Map Size',
]
RESULT_ROW_HEADER = (RESULT_ROW_PREFIX_HEADER + RESULT_ROW_OUTPUT_HEADER
                     + RESULT_ROW_SUFFIX_HEADER)
assert len(RESULT_ROW_HEADER) == 85, len(RESULT_ROW_HEADER)
PROFILES_ROW_HEADER = ['Run_ID', 'Map', 'Num Robots', 'Num Tasks',
                       'Suitability Method', 'RobotProfiles', 'TaskProfiles']


def _scan_completed_runs(dir_path):
    """
    Resume support: scan the output directory's simulation CSVs for runs
    that are already on disk so an interrupted sweep continues instead of
    restarting. Rows are written (and flushed) per completed run, so a
    killed process leaves a valid prefix.

    Returns ({group_key: completed_row_count}, total_rows) where a
    group_key is (map, num_robots, num_tasks, planner, scenario,
    suitability_method, mask_level, mask_policy, method). Legacy files
    without the Mask_* columns are treated as mask_level/policy = None.
    Truncated trailing rows (mid-write kill) are dropped and the file
    is rewritten cleanly.
    """
    counts = {}
    total = 0
    for path in sorted(glob.glob(os.path.join(dir_path, '*simulation*.csv'))):
        base = os.path.basename(path)
        m = re.search(r'_simulation_(.+)\.csv$', base)
        if not m:
            continue
        map_name = m.group(1)
        try:
            with open(path, newline='', encoding='utf-8') as f:
                rows = list(csv.reader(f))
        except OSError:
            continue
        if not rows:
            continue
        header, body = rows[0], rows[1:]
        idx = {name: i for i, name in enumerate(header)}
        need = ('Method', 'Suitability Method', 'Planner', 'Num Robots', 'Num Tasks', 'Scenario')
        if any(n not in idx for n in need):
            continue
        mi, si, pi = idx['Method'], idx['Suitability Method'], idx['Planner']
        ri, ti, ci = idx['Num Robots'], idx['Num Tasks'], idx['Scenario']
        li, yi = idx.get('Mask_Level'), idx.get('Mask_Policy')
        complete = [r for r in body if len(r) == len(header)]
        dropped = len(body) - len(complete)
        if dropped:
            print(f"RESUME: dropping {dropped} truncated row(s) from {base}")
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(complete)
        for r in complete:
            try:
                ml = (r[li] or None) if li is not None else None
                mp = (r[yi] or None) if yi is not None else None
                # 8.1: the noise sigma is part of the resume key so
                # noise-mode rows never resume-skip across sigmas.
                ni = idx.get('Obs_Noise_Sigma')
                ns = (float(r[ni]) if ni is not None and r[ni] not in ("", None) else None)
                key = (map_name, int(r[ri]), int(r[ti]), r[pi], r[ci], r[si], ml, mp, ns, r[mi])
            except (ValueError, IndexError):
                continue
            counts[key] = counts.get(key, 0) + 1
            total += 1
    return counts, total


if __name__ == "__main__":
    # Ablation controls (env-driven so parallel runs need no file edits):
    #   HVBTA_ZERO_PAIR  ("1"/"0") -> V.VOTING_ZERO_PAIR_FILTER
    #   HVBTA_THRESHOLD   ("1"/"0") -> V.VOTING_COVERAGE_THRESHOLD
    #   HVBTA_VOTING_ONLY ("1"/"0") -> drop optimization methods (Murty's
    #     rows are invariant across the flag pair, verified in v9; voting
    #     rows are the ablation's subject)
    #   HVBTA_OUTDIR      -> PO_OUTPUT_DIRNAME
    ABLATION_ZP = os.environ.get("HVBTA_ZERO_PAIR", "1") == "1"
    ABLATION_TH = os.environ.get("HVBTA_THRESHOLD", "1") == "1"
    ABLATION_VOTING_ONLY = os.environ.get("HVBTA_VOTING_ONLY", "0") == "1"
    # 4d (2026-10-26): f for the urgent good-enough threshold (stop when a
    # candidate's coverage-normalized quality >= f * UB). Chosen by the
    # user from the 4c calibration report; unset = legacy fixed 0.7.
    if "HVBTA_UB_F" in os.environ:
        V.VOTING_UB_THRESHOLD_F = float(os.environ["HVBTA_UB_F"])
    # 8.2: the perturbed-JV baseline's own noise sigma. Chosen by the
    # duplicate-retry rule (see docs/checkins/2026-10-26-step8.md); set
    # via env so drivers need no file edits. Default 0.05 per the
    # analysis-plan amendment — the pool driver (phase4_sweep.run_group)
    # hard-sets 0.05, and the serial default of None made serial runs JV
    # unperturbed (a driver mismatch, fixed 2026-10-27).
    O.PERTURBED_JV_SIGMA = (float(os.environ["HVBTA_PJV_SIGMA"])
                             if "HVBTA_PJV_SIGMA" in os.environ else 0.05)
    voting_methods = [
        V.rank_assignments_borda,
        V.rank_assignments_approval,
        V.rank_assignments_majority_judgment,
        V.rank_assignments_cumulative_voting,
        V.rank_assignments_condorcet_method,
        V.rank_assignments_range,
    ]
    voting_names = [func_name(f) for f in voting_methods]
    optimization_methods = [
        # O.cbba_task_allocation,
        # O.ssia_task_allocation,
        # O.ilp_task_allocation,
        # O.jv_task_allocation,
        O.murtys_task_allocation,
        # 8.2: perturbed JV baseline. Every sweep runs 8 allocators.
        O.perturbed_jv_task_allocation,
    ]
    optimization_names = [func_name(f) for f in optimization_methods]
    if ABLATION_VOTING_ONLY:
        optimization_methods = []
        optimization_names = []
    # Direct-assignment methods: the LLM returns (robot_id, task_id) pairs straight up,
    # bypassing the suitability-matrix + voting/optimization stack. Slot here so it
    # benchmarks alongside the others.
    direct_methods = [
        # S.bypass_suitability_from_names_with_llm,
    ]
    direct_names = [func_name(f) for f in direct_methods]
    all_methods = voting_methods + optimization_methods + direct_methods
    # Pretty display names keyed by function object - used for CSV rows and
    # for the resume-skip key so it matches what's on disk exactly.
    method_pretty = {fn: name for fn, name in
                     zip(all_methods, voting_names + optimization_names + direct_names)}
    suitability_methods = [
        S.evaluate_suitability_balanced,
        # S.evaluate_suitability_loose,
        # S.evaluate_suitability_strict,
        # S.evaluate_suitability_from_names_with_llm,
    ]

    small_maps = [
        r"lak105d.map", r"lak102d.map", r"lak107d.map",
    ]
    medium_maps = [
        r"arena.map", r"isound1.map", r"den204d.map",
    ]
    large_maps = [
        r"lak307d.map", r"den001d.map", r"den901d.map",
    ]

    # ============================================================
    # SWEEP CONFIGURATION
    # ============================================================
    # Two configurations below: the OLD one (flat sweep, all fleet
    # sizes on all maps) and the NEW one (balanced pairing, fleet
    # sizes matched to map sizes so 10-robot fleets don't end up
    # on 193x177 maps with 200-step paths and near-zero throughput).
    #
    # Uncomment the one you want to run. The old config is needed
    # to finish interrupted runs from the previous sweep.
    # ============================================================

    # --- OLD CONFIG: flat sweep (all fleet sizes on all maps) ---
    # Uses combat.map + combat2.map (very large, very open).
    # All robot/task sizes run on every map. This produces low
    # completion for small fleets on large maps, but was the
    # configuration used for the first (interrupted) sweep.
    OLD_SWEEP_CONFIGS = [
        (r"combat.map",  [10, 20, 30, 40, 50, 100], [10, 20, 30, 40, 50, 100]),
        (r"combat2.map", [10, 20, 30, 40, 50, 100], [10, 20, 30, 40, 50, 100]),
    ]

    # --- NEW CONFIG: balanced pairing (fleet matched to map size) ---
    # Small maps (~25-30 cells): 10-20 robots/tasks
    # Medium maps (~49-66 cells): 20-30-50 robots/tasks
    # Large maps (~84-128 cells): 50-100 robots/tasks
    # Prevents the 10-robot-on-193x177 mismatch that tanked completion.
    NEW_SWEEP_CONFIGS = [
        # Small maps
        (r"lak105d.map", [10, 20], [10, 20]),
        (r"lak102d.map", [10, 20], [10, 20]),
        (r"lak107d.map", [10, 20], [10, 20]),
        # Medium maps
        (r"arena.map",   [20, 30, 50], [20, 30, 50]),
        (r"isound1.map", [20, 30, 50], [20, 30, 50]),
        (r"den204d.map", [20, 30, 50], [20, 30, 50]),
        # Large maps — higher fleet sizes so robots can actually
        # cover the map and complete tasks within the horizon
        (r"lak307d.map", [50, 75, 100], [50, 75, 100]),
        (r"den001d.map", [50, 75, 100], [50, 75, 100]),
        (r"den901d.map", [50, 75, 100], [50, 75, 100]),
    ]

    # --- CONSTRAINED CONFIG: small high-obstacle maps ---
    # Tests whether constrained environments (tight corridors, dead-ends)
    # change the voting-vs-optimization balance. These maps have 55-66%
    # obstacles, forcing agents through narrow passages where planner
    # failures are more likely and the replan-on-failure mechanism
    # matters more. Fleet sizes kept small so paths are short enough
    # for tasks to actually complete.
    CONSTRAINED_SWEEP_CONFIGS = [
        (r"lak101d.map", [10, 20], [10, 20]),           # 31x30, 66% obs, 318 free
        (r"den404d.map", [10, 20], [10, 20]),           # 34x28, 62% obs, 358 free
        (r"hrt002d.map", [10, 20, 30], [10, 20, 30]),   # 50x49, 69% obs, 754 free
        (r"den009d.map", [10, 20, 30], [10, 20, 30]),   # 34x50, 41% obs, 1003 free
        (r"den101d.map", [10, 20, 30], [10, 20, 30]),   # 41x73, 55% obs, 1360 free
    ]

    # A sweep to break PP and force replanning to occur
    # show the utility of the voting replanning module.
    # Uses small highly-constrained maps (60-70% obstacles) where PP
    # is likely to fail on some instances. Fleet sizes kept small because
    # PP's 10-restart multi-restart is expensive on failures.
    HYPER_CONSTRAINED_CONFIGS = [
        (r"hrt003d.map", [10, 15], [10, 15]),            # 50x49, 70% obs, 747 free
        (r"lak103c.map", [10, 15], [10, 15]),            # 49x49, 65% obs, 850 free
        (r"lak108c.map", [10, 15], [10, 15]),            # 26x27, 60% obs, 279 free
    ]

    # --- PO CONFIG: partially-observable task requirements ---
    # Spec: partial_observability_experiment.md. LaCAM fixed, all 6 voting
    # methods + Murty's K-Best + perturbed JV, balanced suitability only,
    # and the 5 mask conditions (none / low-opt / low-pess / high-opt /
    # high-pess). The mask condition is NOT part of the schedule seed, so
    # all 5 conditions replay identical schedules (spec §5).
    # 2026-10-26 (user-approved expansion): 4 maps spanning the balanced
    # set's regimes - lak102d (small, 54% obstacles; continuity with every
    # prior PO pilot + ablation), arena (medium, 14% - the open-regime
    # medium), den204d (medium, 34% - the structured medium), lak307d
    # (large, 33%). HVBTA_PO=0 disables the PO condition axis (single
    # legacy condition, PO columns get sentinels) - used by e.g. the
    # hyper-constrained PP pilot, which stresses the planner.
    PO_EXPERIMENT = os.environ.get("HVBTA_PO", "1") == "1"
    # 8.1 score-noise experiment map set (2026-10-26 expansion): 3 maps
    # spanning the same regimes - lak102d (small; continuity with the noise
    # pilot + all ablations), den204d (structured medium), lak307d (large;
    # the cheapest large map, den001d/den901d add 2-3x cost for the same
    # call-size coverage). Same 5 sigma levels, 3 reps.
    NOISE_EXPERIMENT = os.environ.get("HVBTA_NOISE", "0") == "1"
    PO_SWEEP_CONFIGS = [
        (r"lak102d.map", [10, 20], [10, 20]),
        (r"arena.map",   [20, 30, 50], [20, 30, 50]),
        (r"den204d.map", [20, 30, 50], [20, 30, 50]),
        (r"lak307d.map", [50, 75, 100], [50, 75, 100]),
    ]
    NOISE_SWEEP_CONFIGS = [
        (r"lak102d.map", [10, 20], [10, 20]),
        (r"den204d.map", [20, 30, 50], [20, 30, 50]),
        (r"lak307d.map", [50, 75, 100], [50, 75, 100]),
    ]
    # Small-config override for verification runs (parallel==serial check).
    _sizes = os.environ.get("HVBTA_SIZES")
    if _sizes:
        _r = int(_sizes)
        PO_SWEEP_CONFIGS = [(r"lak102d.map", [_r], [_r])]
        NOISE_SWEEP_CONFIGS = PO_SWEEP_CONFIGS
    # HVBTA_CONFIG=hyper -> the three hyper-constrained maps (PP pilot).
    if os.environ.get("HVBTA_CONFIG") == "hyper":
        PO_SWEEP_CONFIGS = HYPER_CONSTRAINED_CONFIGS
    num_repetitions = int(os.environ.get("HVBTA_REPS", "3"))
    # Every run writes to a NEW directory - never an existing one.
    # Fresh dir => the resume scanner finds no completed rows => nothing
    # is skipped, every config re-runs (per agent instructions §0.4).
    PO_OUTPUT_DIRNAME = os.environ.get("HVBTA_OUTDIR", "pilot_po_P3_collision_v9_smoke_nudge_off2")
    # Phase 3v3: idle agents + step-aside goals + 1s planner budget +
    # 10-min run limit. The v2 dir holds the interrupted 18-hour partial.
    V.VOTING_ZERO_PAIR_FILTER = ABLATION_ZP
    V.VOTING_COVERAGE_THRESHOLD = ABLATION_TH
    from hvbta.partial_observability import PO_CONDITIONS, MASK_COUNTS

    # Report A collector (Phase 2 addition): record every positive balanced-
    # suitability entry from every matrix scored during Mask_Level=none runs,
    # bucketed per scenario, plus low-priority task tallies. The wrapper is
    # only active during none runs; sm_name / seed derivation are untouched.
    _suit_collector = {"active": False, "scenario": None,
                       "positives": {}, "unclipped": {},
                       "task_ids": {}, "low_task_ids": {}}

    def _collecting_sm(robots, tasks, *, map_size=None):
        if _suit_collector["active"]:
            scen = _suit_collector["scenario"]
            _unclip = []
            S.UNCLIPPED_COLLECTOR = _unclip
            try:
                result = S.evaluate_suitability_balanced(robots, tasks, map_size=map_size)
            finally:
                S.UNCLIPPED_COLLECTOR = None
            _m = np.asarray(result)
            _suit_collector["positives"].setdefault(scen, []).extend(_m[_m > 0].tolist())
            _suit_collector["unclipped"].setdefault(scen, []).extend(_unclip)
            seen = _suit_collector["task_ids"].setdefault(scen, set())
            low = _suit_collector["low_task_ids"].setdefault(scen, set())
            for _t in tasks:
                if _t.task_id not in seen:
                    seen.add(_t.task_id)
                    if getattr(_t, "priority_level", "medium") == "low":
                        low.add(_t.task_id)
        else:
            result = S.evaluate_suitability_balanced(robots, tasks, map_size=map_size)
        return result

    # Uncomment ONE of the following lines:
    # sweep_configs = OLD_SWEEP_CONFIGS        # finish interrupted first run
    # sweep_configs = CONSTRAINED_SWEEP_CONFIGS  # constrained maps test
    # sweep_configs = NEW_SWEEP_CONFIGS        # balanced pairing for clean results
    sweep_configs = HYPER_CONSTRAINED_CONFIGS
    if PO_EXPERIMENT:
        sweep_configs = PO_SWEEP_CONFIGS
    if NOISE_EXPERIMENT:
        sweep_configs = NOISE_SWEEP_CONFIGS
    # ============================================================

    map_dir = r"MAPF_benchmark_maps"
    add_tasks = True
    add_robots = True
    remove_robots = True
    tasks_to_add = 2
    robots_to_add = 1
    robots_to_remove = 1
    robot_generation_strict = True
    task_generation_strict = True
    remove_tasks = True
    tasks_to_remove = 1
    # PO runs fix LaCAM (spec §8); everything else keeps the configured planner.
    # HVBTA_PLANNER overrides the planner axis (e.g. the hyper-constrained
    # PP pilot forces pp to stress replanning under failures).
    PLANNER = [os.environ.get("HVBTA_PLANNER", "lacam" if PO_EXPERIMENT else "pp")]
    TARGET_EVENT_COUNT_K = 125
    TRANSIENT_FLOOR = 200

    # NOTE: every run writes to a fresh PO_OUTPUT_DIRNAME; never an existing
    # results directory (agent instructions 0.3). Always honor the
    # env-driven PO_OUTPUT_DIRNAME (HVBTA_OUTDIR) regardless of PO mode -
    # it must NEVER silently fall back to a legacy results directory.
    dir_path = os.path.join('hvbta', 'io', 'results', PO_OUTPUT_DIRNAME)
    po_diag_dir = os.path.join(dir_path, 'po_diagnostics')
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Resume support: an interrupted sweep re-launch continues from where it
    # stopped (rows are matched on the full group key plus rep count). Run
    # IDs are now the deterministic per-group IDs (make_run_id), so no
    # counter continuation is needed. Must run AFTER dir_path is chosen.
    resume_counts, resume_rows = _scan_completed_runs(dir_path)
    if resume_rows:
        print(f"RESUME: {resume_rows} completed run row(s) found in {dir_path} "
              f"across {len(resume_counts)} group key(s) - completed runs will be skipped.")
    os.makedirs(dir_path, exist_ok=True)

    for map_file_name, robot_sizes, task_sizes in sweep_configs:
        map_file = os.path.join(map_dir, map_file_name)
        grid = load_map(map_file)
        HYPOTENUSE = (len(grid) ** 2 + len(grid[0]) ** 2) ** 0.5
        dims = (len(grid), len(grid[0]))
        map_size = "Small" if HYPOTENUSE < 55 else "Medium" if HYPOTENUSE < 105 else "Large"
        obstacles = create_obstacle_list(grid)
        map_dict = {'dimension': dims, 'obstacles': obstacles}

        results_path = os.path.join(dir_path, f"{stamp}_simulation_{os.path.basename(map_file)}.csv")
        profiles_path = os.path.join(dir_path, f"{stamp}_profiles_{os.path.basename(map_file)}.csv")

        with open(results_path, mode="w", newline='') as file, \
             open(profiles_path, mode="w", newline='') as profile_file:
            writer = csv.writer(file)
            profiles_w = csv.writer(profile_file)

            writer.writerow(RESULT_ROW_HEADER)
            profiles_w.writerow(PROFILES_ROW_HEADER)

            for num_robots in robot_sizes:
                print(f"\n\n\nSTARTING SIMULATION FOR {num_robots} ROBOTS")
                for num_tasks in task_sizes:
                    print(f"\n\n\nSTARTING SIMULATION FOR {num_tasks} TASKS")
                    # Candidate pool scales with problem size.
                    nc = max(num_robots, num_tasks) * 6
                    for algo in PLANNER:
                        print(f"\n\n\nSTARTING SIMULATION WITH {algo} PLANNER")
                        workload = max(1.0, num_tasks / num_robots)
                        extender = 1.5 if workload > 10 else 1.2 if workload > 5 else 1.0
                        travel_floor = int(HYPOTENUSE * 1.5 * extender)

                        for scenario_name, regime in REGIMES.items():
                            print(f"\n\n\nSTARTING SCENARIO: {scenario_name}")
                            task_arrival_rate = regime["task_arrival_rate"]
                            task_cancellation_rate = regime["task_cancellation_rate"]
                            robot_arrival_rate = regime["robot_arrival_rate"]
                            robot_departure_rate = regime["robot_departure_rate"]

                            for sm in suitability_methods:
                                sm_name = func_name(sm)
                                for rep in range(num_repetitions):
                                    print(f"\n\n\nSTARTING SIMULATION REPETITION {rep+1}/{num_repetitions}")
                                    voting_outputs = []
                                    assignment_infos = []

                                    # Pre-compute the group's schedule seed BEFORE
                                    # generating entities (it depends only on the group
                                    # key, not on the drawn entities), then seed the
                                    # RNGs so per-rep entity generation is reproducible
                                    # across processes (agent instructions Step 1.1).
                                    # zlib.adler32 is stable across runs; hash() is not.
                                    seed_tuple = (
                                        os.path.basename(map_file), num_robots, num_tasks,
                                        nc, sm_name, rep, scenario_name
                                    )
                                    schedule_seed = zlib.adler32(
                                        repr(seed_tuple).encode()
                                    ) & 0xFFFFFFFF
                                    # Deterministic per-group Run ID (schema
                                    # unification 2026-10-27: replaces the
                                    # on-disk row counter, which depended on
                                    # resume state and made the profiles join
                                    # and cross-runner comparisons impossible).
                                    run_id = make_run_id(
                                        os.path.basename(map_file), num_robots,
                                        num_tasks, scenario_name, rep)
                                    _seed_rngs(_derived_seed(schedule_seed, "entities"))

                                    # Spawn rule: robots may not overlap each other, tasks may not overlap
                                    # each other, but a robot and a task may share a cell.
                                    robot_locs = set()
                                    robots = []
                                    # Strict-only generation (see todo item).
                                    for idx in range(num_robots):
                                        r = G.generate_random_robot_profile_strict(f"R{idx+1}", grid, robot_locs)
                                        robots.append(r)
                                        robot_locs.add(r.location)
                                    robot_profiles = [r.strict_profile_name for r in robots]

                                    task_locs = set()
                                    tasks = []
                                    for idx in range(num_tasks):
                                        t = G.generate_random_task_description_strict(f"T{idx+1}", grid, task_locs, [])
                                        tasks.append(t)
                                        task_locs.add(t.location)
                                    task_profiles = [t.strict_profile_name for t in tasks]

                                    # Phase 2: build-long → compute-horizon → truncate.
                                    # Build the schedule to a generous provisional length first.
                                    provisional_steps = max(TRANSIENT_FLOOR, int(HYPOTENUSE * 3))
                                    event_schedule = build_event_schedule(
                                        robots, tasks, grid, provisional_steps,
                                        add_tasks=add_tasks,
                                        add_robots=add_robots,
                                        remove_robots_flag=remove_robots,
                                        remove_tasks_flag=remove_tasks,
                                        task_arrival_rate=task_arrival_rate,
                                        robot_arrival_rate=robot_arrival_rate,
                                        robot_departure_rate=robot_departure_rate,
                                        task_cancellation_rate=task_cancellation_rate,
                                        tasks_to_add=tasks_to_add,
                                        robots_to_add=robots_to_add,
                                        robots_to_remove=robots_to_remove,
                                        tasks_to_remove=tasks_to_remove,
                                        robot_generation_strict=robot_generation_strict,
                                        task_generation_strict=task_generation_strict,
                                        seed=schedule_seed,
                                    )

                                    # Compute the real horizon as the max of three floors.
                                    event_horizon = horizon_for_event_count(
                                        event_schedule, TARGET_EVENT_COUNT_K
                                    )
                                    max_time_steps = max(
                                        TRANSIENT_FLOOR,
                                        event_horizon,
                                        travel_floor,
                                    )
                                    # Truncate the schedule to the real horizon.
                                    event_schedule = event_schedule[:max_time_steps]

                                    # Record which floor bound the horizon (for CSV provenance).
                                    horizon_binding_term = (
                                        "transient" if TRANSIENT_FLOOR >= event_horizon
                                        and TRANSIENT_FLOOR >= travel_floor
                                        else "events" if event_horizon >= travel_floor
                                        else "travel"
                                    )
                                    # Count scheduled events actually in the truncated schedule.
                                    scheduled_event_count = sum(
                                        len(e["add_tasks"]) + len(e["add_robots"])
                                        + len(e["remove_robot_ids"]) + len(e["remove_task_ids"])
                                        for e in event_schedule
                                    )

                                    if robot_generation_strict and task_generation_strict:
                                        profiles_w.writerow(build_profiles_row(
                                            run_id, os.path.basename(map_file),
                                            num_robots, num_tasks, sm_name,
                                            json.dumps(robot_profiles),
                                            json.dumps(task_profiles)))

                                    # Partial-observability conditions (spec §8). The
                                    # condition loop sits INSIDE the rep loop so all 5
                                    # conditions replay the identical schedule and masks
                                    # (mask level/policy are not in the schedule seed).
                                    # When PO_EXPERIMENT is off there is a single legacy
                                    # condition and the PO machinery stays disabled.
                                    # NOISE_EXPERIMENT (8.1): the condition axis
                                    # becomes sigma_obs in {0, 0.05, 0.1, 0.2,
                                    # 0.3} with masks off (none/none). sigma is
                                    # not in any seed, so every condition
                                    # replays the identical schedule and the
                                    # sigma=0 rows reproduce the no-noise
                                    # results exactly. (NOISE_EXPERIMENT is
                                    # read once at __main__ entry.)
                                    # HVBTA_NOISE_SIGMAS overrides the
                                    # sigma list (comma-separated) - used by
                                    # the anytime ablation's sigma=0-only
                                    # serial arm.
                                    NOISE_SIGMAS = [float(x) for x in
                                                    os.environ.get("HVBTA_NOISE_SIGMAS",
                                                                   "0,0.05,0.1,0.2,0.3").split(",")]
                                    if NOISE_EXPERIMENT:
                                        noise_conditions = (("none", "none", s) for s in NOISE_SIGMAS)
                                    elif PO_EXPERIMENT:
                                        noise_conditions = ((ml, mp, None) for ml, mp in PO_CONDITIONS)
                                    else:
                                        noise_conditions = ((None, None, None),)
                                    for mask_level, mask_policy, noise_sigma in noise_conditions:
                                        # Resume: skip the whole condition (initial matrix +
                                        # assignments + all method runs) when every method for
                                        # this rep is already on disk.
                                        if resume_counts and all(
                                            resume_counts.get(
                                                (os.path.basename(map_file), num_robots, num_tasks,
                                                 algo, scenario_name, sm_name, mask_level, mask_policy,
                                                 noise_sigma, _n), 0) > rep
                                            for _n in (voting_names + optimization_names + direct_names)
                                        ):
                                            continue
                                        po_cond = (
                                            PartialObservabilityState(schedule_seed, mask_level, mask_policy)
                                            if PO_EXPERIMENT else None
                                        )
                                        # Report A collector: active only for the true-attribute
                                        # (Mask_Level=none) conditions.
                                        _suit_collector["active"] = (mask_level == "none" or mask_level is None)
                                        _suit_collector["scenario"] = scenario_name
                                        sm_for_matrix = po_cond.masked_scorer(_collecting_sm) if po_cond is not None else _collecting_sm
                                        # 8.1: the initial allocation also sees the
                                        # noisy view (tick 0).
                                        if noise_sigma is not None:
                                            from hvbta.partial_observability import noisy_scorer as _noisy
                                            sm_for_matrix = _noisy(sm_for_matrix, noise_sigma,
                                                                   schedule_seed, [0])

                                        voting_outputs = []
                                        assignment_infos = []

                                        if getattr(sm, "_is_llm_batch", False):
                                            result = sm(robots, tasks)
                                            # LLM scorers return (matrix, parse_failed); unwrap so the
                                            # matrix is what flows through to llm_cache and downstream
                                            # indexing in reassign_robots_to_tasks_direct.
                                            suitability_matrix = result[0] if isinstance(result, tuple) else result
                                        else:
                                            # PO: the initial allocation is also an allocator
                                            # decision, so it scores the masked views too.
                                            suitability_matrix = S.calculate_suitability_matrix(robots, tasks, sm_for_matrix, HYPOTENUSE)

                                        # PO: report the initial Total Score /
                                        # Mean_Assignment_Suitability on TRUE attributes
                                        # (spec §7 - masked-view scores would inflate quality
                                        # under optimistic masking). Non-PO path unchanged.
                                        def _record_initial(run_id, method_name, out, score, length, per_agent):
                                            total, per_agent = _initial_quality_fields(
                                                po_cond, out, score, per_agent,
                                                robots, tasks, HYPOTENUSE)
                                            return _record_assignment(assignment_infos, run_id, method_name, sm_name, num_robots, num_tasks, nc, total, length, per_agent)

                                        # Direct LLM assignment is independent of the matrix; time it and
                                        # score its chosen pairs against the rule-based matrix for parity
                                        # with voting/optimization rows.
                                        direct_outputs = []
                                        for d_fn, d_name in zip(direct_methods, direct_names):
                                            d_start = time.perf_counter_ns()
                                            out, parse_failed = d_fn(robots, tasks)
                                            d_length = (time.perf_counter_ns() - d_start) / 1000.0
                                            if parse_failed:
                                                print(f"[{d_name}] parse failed - empty assignment fallback")
                                            pairs = out[0]
                                            per_agent_direct = [float(suitability_matrix[r][t]) for (r, t) in pairs] if not isinstance(suitability_matrix, tuple) else [0.0] * len(pairs)
                                            d_score = sum(per_agent_direct)
                                            initial_jains = _record_initial(run_id, d_name, out, d_score, d_length, per_agent_direct)
                                            direct_outputs.append((out, initial_jains))

                                        if suitability_all_zero(suitability_matrix):
                                            # 2.7: skip allocation (no random fallback) for
                                            # voting AND Murty's, in ALL runs - matching the
                                            # main-loop guard. Empty initial assignment; the
                                            # first replan tick handles whatever is feasible.
                                            empty_out = ([], list(range(len(robots))), list(range(len(tasks))))
                                            # Record a proper 13-item assignment row per method
                                            # so the CSV row prefix stays aligned.
                                            for _n in (voting_names + optimization_names + direct_names):
                                                _record_initial(run_id, _n, empty_out, 0.0, 0.0, [])
                                            outputs = [(empty_out, 0.0)] * len(all_methods)
                                        else:
                                            # voting - normal. Each method's initial assignment is
                                            # RNG-isolated (agent instructions Step 1.2) so one
                                            # method's candidate-generation draws can never change
                                            # another method's initial assignment.
                                            for method_fn, method_name in zip(voting_methods, voting_names):
                                                _seed_rngs(_derived_seed(schedule_seed, "initial", method_name))
                                                output, score, length, per_agent = V.assign_tasks_with_voting(
                                                    robots, tasks, suitability_matrix, nc, method_fn,
                                                    tie_seed=_derived_seed(schedule_seed, "ties", "initial", method_name))
                                                initial_jains = _record_initial(run_id, method_name, output, score, length, per_agent)
                                                voting_outputs.append((output, initial_jains))
                                            # optimization - normal
                                            opt_outputs = []
                                            for opt_fn, opt_name in zip(optimization_methods, optimization_names):
                                                _seed_rngs(_derived_seed(schedule_seed, "initial", opt_name))
                                                init_fn = opt_fn
                                                if opt_fn is O.perturbed_jv_task_allocation:
                                                    # 8.2: initial solve draws the baseline's
                                                    # own noise from the per-run seed at
                                                    # replan index "initial".
                                                    init_fn = O.perturbed_jv_factory(
                                                        _derived_seed(schedule_seed, "run", opt_name,
                                                                      "pjv", "initial"))
                                                out, score, length, per_agent = O.assign_tasks_with_method(init_fn, suitability_matrix)
                                                initial_jains = _record_initial(run_id, opt_name, out, score, length, per_agent)
                                                opt_outputs.append((out, initial_jains))
                                            outputs = voting_outputs + opt_outputs + direct_outputs

                                        for idx, (out_tuple, meth) in enumerate(zip(outputs, all_methods)):
                                            out, initial_jains = out_tuple
                                            # Resume: skip this individual run when its rep is
                                            # already on disk (count > rep, 0-indexed).
                                            if resume_counts.get(
                                                    (os.path.basename(map_file), num_robots, num_tasks,
                                                     algo, scenario_name, sm_name, mask_level, mask_policy,
                                                     method_pretty[meth]), 0) > rep:
                                                print(f"RESUME: skipping completed run "
                                                      f"{method_pretty[meth]} (map={os.path.basename(map_file)}, "
                                                      f"robots={num_robots}, scenario={scenario_name}, rep={rep}, "
                                                      f"mask={mask_level}/{mask_policy})")
                                                continue
                                            po_diag_path = (
                                                os.path.join(po_diag_dir, f"run{run_id}_"
                                                             f"{mask_level or 'nomask'}-{mask_policy or 'none'}"
                                                             f"-s{noise_sigma if noise_sigma is not None else 'na'}"
                                                             f"_{func_name(meth)}.json")
                                                if PO_EXPERIMENT else None
                                            )
                                            output_tuple = benchmark_simulation(
                                                output=out,
                                                robots=copy.deepcopy(robots),
                                                tasks=copy.deepcopy(tasks),
                                                num_candidates=nc,
                                                voting_method=meth,
                                                grid=grid,
                                                map_dict=map_dict,
                                                suitability_method=_collecting_sm,
                                                suitability_matrix=suitability_matrix,
                                                max_time_steps=max_time_steps,
                                                add_tasks=add_tasks,
                                                add_robots=add_robots,
                                                remove_robots=remove_robots,
                                                tasks_to_add=tasks_to_add,
                                                robots_to_add=robots_to_add,
                                                robots_to_remove=robots_to_remove,
                                                robot_generation_strict=robot_generation_strict,
                                                task_generation_strict=task_generation_strict,
                                                initial_jains_index=initial_jains,
                                                task_arrival_rate=task_arrival_rate,
                                                robot_arrival_rate=robot_arrival_rate,
                                                robot_departure_rate=robot_departure_rate,
                                                task_cancellation_rate=task_cancellation_rate,
                                                remove_tasks=remove_tasks,
                                                tasks_to_remove=tasks_to_remove,
                                                event_schedule=event_schedule,
                                                mask_level=mask_level,
                                                mask_policy=mask_policy,
                                                noise_sigma=noise_sigma,
                                                schedule_seed=schedule_seed,
                                                run_seed=_derived_seed(schedule_seed, "run", method_pretty[meth]),
                                                po_diag_path=po_diag_path,
                                                planner=algo,
                                                scenario=scenario_name,
                                                scheduled_event_count=scheduled_event_count,
                                                horizon_binding_term=horizon_binding_term,
                                            )
                                            row_prefix = assignment_infos[idx] if idx < len(assignment_infos) else [run_id, func_name(meth), sm_name, num_robots, num_tasks, nc, 0, 0, 0, 0, 0, 0, 0]
                                            # Inject the planner name at the CSV's 'Planner' column slot
                                            # (after Run ID / Method / Suitability Method, before Num Robots).
                                            row_prefix = [row_prefix[0], row_prefix[1], row_prefix[2], algo] + row_prefix[3:]
                                            # Config covariates + provenance, appended before the map_size suffix.
                                            # Row assembly is shared with the pool driver (build_result_row).
                                            writer.writerow(build_result_row(
                                                row_prefix, output_tuple, scenario_name, regime,
                                                mask_level if PO_EXPERIMENT else "",
                                                MASK_COUNTS.get(mask_level, -1) if PO_EXPERIMENT else -1,
                                                mask_policy if PO_EXPERIMENT else "",
                                                noise_sigma,
                                                map_size))
                                            # Flush per row so an interrupted sweep keeps every
                                            # completed run on disk for the resume scanner.
                                            file.flush()

    # ---- End-of-sweep dumps (outside the CSV writer's `with` block) ----
    # 2.5: per-rule top-2 tie frequencies across the whole sweep.
    _suit_collector["active"] = False
    with open(os.path.join(dir_path, "tie_stats.json"), "w", encoding="utf-8") as f:
        json.dump({rule: {"calls": c, "top2_ties": t} for rule, (c, t) in V.TIE_STATS.items()}, f, indent=2)

    # Report A: balanced-suitability distribution over all feasible pairs
    # scored during the none conditions, per scenario - both the clipped
    # (matrix) values and the raw pre-clip values (saturation analysis).
    _report_a = {}
    for scen, vals in _suit_collector["positives"].items():
        arr = np.asarray(vals)
        unc = np.asarray(_suit_collector["unclipped"].get(scen, []))
        hist, _ = np.histogram(arr, bins=20, range=(0.0, 1.0))
        pct = {p: float(np.percentile(arr, p)) for p in (1, 5, 25, 50, 75, 95)}
        n_tasks = len(_suit_collector["task_ids"].get(scen, ()))
        n_low = len(_suit_collector["low_task_ids"].get(scen, ()))
        _report_a[scen] = {
            "n_entries": int(arr.size),
            "histogram_20_bins": [int(h) for h in hist],
            "percentiles": pct,
            "share_above_0.95": float(np.mean(arr > 0.95)),
            "share_above_0.99": float(np.mean(arr > 0.99)),
            "unclipped": {
                "n_entries": int(unc.size),
                "share_above_1.0": float(np.mean(unc > 1.0)) if unc.size else 0.0,
                "percentiles": {p: float(np.percentile(unc, p)) for p in (1, 5, 25, 50, 75, 95)} if unc.size else {},
            },
            "distinct_tasks_seen": n_tasks,
            "low_priority_tasks": n_low,
            "low_priority_share": (n_low / n_tasks) if n_tasks else 0.0,
        }
    with open(os.path.join(dir_path, "suitability_distribution.json"), "w", encoding="utf-8") as f:
        json.dump(_report_a, f, indent=2)
    print("SWEEP COMPLETE: tie_stats.json and suitability_distribution.json written to", dir_path)
