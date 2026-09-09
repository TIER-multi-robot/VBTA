import time
import random
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
from hvbta.allocators.misc_assignment import (
    add_new_tasks, add_new_robots, remove_random_robots, remove_random_tasks,
    remove_task_by_id, remove_robot_by_id,
)
from hvbta.models import CapabilityProfile, TaskDescription


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
    # "calm2":       dict(task_arrival_rate=0.15, task_cancellation_rate=0.01,
    #                    robot_arrival_rate=0.01, robot_departure_rate=0.01),
    # "task_churn1": dict(task_arrival_rate=0.20, task_cancellation_rate=0.10,  # original kinda works already
    #                    robot_arrival_rate=0.01, robot_departure_rate=0.01),
    "task_churn": dict(task_arrival_rate=0.30, task_cancellation_rate=0.15,
                       robot_arrival_rate=0.01, robot_departure_rate=0.01),
    "fleet_churn":dict(task_arrival_rate=0.05, task_cancellation_rate=0.02,
                       robot_arrival_rate=0.30, robot_departure_rate=0.15),
    # "fleet_churn2":dict(task_arrival_rate=0.10, task_cancellation_rate=0.02,
    #                    robot_arrival_rate=0.10, robot_departure_rate=0.10),
    # "overload1":   dict(task_arrival_rate=0.45, task_cancellation_rate=0.02,
    #                    robot_arrival_rate=0.02, robot_departure_rate=0.04),
    "overload":   dict(task_arrival_rate=0.60, task_cancellation_rate=0.02,
                       robot_arrival_rate=0.03, robot_departure_rate=0.04),
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
                new_t = _with_seeded_random(
                    gen_seed,
                    G.generate_random_task_description_strict, tid, grid, set(), [],
                )
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
                    G.generate_random_robot_profile_strict, rid, grid, set(),
                )
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
        # MAPF planner selector ("cbs" | "pp" | "pibt" | "lacam"). Routes through
        # MAPF_wrapper.run_planner so the outer loop is untouched. Default "cbs"
        # during the refactor so partial progress never breaks a run.
        planner: str = "cbs",
        # Phase 4 CSV-provenance fields, passed through from the sweep so the
        # row is self-describing. These are constant per run, not computed
        # inside the sim — they're just threaded to the return tuple.
        scenario: str = "",
        scheduled_event_count: int = 0,
        horizon_binding_term: str = ""):
    print(f"SUITABILITY METHOD: {suitability_method}")

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

    # apply initial assignment
    assigned_pairs = output[0]
    for robot_idx, task_idx in assigned_pairs:
        r = robots[robot_idx]
        t = tasks[task_idx]
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

    avg_path_length, _ = run_planner(planner, robots, start_positions, goal_positions, map_dict)

    # create the initial state
    previous_active, previous_goals, previous_unassigned_robots, previous_unassigned_tasks = state_check(robots, unassigned_robots, unassigned_tasks)
    events = {"new_tasks": 0, "new_robots": 0, "completed_tasks": 0}
    idle_steps = {r.robot_id: 0 for r in robots}
    robot_max_id = len(robots) + 1
    task_max_id = len(tasks) + 1

    for time_step in range(max_time_steps):
        # update unnasigned robots and tasks
        unassigned_robots = [r.robot_id for r in robots if not r.assigned]
        unassigned_tasks = [t.task_id for t in tasks if not t.assigned]

        # simulate timestep
        completed_this_step, _unassigned_count, total_reward, total_success = simulate_time_step(
            robots, tasks, unassigned_robots, unassigned_tasks,
            suitability_method, occupied_positions, start_positions,
            goal_positions, 1.0, total_reward, total_success,
        )
        # Per-step throughput signal (used for mean + variance at end).
        completed_per_step.append(completed_this_step)
        # Phase 0: log pool/fleet size every timestep, right after the step.
        task_pool_size_per_step.append(len(tasks))
        fleet_size_per_step.append(len(robots))

        # Record the first moment the task pool hits 0 — useful METRIC even
        # when we don't terminate (the pool may refill from future arrivals).
        # Captured whether or not the sim breaks here.
        if len(tasks) == 0 and all_tasks_completed_at_step is None:
            all_tasks_completed_at_step = time_step + 1

        # Termination: pool empty AND no future task arrivals in the schedule.
        # This prevents the sim from ending lifelong runs early when future
        # arrivals would refill the pool — the confound the event schedule
        # exists to remove. Under overload the pool never empties, so this
        # never fires and the run goes to max_time_steps (correct).
        no_future_tasks = (
            event_schedule is None
            or all(len(e["add_tasks"]) == 0 for e in event_schedule[time_step + 1:])
        )
        if len(tasks) == 0 and no_future_tasks:
            print(f"All tasks completed at time step {time_step + 1} (no future arrivals).")
            total_time_steps = time_step + 1
            break

        # update events with any completed tasks this timestep
        events["completed_tasks"] += completed_this_step
        # set the replan_cbs flag
        should_replan_cbs = completed_this_step > 0

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
        # completions + |task pool delta| + |robot pool delta| approximates
        # total activity this step (a heuristic; exactness is not required).
        events_this_step = (
            completed_this_step
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
            should_replan_cbs = True

        if should_replan:
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
                suitability_source = suitability_method

            # assign with voting
            if voting_method in voting_methods:
                # Anytime pressure: any urgent unassigned task or a new-robot
                # event this tick triggers a tight budget so voting bails on
                # the first good-enough candidate.
                urgent_pressure = (percentage_urgent > 0.0) or (robot_pressure > 0.0) # NOTE: here I can take out robot pressure
                voting_budget_ms = 20.0 if urgent_pressure else None
                voting_score_threshold = 0.7 if urgent_pressure else None

                # Adaptive candidate count: recent churn shrinks the pool so
                # voting responds fast under stress and explores broadly when
                # calm. Floors at MIN_CANDIDATES.
                recent_event_count = sum(recent_events_window)
                effective_candidates = max(MIN_CANDIDATES, num_candidates // (1 + recent_event_count))

                # Hall of fame: hand the surviving cached candidates to the pool
                # and collect this round's top-K back out.
                top_k_out_ids = []

                print(f"REASSIGNING WITH VOTING METHOD: {voting_method_name}"
                      f"{f' (URGENT: pct_urgent={percentage_urgent:.0%}, robot_pressure={int(robot_pressure)}, 20ms budget, 0.7 threshold)' if urgent_pressure else ''}"
                      f"{f' [warm-start seed: {len(previous_best_id_pairs)} pairs]' if previous_best_id_pairs else ''}"
                      f" [candidates: {effective_candidates}/{num_candidates}, recent_events={recent_event_count}]"
                      f"{f' [hall-of-fame: {len(candidate_cache)} cached]' if candidate_cache else ''}")
                total_reassignments += 1
                _, unassigned_robots, unassigned_tasks, reassign_score, reassign_length, reassign_per_agent_scores = V.reassign_robots_to_tasks(
                    robots, tasks, effective_candidates, voting_method, suitability_source,
                    unassigned_robots, unassigned_tasks, start_positions, goal_positions, HYPOTENUSE,
                    time_budget_ms=voting_budget_ms,
                    score_threshold=voting_score_threshold,
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
            elif voting_method in optimization_methods:
                print(f"REASSIGNING WITH OPTIMIZATION METHOD: {optimization_method_name}")
                total_reassignments += 1
                _, unassigned_robots, unassigned_tasks, reassign_score, reassign_length, reassign_per_agent_scores = O.reassign_robots_to_tasks_with_method(
                    robots, tasks, num_candidates, voting_method, suitability_source,
                    unassigned_robots, unassigned_tasks, voting_method, start_positions, goal_positions, HYPOTENUSE,
                )
                if reassign_per_agent_scores:
                    reassignment_jains_indices.append(calculate_jains_index(reassign_per_agent_scores))
                    reassignment_fairness_metrics.append(compute_all_fairness_metrics(reassign_per_agent_scores))

            # assign with direct LLM (bypass the matrix)
            elif getattr(voting_method, "_is_llm_direct", False):
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

            # update location dicts
            for robot in robots:
                if robot.assigned and robot.current_task:
                    start_positions[robot.robot_id] = tuple(robot.location)
                    goal_positions[robot.robot_id] = tuple(robot.current_task.location)
                else:
                    start_positions.pop(robot.robot_id, None)
                    goal_positions.pop(robot.robot_id, None)

        # Record any task that is currently assigned - this is the "attempted" set.
        # We do this every timestep (cheap) so cancellations of never-attempted tasks
        # do not incorrectly count them.
        for t in tasks:
            if t.assigned:
                tasks_ever_assigned.add(t.task_id)

        # rerun cbs if needed, update state, and clear events
        if should_replan_cbs and start_positions and goal_positions:
            print("\nRERUNNING PLANNER\n")
            new_len, planner_okay = run_planner(planner, robots, start_positions, goal_positions, map_dict)
            avg_path_length = new_len or avg_path_length
            if planner_okay and pending_event_steps:
                # Adaptation latency: elapsed timesteps from each event to this
                # successful resolution. Drain the queue.
                for ev_step in pending_event_steps:
                    adaptation_latencies.append(time_step - ev_step)
                pending_event_steps.clear()
            previous_active, previous_goals, previous_unassigned_robots, previous_unassigned_tasks = state_check(robots, unassigned_robots, unassigned_tasks)
            events = {k: 0 for k in events}

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
        planner: str = "cbs",
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
        planner=planner,
        scenario=scenario,
        scheduled_event_count=scheduled_event_count,
        horizon_binding_term=horizon_binding_term,
    )
    execution_time = time.perf_counter_ns() - start_time
    print(f"Simulation completed in {execution_time:.5f} nanoseconds.")
    return output_tuple + (execution_time,)


def _record_assignment(assignment_infos, run_id, method_name, sm_name, num_robots,
                       num_tasks, nc, score, planning_time_us, per_agent_scores):
    """Compute initial fairness metrics, derived score fields, and append a row."""
    metrics = compute_all_fairness_metrics(per_agent_scores)
    assigned_count = len(per_agent_scores) if per_agent_scores else 0
    # arithmetic mean suitability across assigned pairs (score / assigned_count).
    mean_assignment_suitability = (score / assigned_count) if assigned_count > 0 else 0.0
    assignment_infos.append(
        [run_id, method_name, sm_name, num_robots, num_tasks, nc, score,
         mean_assignment_suitability, planning_time_us] + list(metrics)
    )
    return metrics[0]  # initial jains


if __name__ == "__main__":
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
        O.cbba_task_allocation,
        O.ssia_task_allocation,
        O.ilp_task_allocation,
        O.jv_task_allocation,
    ]
    optimization_names = [func_name(f) for f in optimization_methods]
    # Direct-assignment methods: the LLM returns (robot_id, task_id) pairs straight up,
    # bypassing the suitability-matrix + voting/optimization stack. Slot here so it
    # benchmarks alongside the others.
    direct_methods = [
        # S.bypass_suitability_from_names_with_llm,
    ]
    direct_names = [func_name(f) for f in direct_methods]
    all_methods = voting_methods + optimization_methods + direct_methods
    suitability_methods = [
        S.evaluate_suitability_balanced,
        S.evaluate_suitability_loose,
        S.evaluate_suitability_strict,
        # S.evaluate_suitability_from_names_with_llm,
    ]

    small_maps = [
        r"den201d.map", r"den202d.map", r"den404d.map", r"lak101d.map",
        r"lak102d.map", r"lak105d.map", r"lak107d.map", r"lak108d.map",
    ]
    medium_maps = [
        r"arena.map", r"den009d.map", r"den101d.map", r"den204d.map",
        r"den207d.map", r"den403d.map", r"den405d.map", r"den407d.map",
        r"den408d.map", r"hrt002d.map", r"isound1.map", r"lak103d.map",
        r"lak104d.map",
    ]
    large_maps = [
        r"den001d.map", r"den020d.map", r"den203d.map", r"den206d.map",
        r"den308d.map", r"den312d.map", r"den900d.map", r"den901d.map",
        r"den998d.map", r"hrt001d.map", r"lak106d.map", r"lak203d.map",
        r"lak307d.map", r"ost002d.map",
    ]
    map_paths = (
        random.sample(small_maps, 3)
        + random.sample(medium_maps, 3)
        + random.sample(large_maps, 3)
    )
    # map_paths = [r"arena.map"]

    robot_sizes = [10, 20, 30, 40, 50, 100]
    task_sizes = [10, 20, 30, 40, 50, 100]
    # Candidate pool sizes scale with the problem: nc = max(robot, task) * 5.
    # Computed inside the sweep loop per (num_robots, num_tasks) pair.
    candidate_sizes = None  # placeholder; computed dynamically below
    Run_ID = 1
    num_repetitions = 10
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
    # MAPF planner: using PP during development (fastest, 100% on solvable
    # instances). Flip to ["cbs", "pp", "pibt"] for the full sweep.
    PLANNER = ["pp"]
    # Phase 2: target event count K for the event-exposure-based horizon.
    # Start at 50; raise if per-event metrics look noisy.
    TARGET_EVENT_COUNT_K = 125
    # Phase 2: transient floor — minimum steps before any regime's run can
    # end. Tune after Phase 0 data shows where the transient flattens.
    TRANSIENT_FLOOR = 200
    map_dir = r"MAPF_benchmark_maps"

    dir_path = os.path.join('hvbta', 'io', 'results')
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(dir_path, exist_ok=True)
    full_paths = [os.path.join(map_dir, m) for m in map_paths]

    for map_file in full_paths:
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

            writer.writerow([
                # Row identity
                'Run ID', 'Method', 'Suitability Method', 'Planner',
                'Num Robots', 'Num Tasks', 'Num Candidates',
                # Initial allocation quality
                'Total Score', 'Mean_Assignment_Suitability', 'Planning_Time_us',
                # Initial fairness (trimmed to 4 canonical values)
                'Init_Jains_Index', 'Init_Gini', 'Init_Median', 'Init_IQR',
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
                # Full-schedule denominators, constant across methods in a (rep, sm) group.
                'Full_Schedule_Tasks_Spawned', 'Full_Schedule_Robots_Max',
                # Urgency + pressure bucketing keys.
                'Percentage_Urgent_Mean', 'Percentage_Urgent_Max', 'Robot_Pressure_Fraction',
                # Reassignment fairness (trimmed to 3 canonical values; Jains already exposed above)
                'Avg_Reass_Gini', 'Avg_Reass_Median', 'Avg_Reass_IQR',
                # Empirical regime classification (computed per run from pool-size log)
                'Mean_Task_Pool_Size', 'Pool_Size_Trend',
                # Horizon provenance (so the horizon choice is auditable)
                'Scheduled_Event_Count', 'Horizon_Binding_Term',
                # Overhead
                'Execution Time',
                # Config covariates (constant per run)
                'Scenario', 'Task_Arrival_Rate', 'Task_Cancellation_Rate',
                'Robot_Arrival_Rate', 'Robot_Departure_Rate',
                # Map size
                'Map Size',
            ])
            profiles_w.writerow([
                'Run_ID', 'Map', 'Num Robots', 'Num Tasks',
                'Suitability Method', 'RobotProfiles', 'TaskProfiles',
            ])

            for num_robots in robot_sizes:
                print(f"\n\n\nSTARTING SIMULATION FOR {num_robots} ROBOTS")
                for num_tasks in task_sizes:
                    print(f"\n\n\nSTARTING SIMULATION FOR {num_tasks} TASKS")
                    # Candidate pool scales with problem size.
                    nc = max(num_robots, num_tasks) * 5
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

                                    # Pre-build one deterministic event schedule per
                                    # (map, num_robots, num_tasks, nc, sm, rep, scenario).
                                    # Every method + planner in this group replays the SAME schedule
                                    # so their comparisons are fair. Uses zlib.adler32 (stable across
                                    # runs) instead of hash() (which is per-process-randomized in
                                    # Python 3 unless PYTHONHASHSEED is fixed).
                                    import zlib
                                    seed_tuple = (
                                        os.path.basename(map_file), num_robots, num_tasks,
                                        nc, sm_name, rep, scenario_name
                                    )
                                    schedule_seed = zlib.adler32(
                                        repr(seed_tuple).encode()
                                    ) & 0xFFFFFFFF

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

                                    if getattr(sm, "_is_llm_batch", False):
                                        result = sm(robots, tasks)
                                        # LLM scorers return (matrix, parse_failed); unwrap so the
                                        # matrix is what flows through to llm_cache and downstream
                                        # indexing in reassign_robots_to_tasks_direct.
                                        suitability_matrix = result[0] if isinstance(result, tuple) else result
                                    else:
                                        suitability_matrix = S.calculate_suitability_matrix(robots, tasks, sm, HYPOTENUSE)

                                    if robot_generation_strict and task_generation_strict:
                                        profiles_w.writerow([
                                            Run_ID, os.path.basename(map_file),
                                            num_robots, num_tasks, sm_name,
                                            json.dumps(robot_profiles), json.dumps(task_profiles),
                                        ])

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
                                        initial_jains = _record_assignment(assignment_infos, Run_ID, d_name, sm_name, num_robots, num_tasks, nc, d_score, d_length, per_agent_direct)
                                        direct_outputs.append((out, initial_jains))

                                    if suitability_all_zero(suitability_matrix):
                                        # voting - random fallback
                                        for method_fn, method_name in zip(voting_methods, voting_names):
                                            output, score, length, per_agent = V.assign_tasks_randomly(robots, tasks, suitability_matrix, nc)
                                            initial_jains = _record_assignment(assignment_infos, Run_ID, method_name, sm_name, num_robots, num_tasks, nc, score, length, per_agent)
                                            voting_outputs.append((output, initial_jains))
                                        # optimization - random fallback
                                        opt_outputs = []
                                        for opt_fn, opt_name in zip(optimization_methods, optimization_names):
                                            out, score, length, per_agent = O.assign_tasks_with_method_randomly(opt_fn, suitability_matrix, nc)
                                            initial_jains = _record_assignment(assignment_infos, Run_ID, opt_name, sm_name, num_robots, num_tasks, nc, score, length, per_agent)
                                            opt_outputs.append((out, initial_jains))
                                        outputs = voting_outputs + opt_outputs + direct_outputs
                                    else:
                                        # voting - normal
                                        for method_fn, method_name in zip(voting_methods, voting_names):
                                            output, score, length, per_agent = V.assign_tasks_with_voting(robots, tasks, suitability_matrix, nc, method_fn)
                                            initial_jains = _record_assignment(assignment_infos, Run_ID, method_name, sm_name, num_robots, num_tasks, nc, score, length, per_agent)
                                            voting_outputs.append((output, initial_jains))
                                        # optimization - normal
                                        opt_outputs = []
                                        for opt_fn, opt_name in zip(optimization_methods, optimization_names):
                                            out, score, length, per_agent = O.assign_tasks_with_method(opt_fn, suitability_matrix)
                                            initial_jains = _record_assignment(assignment_infos, Run_ID, opt_name, sm_name, num_robots, num_tasks, nc, score, length, per_agent)
                                            opt_outputs.append((out, initial_jains))
                                        outputs = voting_outputs + opt_outputs + direct_outputs

                                    for idx, (out_tuple, meth) in enumerate(zip(outputs, all_methods)):
                                        out, initial_jains = out_tuple
                                        output_tuple = benchmark_simulation(
                                            output=out, 
                                            robots=copy.deepcopy(robots), 
                                            tasks=copy.deepcopy(tasks),
                                            num_candidates=nc, 
                                            voting_method=meth, 
                                            grid=grid, 
                                            map_dict=map_dict, 
                                            suitability_method=sm, 
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
                                            planner=algo,
                                            scenario=scenario_name,
                                            scheduled_event_count=scheduled_event_count,
                                            horizon_binding_term=horizon_binding_term,
                                        )
                                        row_prefix = assignment_infos[idx] if idx < len(assignment_infos) else [Run_ID, func_name(meth), sm_name, num_robots, num_tasks, nc, 0, 0, 0, 0]
                                        # Inject the planner name at the CSV's 'Planner' column slot
                                        # (after Run ID / Method / Suitability Method, before Num Robots).
                                        row_prefix = [row_prefix[0], row_prefix[1], row_prefix[2], algo] + row_prefix[3:]
                                        # Config covariates + provenance, appended before the map_size suffix.
                                        config_suffix = [
                                            scenario_name,
                                            task_arrival_rate, task_cancellation_rate,
                                            robot_arrival_rate, robot_departure_rate,
                                        ]
                                        writer.writerow(row_prefix + list(output_tuple) + config_suffix + [map_size])
                                    Run_ID += 1
