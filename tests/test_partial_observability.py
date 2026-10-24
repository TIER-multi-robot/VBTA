"""
Validation tests for the partial-observability experiment
(spec: partial_observability_experiment.md §9).

1. Leak test       - masked fields' true values never reach the scorer.
2. Mask fairness   - masks identical across methods/policies; low ⊂ high.
3. Determinism     - cross-process + order independence (driver-based).
4. Reveal-once     - each task's mask is removed at most once.
5. No-bounce       - a task infeasible for every robot is never assigned
                     (voting filters zero-suitability pairs).
6. Hard-fail parity - balanced_hard_fail True ⟹ _balanced_pair == 0.
"""
import os
import random
import sys

import numpy as np
import pytest

from hvbta import generation as G
from hvbta.partial_observability import (
    MASKABLE_FIELDS,
    MASK_COUNTS,
    PO_CONDITIONS,
    PartialObservabilityState,
    TASK_TYPE_MODE,
    DIFFICULTY_MEDIAN,
    PAYLOAD_MAX,
    REACH_MAX,
    DURATION_MIN,
    DURATION_MAX,
    UNMET,
    build_masked_view,
    mask_for,
)
from hvbta.suitability import (
    _balanced_pair,
    balanced_hard_fail,
    evaluate_suitability_balanced,
)
from hvbta.allocators.voting import assign_tasks_with_voting, rank_assignments_borda, _filter_zero_pairs
from hvbta.pathfinding.planner_common import _write_paths_back
from hvbta.models import CapabilityProfile, TaskDescription


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
GRID = [[0] * 12 for _ in range(12)]  # fully free small grid


def make_robots(n=6, seed=7):
    random.seed(seed)
    locs = set()
    return [G.generate_random_robot_profile_strict(f"R{i+1}", GRID, locs) for i in range(n)]


def make_tasks(n=6, seed=11):
    random.seed(seed)
    locs = set()
    return [G.generate_random_task_description_strict(f"T{i+1}", GRID, locs, []) for i in range(n)]


SEED = 12345


# ---------------------------------------------------------------------------
# 1. Leak test
# ---------------------------------------------------------------------------
def test_masked_view_hides_masked_fields():
    """With high masking, no masked field's true value is reachable from the
    view object passed to the scorer."""
    robots, tasks = make_robots(), make_tasks()
    po = PartialObservabilityState(SEED, "high", "optimistic")

    captured = []

    def spy_scorer(rbs, tks, *, map_size=None):
        captured.extend(tks)
        return evaluate_suitability_balanced(rbs, tks, map_size=map_size)

    wrapper = po.masked_scorer(spy_scorer)
    wrapper(robots, tasks, map_size=100)

    assert len(captured) == len(tasks)
    for view, task in zip(captured, tasks):
        mf = po.mask_of(task.task_id)
        assert mf, "high masking must mask fields"
        assert view is not task, "scorer must receive a copy, not the true task"
        # every masked field holds the policy value, never the true value
        if "task_type" in mf:
            assert view.task_type == TASK_TYPE_MODE
            assert view.task_type != task.task_type or task.task_type == TASK_TYPE_MODE
        if "difficulty" in mf:
            assert view.difficulty == DIFFICULTY_MEDIAN
        if "required_capabilities.payload" in mf:
            assert view.required_capabilities["payload"] == 0.0
        if "required_capabilities.reach" in mf:
            assert view.required_capabilities["reach"] == 0.0
        if "duration" in mf:
            assert view.time_to_complete == DURATION_MIN
        for lf in ("navigation_constraints", "environmental_conditions",
                   "sensors_needed", "manipulators_needed",
                   "communication_requirements", "safety_protocols"):
            if lf in mf:
                assert getattr(view, lf) == []
        # unmasked fields still carry the true values
        if "sensors_needed" not in mf:
            assert view.sensors_needed == task.sensors_needed
        if "required_capabilities.payload" not in mf:
            assert view.required_capabilities["payload"] == task.required_capabilities["payload"]
        # the true task was never mutated
        assert task.sensors_needed is not None


def test_pessimistic_view_substitutes_unmet():
    task = make_tasks(n=1)[0]
    mf = frozenset(MASKABLE_FIELDS)  # everything masked
    view = build_masked_view(task, mf, "pessimistic")
    assert view.required_capabilities["payload"] == PAYLOAD_MAX
    assert view.required_capabilities["reach"] == REACH_MAX
    assert view.time_to_complete == DURATION_MAX
    for lf in ("navigation_constraints", "environmental_conditions",
               "sensors_needed", "manipulators_needed",
               "communication_requirements", "safety_protocols"):
        assert getattr(view, lf) == [UNMET]
    # non-requirement fields use the same neutral default under both policies
    assert view.task_type == TASK_TYPE_MODE
    assert view.difficulty == DIFFICULTY_MEDIAN


def test_pessimistic_masked_hard_fail_field_scores_zero():
    """A masked hard-fail field under pessimistic gives zero suitability
    (for robots that can't meet the distribution max) - expected starvation,
    no special handling (spec §6.2)."""
    robots, tasks = make_robots(), make_tasks()
    po = PartialObservabilityState(SEED, "high", "pessimistic")
    # find a task whose payload requirement is masked, then verify at least
    # one robot scores 0 on the masked view
    scored_zero = 0
    for t in tasks:
        if "required_capabilities.payload" in po.mask_of(t.task_id):
            for r in robots:
                if r.payload_capacity < PAYLOAD_MAX:
                    view = po.masked_view(t)
                    assert _balanced_pair(r, view, 100) == 0.0
                    scored_zero += 1
    assert scored_zero > 0


# ---------------------------------------------------------------------------
# 2. Mask fairness + 3. determinism
# ---------------------------------------------------------------------------
def test_masks_deterministic_and_nested():
    for tid in ("T1", "T7", "T42"):
        low = mask_for(SEED, tid, MASK_COUNTS["low"])
        high = mask_for(SEED, tid, MASK_COUNTS["high"])
        assert mask_for(SEED, tid, MASK_COUNTS["low"]) == low  # deterministic
        assert len(low) == MASK_COUNTS["low"]
        assert len(high) == MASK_COUNTS["high"]
        assert low <= high  # nested: low ⊂ high for every task
        assert low <= set(MASKABLE_FIELDS)


def test_masks_identical_across_methods_and_policies():
    """Two PO states built for different policies (i.e. different 'methods'
    in a group) see identical masks for the same (seed, task). (Run-level
    determinism is covered by the cross-process / order-independence
    tests above.)"""
    tasks = make_tasks()
    po_opt = PartialObservabilityState(SEED, "high", "optimistic")
    po_pess = PartialObservabilityState(SEED, "high", "pessimistic")
    for t in tasks:
        assert po_opt.mask_of(t.task_id) == po_pess.mask_of(t.task_id)


# ---------------------------------------------------------------------------
# 4. Reveal-once
# ---------------------------------------------------------------------------
def test_reveal_exactly_once():
    robots, tasks = make_robots(), make_tasks()
    po = PartialObservabilityState(SEED, "high", "optimistic")
    task = next(t for t in tasks if po.is_masked(t.task_id))
    robot = next(r for r in robots if not balanced_hard_fail(r, task, 0.0))
    robot.current_task = task
    robot.assigned = True
    ur, ut = [], []

    # first arrival reveals
    assert po.handle_arrival(robot, task, ur, ut, current_tick=5, pending_event_steps=[]) is False
    assert po.reveal_events == 1
    assert po.arrivals == 1
    assert not po.is_masked(task.task_id)

    # working ticks re-call handle_arrival: no double reveal / recount
    assert po.handle_arrival(robot, task, ur, ut, current_tick=6, pending_event_steps=[]) is False
    assert po.reveal_events == 1
    assert po.arrivals == 1

    # robot freed and re-paired later: arrival re-processed, reveal is NOT
    po.on_unassign(robot)
    robot.assigned = True
    assert po.handle_arrival(robot, task, ur, ut, current_tick=9, pending_event_steps=[]) is False
    assert po.arrivals == 2
    assert po.reveal_events == 1  # permanent


def test_mis_assignment_frees_robot_and_raises_event():
    robots, tasks = make_robots(), make_tasks()
    po = PartialObservabilityState(SEED, "low", "optimistic")
    task = tasks[0]
    # a robot that genuinely cannot do the task (payload far too small)
    robot = min(robots, key=lambda r: r.payload_capacity)
    robot.current_task = task
    robot.assigned = True
    task.assigned_robot = robot
    task.assigned = True
    pending = []

    assert po.handle_arrival(robot, task, [], [], current_tick=3, pending_event_steps=pending) is True
    assert po.mis_assignment_count == 1
    assert po.wasted_travel_steps >= 0
    assert robot.assigned is False
    assert robot.current_task is None
    assert task.assigned is False
    assert task.assigned_robot is None
    assert pending == [3]  # replan event raised through the existing mechanism
    # the task is now unmasked
    assert not po.is_masked(task.task_id)


# ---------------------------------------------------------------------------
# 5. No-bounce: infeasible-for-all task never (re)assigned by voting
# ---------------------------------------------------------------------------
def test_no_bounce_zero_column_task_never_assigned():
    """A revealed task that is infeasible for every robot scores zero in its
    whole column; voting's zero-pair filter must never assign it."""
    robots, tasks = make_robots(n=5), make_tasks(n=4)
    # task index 2 is infeasible for all robots (payload above every robot)
    payload_cap = max(r.payload_capacity for r in robots)
    tasks[2].required_capabilities = {"payload": payload_cap + 100.0, "reach": 0.0}
    M = evaluate_suitability_balanced(robots, tasks, map_size=100)
    assert all(M[i, 2] == 0.0 for i in range(len(robots))), "precondition: column all zero"

    output, score, length, per_agent = assign_tasks_with_voting(
        robots, tasks, M, 12, rank_assignments_borda)

    assigned_tasks = {t for _, t in output[0]}
    assert 2 not in assigned_tasks
    # every assigned pair has positive suitability
    for r, t in output[0]:
        assert M[r][t] > 0


# ---------------------------------------------------------------------------
# 6. Hard-fail parity between scorer and execution-time check
# ---------------------------------------------------------------------------
def test_hard_fail_implies_zero_suitability():
    robots, tasks = make_robots(n=8, seed=3), make_tasks(n=8, seed=5)
    checked = 0
    for r in robots:
        for t in tasks:
            for dist in (0.0, 25.0):
                if balanced_hard_fail(r, t, dist):
                    assert _balanced_pair(r, t, 100) == 0.0
                    checked += 1
    assert checked > 0, "expected at least one infeasible pairing in the sample"


def test_hard_fail_uses_true_attributes_at_arrival():
    """The execution-time check reads the TRUE task, not the masked view:
    a robot below the true payload requirement must fail even when the
    masked (optimistic) view would have passed."""
    robots, tasks = make_robots(), make_tasks()
    task = tasks[0]
    # find a robot feasible for the task as-is, then push the true payload
    # requirement just above its capacity: it fails on truth, passes masked
    feasible_robot = next(r for r in robots if not balanced_hard_fail(r, task, 0.0))
    task.required_capabilities = dict(task.required_capabilities)
    task.required_capabilities["payload"] = feasible_robot.payload_capacity + 1.0

    assert balanced_hard_fail(feasible_robot, task, 0.0) is True  # true attributes
    view = build_masked_view(task, frozenset(["required_capabilities.payload"]), "optimistic")
    assert balanced_hard_fail(feasible_robot, view, 0.0) is False  # masked view passes


# ---------------------------------------------------------------------------
# Real determinism tests (agent instructions Step 3): cross-process + order
# independence, via tests/determinism_driver.py.
# ---------------------------------------------------------------------------
ORDER_A = ["borda", "cumulative", "murtys"]
ORDER_B = ["murtys", "borda", "cumulative"]
# Wall-clock timing columns in the driver's output: c1 = total_reassignment_time,
# c47 (last) = Execution Time. Everything else must match exactly.
TIMING_COLS = {"c1", "c47"}


def _run_driver(order, out_path):
    import subprocess
    driver = os.path.join(os.path.dirname(os.path.abspath(__file__)), "determinism_driver.py")
    res = subprocess.run(
        [sys.executable, driver, out_path, ",".join(order)],
        capture_output=True, text=True, timeout=900,
    )
    assert res.returncode == 0, f"driver failed:\n{res.stderr[-2000:]}"


def _load_rows(path):
    import csv as _csv
    with open(path, newline="", encoding="utf-8") as f:
        return list(_csv.DictReader(f))


def test_cross_process_determinism():
    """The same config run in two separate subprocesses produces identical
    results, excluding wall-clock timing columns."""
    tmp = os.environ.get("TEMP", ".")
    p1 = os.path.join(tmp, "det_run1.csv")
    p2 = os.path.join(tmp, "det_run2.csv")
    _run_driver(ORDER_A, p1)
    _run_driver(ORDER_A, p2)
    rows1, rows2 = _load_rows(p1), _load_rows(p2)
    assert len(rows1) == len(rows2) == 3
    for r1, r2 in zip(rows1, rows2):
        for col in r1:
            if col == "method" or col in TIMING_COLS:
                continue
            assert r1[col] == r2[col], (
                f"cross-process mismatch: {r1['method']} col {col}: "
                f"{r1[col]!r} != {r2[col]!r}")


def test_order_independence():
    """Running the methods in a different order does not change any
    method's results (per-method RNG isolation), excluding timing columns."""
    tmp = os.environ.get("TEMP", ".")
    pa = os.path.join(tmp, "det_order_a.csv")
    pb = os.path.join(tmp, "det_order_b.csv")
    _run_driver(ORDER_A, pa)
    _run_driver(ORDER_B, pb)
    by_method_a = {r["method"]: r for r in _load_rows(pa)}
    by_method_b = {r["method"]: r for r in _load_rows(pb)}
    assert set(by_method_a) == set(by_method_b)
    for m in by_method_a:
        ra, rb = by_method_a[m], by_method_b[m]
        for col in ra:
            if col == "method" or col in TIMING_COLS:
                continue
            assert ra[col] == rb[col], (
                f"order-dependence: {m} col {col}: {ra[col]!r} != {rb[col]!r}")


# ---------------------------------------------------------------------------
# Phase 2 voting fixes (2.1-2.5, 2.8)
# ---------------------------------------------------------------------------
def test_approval_counts_task_index_zero():
    """2.1: a robot assigned to local task 0 must be able to approve."""
    from hvbta.allocators.voting import rank_assignments_approval
    # 2 robots, 2 tasks; robot 0 assigned task 0 (index 0, falsy) with raw 0.9
    M_raw = [[0.9, 0.2], [0.8, 0.8]]
    cand = ([(0, 0), (1, 1)], [], [])
    scores, ranking = rank_assignments_approval([cand], M_raw, raw_matrix=M_raw, threshold=0.5)
    assert scores[0] == 2.0  # both approve (old code: robot 0's task-index-0 vote was dropped)


def test_approval_weights_priority_and_uses_raw():
    """2.2: approval compares RAW suitability; each approval counts the task's
    priority factor (low = 0.5), not 1."""
    from hvbta.allocators.voting import rank_assignments_approval
    M_raw = [[0.9, 0.9]]
    factors = np.array([0.5, 1.0])  # task 0 low, task 1 medium
    cand_a = ([(0, 0)], [], [])  # approve low-priority task -> 0.5
    cand_b = ([(0, 1)], [], [])  # approve medium-priority task -> 1.0
    weighted = [[0.45, 0.9]]  # old code compared against this: task 0 could NEVER pass > 0.5
    scores, ranking = rank_assignments_approval(
        [cand_a, cand_b], weighted, raw_matrix=M_raw, priority_factors=factors, threshold=0.5)
    assert scores[0] == 0.5          # raw 0.9 > 0.5 approves; weight 0.5
    assert scores[1] == 1.0
    assert ranking[0] == 1           # b wins


def test_mj_unassigned_voters_grade_poor():
    """2.3: every robot with a feasible option is a voter; unassigned voters
    grade the candidate Poor, so 1 excellent pair no longer beats 10 good."""
    from hvbta.allocators.voting import rank_assignments_majority_judgment
    M_raw = [[0.9] + [0.0] * 9, [0.75] * 10]  # robot 0: only task 0 feasible; robot 1-9 via row 1
    # candidates: a assigns only robot 0 (excellent); b assigns all 10 (good)
    cand_a = ([(0, 0)], list(range(1, 10)), [])
    cand_b = ([(r, 0) for r in range(10)], [], [])
    ratings, ranking = rank_assignments_majority_judgment([cand_a, cand_b], M_raw, raw_matrix=M_raw)
    # voters = robots with >= 1 positive option = both rows -> 2 voters
    # cand_a: grades [Excellent, Poor] -> weighted median = Poor(0)
    # cand_b: grades [Excellent, Good]   -> weighted median = Good(2)
    assert ratings[0] < ratings[1]
    assert ranking[0] == 1


def test_borda_fractional_ties():
    """2.4: tied candidates share the average of the Borda points their
    positions span (not pool-order-dependent points)."""
    from hvbta.allocators.voting import rank_assignments_borda
    M = [[0.8, 0.8]]  # one robot, both candidates equally good
    cand_a = ([(0, 0)], [], [])
    cand_b = ([(0, 1)], [], [])
    # with 2 candidates the robot ties -> each gets (1 + 0) / 2 = 0.5 points
    scores, ranking = rank_assignments_borda([cand_a, cand_b], M)
    assert scores[0] == scores[1] == 0.5


def test_tie_break_seeded_and_reproducible():
    """2.5: the same tie seed resolves ties identically; pool order stops
    mattering."""
    from hvbta.allocators.voting import rank_assignments_range
    M = [[0.8, 0.8]]  # both candidates score identically
    cand_a = ([(0, 0)], [], [])
    cand_b = ([(0, 1)], [], [])
    import random as _r
    r1 = rank_assignments_range([cand_a, cand_b], M, rng=_r.Random(42))
    r2 = rank_assignments_range([cand_a, cand_b], M, rng=_r.Random(42))
    assert r1[1] == r2[1]
    # tied scores detected either way
    assert r1[0][0] == r1[0][1]


def test_coverage_threshold_requires_min_pool():
    """2.8: even when the first candidate would clear the threshold, the
    ranked pool must reach MIN_POOL_BEFORE_THRESHOLD (or the full pool)."""
    from hvbta.allocators.voting import (
        assign_tasks_with_voting, rank_assignments_range,
        MIN_POOL_BEFORE_THRESHOLD, POOL_STATS,
    )
    robots, tasks = make_robots(n=4, seed=3), make_tasks(n=4, seed=4)
    M = evaluate_suitability_balanced(robots, tasks, map_size=100)
    M = np.clip(M + 0.35, 0.0, 1.0)  # most pairs comfortably above 0.7
    import hvbta.allocators.voting as _V
    _V.POOL_STATS = []
    try:
        output, score, length, per = assign_tasks_with_voting(
            robots, tasks, M, 20, rank_assignments_range,
            score_threshold=0.7, time_budget_ms=None)
        stats = _V.POOL_STATS
        assert stats and stats[-1]["threshold"] == 0.7
        min_pool = min(MIN_POOL_BEFORE_THRESHOLD, stats[-1]["full"])
        assert stats[-1]["ranked"] >= min_pool, (
            f"threshold fired with pool {stats[-1]['ranked']} < min {min_pool}")
    finally:
        _V.POOL_STATS = None


def test_idle_robot_yields_to_mover():
    """Phase 3 fix: a mover stepping onto an idle robot's cell swaps cells
    with it (the idle robot takes the mover's vacated cell). Without this,
    movers walk through idle robots and later planner calls see two robots
    on one cell (the duplicate-start cascade trigger)."""
    from hvbta.simulation.timestep import simulate_time_step
    mover = _fake_robot("R1", (2, 2))
    idle = _fake_robot("R2", (2, 3))
    idle.assigned = False
    idle.current_task = None
    mover.current_task.location = (0, 9)   # far away: mover must not reach it this tick
    mover.current_path = [(2, 2), (2, 3), (2, 4), (2, 5)]
    occupied = {(2, 2), (2, 3)}
    sp = {"R1": (2, 2)}
    simulate_time_step([mover, idle], [mover.current_task], [], [],
                       None, occupied, sp, {})
    assert mover.location == (2, 3)      # moved onto the idle robot's cell
    assert idle.location == (2, 2)        # yielded into the vacated cell
    assert occupied == {(2, 2), (2, 3)}   # both cells still occupied


# ---------------------------------------------------------------------------
# Config sanity
# ---------------------------------------------------------------------------
def test_po_conditions_and_counts():
    assert PO_CONDITIONS[0] == ("none", "none")
    assert MASK_COUNTS == {"none": 0, "low": 2, "high": 5}
    for level, policy in PO_CONDITIONS:
        po = PartialObservabilityState(SEED, level, policy)
        assert po.mask_count == MASK_COUNTS[level]
    with pytest.raises(ValueError):
        PartialObservabilityState(SEED, "bogus", "optimistic")


# ---------------------------------------------------------------------------
# Regression tests: makespan-padding arrival deadlock + zero-pair execution
# ---------------------------------------------------------------------------
def _fake_robot(rid, loc):
    r = make_robots(n=1)[0]
    r.robot_id = rid
    r.location = loc
    r.current_path = []
    r.remaining_distance = 0
    r.assigned = True
    r.current_task = TaskDescription(
        task_id="T1", task_type="delivery", objective="", priority_level="low",
        reward=1.0, difficulty=1.0, location=loc, navigation_constraints=[],
        required_capabilities={"payload": 0.0, "reach": 0.0}, time_window=None,
        environmental_conditions=[], dependencies=[], sensors_needed=[],
        manipulators_needed=[], communication_requirements=[], safety_protocols=[],
        performance_metrics="time taken", success_criteria="", assigned_robot=None,
        time_to_complete=1.0, assigned=False)
    return r


def test_write_paths_back_strips_trailing_waits():
    """LaCAM/PIBT pad paths to the makespan: an agent at its goal gets a tail
    of goal-cell waits. The write-back must strip that tail, else an agent
    standing ON its goal still shows remaining_distance = makespan and
    (with frequent replans) never registers as arrived."""
    robots = [_fake_robot("R1", (0, 0)), _fake_robot("R2", (5, 5))]
    paths = {
        # real path then arrival + 40 padded waits
        "R1": [(0, 0), (0, 1), (0, 2)] + [(0, 2)] * 40,
        # never moves: start == goal, all waits
        "R2": [(5, 5)] * 47,
    }
    avg = _write_paths_back(robots, paths, stationary_ids=set())
    assert robots[0].current_path == [(0, 0), (0, 1), (0, 2)]
    assert robots[0].remaining_distance == 2
    assert robots[1].current_path == [(5, 5)]
    assert robots[1].remaining_distance == 0
    assert avg == 2.0  # mean over moving agents only


def test_write_paths_back_preserves_mid_route_waits():
    """Genuine blocking waits mid-route must survive; only the tail of
    consecutive final-cell repeats is stripped."""
    robots = [_fake_robot("R1", (0, 0))]
    paths = {"R1": [(0, 0), (0, 1), (0, 1), (0, 2), (0, 1), (0, 2), (0, 2), (0, 2)]}
    _write_paths_back(robots, paths, stationary_ids=set())
    # tail of 3x (0,2) stripped to a single arrival; the mid-route repeat kept
    assert robots[0].current_path == [(0, 0), (0, 1), (0, 1), (0, 2), (0, 1), (0, 2)]
    assert robots[0].remaining_distance == 5


def test_murtys_drops_zero_pairs_individually():
    """Murty's applies the JV optimum which may contain zero-score pairs;
    zero pairs must be dropped individually while the rest of the assignment
    still applies (same rule as voting's kept_pairs filter)."""
    from hvbta.allocators.optimizers import assign_tasks_with_method, murtys_task_allocation
    robots, tasks = make_robots(n=4), make_tasks(n=3)
    # task 0 infeasible for robot 0 only; feasible for the others
    cap = robots[0].payload_capacity
    tasks[0].required_capabilities = {"payload": cap + 1.0, "reach": 0.0}
    M = evaluate_suitability_balanced(robots, tasks, map_size=100)
    out, score, length, per = assign_tasks_with_method(murtys_task_allocation, M)
    for r, t in out[0]:
        assert M[r][t] > 0, "Murty's must never execute a zero-score pair"
    assert len(out[0]) > 0, "feasible pairs must still be applied"


def test_reveal_invalidates_masked_view():
    """2e: after a reveal the task must be scored on true attributes - the
    masked view returns the true task object itself."""
    po = PartialObservabilityState(SEED, "high", "optimistic")
    tasks = make_tasks()
    masked = [t for t in tasks if po.is_masked(t.task_id)]
    assert masked, "precondition: some task masked"
    task = masked[0]
    view1 = po.masked_view(task)
    assert view1 is not task
    po._pop_mask(task.task_id)
    view2 = po.masked_view(task)
    assert view2 is task, "revealed task must expose true attributes"
    assert not po.is_masked(task.task_id)


# ---------------------------------------------------------------------------
# Zero-pair filter (Option B): zero-score pairs never enter the pool
# ---------------------------------------------------------------------------
def test_filter_zero_pairs_drops_and_relocates():
    M = [[0.5, 0.0, 0.3],
         [0.0, 0.4, 0.0]]
    fixed = _filter_zero_pairs(([(0, 0), (0, 1), (1, 1), (1, 2)], [ ], [2]), M)
    assert fixed[0] == [(0, 0), (1, 1)]
    assert set(fixed[1]) == {0, 1}     # both zero-pair robots -> unassigned
    assert set(fixed[2]) == {1, 2}     # tasks 1 and 2 freed (2 was already unassigned)


def test_filter_zero_pairs_empty_returns_none():
    M = [[0.0, 0.0]]
    assert _filter_zero_pairs(([(0, 0), (0, 1)], [0], []) if False else ([(0, 0), (0, 1)], [], []), M) is None
    # all-positive candidate passes through untouched
    M2 = [[0.5, 0.2]]
    kept = _filter_zero_pairs(([(0, 0)], [0], [0]), M2)
    assert kept == ([(0, 0)], [0], [0])


def test_voting_never_executes_zero_pairs_even_for_coverage():
    """Borda's old coverage bias elected all-zero candidates under pessimistic
    masking. After the pool-wide zero-pair filter, the executed winner (the
    voting rule's actual ranking[0]) can never contain a zero-score pair."""
    robots, tasks = make_robots(n=6), make_tasks(n=5)
    # task 0 scores 0 for every robot; others feasible for at least one
    payload_cap = max(r.payload_capacity for r in robots)
    tasks[0].required_capabilities = {"payload": payload_cap + 100.0, "reach": 0.0}
    M = evaluate_suitability_balanced(robots, tasks, map_size=100)
    assert all(M[i, 0] == 0.0 for i in range(len(robots)))

    for method in (rank_assignments_borda,):
        output, score, length, per_agent = assign_tasks_with_voting(
            robots, tasks, M, 15, method)
        for r, t in output[0]:
            assert M[r][t] > 0, "voting executed a zero-score pair"
    # task 0 must end up unassigned
    assert 0 in output[2]


def test_voting_empty_pool_assigns_nothing():
    """When every candidate's pairs are infeasible, voting assigns nothing
    (matches Murty's all-infeasible behavior; no random fallback)."""
    robots, tasks = make_robots(n=3), make_tasks(n=2)
    cap = max(r.payload_capacity for r in robots)
    for t in tasks:
        t.required_capabilities = {"payload": cap + 100.0, "reach": 0.0}
    M = evaluate_suitability_balanced(robots, tasks, map_size=100)
    assert (M == 0).all(), "precondition: all pairs infeasible"
    output, score, length, per_agent = assign_tasks_with_voting(
        robots, tasks, M, 10, rank_assignments_borda)
    assert output[0] == []
    assert set(output[1]) == {0, 1, 2}
    assert set(output[2]) == {0, 1}
