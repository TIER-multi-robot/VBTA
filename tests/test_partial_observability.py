"""
Validation tests for the partial-observability experiment
(spec: partial_observability_experiment.md §9).

1. Leak test       - masked fields' true values never reach the scorer.
2. Mask fairness   - masks identical across methods/policies; low ⊂ high.
3. Determinism     - same seed -> same masks, same masked views.
4. Reveal-once     - each task's mask is removed at most once.
5. No-bounce       - a task infeasible for every robot is never assigned
                     (voting filters zero-suitability pairs).
6. Hard-fail parity - balanced_hard_fail True ⟹ _balanced_pair == 0.
"""
import random

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
    in a group) see identical masks for the same (seed, task)."""
    tasks = make_tasks()
    po_opt = PartialObservabilityState(SEED, "high", "optimistic")
    po_pess = PartialObservabilityState(SEED, "high", "pessimistic")
    for t in tasks:
        assert po_opt.mask_of(t.task_id) == po_pess.mask_of(t.task_id)
    # and a second instance with the same config reproduces them (§9.3)
    po_again = PartialObservabilityState(SEED, "high", "optimistic")
    for t in tasks:
        assert po_again.mask_of(t.task_id) == po_opt.mask_of(t.task_id)


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
