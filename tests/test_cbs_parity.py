"""
Phase 6 parity gate: prove routing CBS through the new wrapper didn't break it.

CBS is inherently NONDETERMINISTIC (the parallel low-level A* in
Final_CBS.compute_solution returns different optimal paths across runs -- three
back-to-back calls to the old run_cbs on the same input produce three different
path sets). Exact-path parity is therefore impossible: old run_cbs doesn't even
reproduce itself. CBS is, however, COST-OPTIMAL, so the optimal cost is stable
across runs. The real question -- "did restructuring break CBS?" -- is answered
by checking the invariants CBS actually guarantees, split into two cases:

  1. NO-IDLE instances (every robot has a task) -- the parity case.
     The frozen golden run_cbs (below) and run_planner("cbs", ...) must both:
       - succeed,
       - return the SAME optimal cost (avg_len + sum of remaining distances),
       - produce valid paths (each mover: path[0]==start, path[-1]==goal),
       - produce conflict-free paths (no vertex or edge conflicts among movers).
     Cost match + valid + conflict-free == "CBS still behaves like CBS."
     (Per-robot path/rd equality is NOT asserted -- CBS is nondeterministic
     about which robot takes a detour, only the total cost is invariant.)

  2. WITH-IDLE instances -- the new-behavior case.
     The new cbs_planner passes stationary blockers to Environment (the golden
     run_cbs's build_cbs_agents filtered them back out, so its augmentation was
     a no-op). So the new paths must additionally be collision-free w.r.t.
     idle robots' cells: no mover's path may pass through an idle robot's
     cell. Different invariant from parity, on purpose -- the old behavior on
     these instances is known-wrong, so we assert the new intended behavior.

Runnable both ways:
    pytest tests/test_cbs_parity.py
    python tests/test_cbs_parity.py
"""

import copy
import os
import sys

# Make the repo root importable when run directly (python tests/...).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hvbta.pathfinding.Final_CBS import CBS, Environment
from hvbta.pathfinding.CBS_helper import build_cbs_agents
from hvbta.pathfinding.MAPF_wrapper import run_planner          # new, dispatch


# Frozen golden reference of the OLD run_cbs behavior (the version that lived in
# main_simulation.py before the Phase 6 retirement). Kept here so the parity
# gate survives the deletion of run_cbs from the sim: the test diffs the new
# run_planner("cbs", ...) against this frozen reference, not against live sim
# code. On the no-idle parity fixtures this is exactly what the old run_cbs
# did (build_cbs_agents filters unassigned robots, so its stationary-blocker
# augmentation was a no-op there anyway).
def _golden_run_cbs(robots, start_positions, goal_positions, map_dict):
    if not start_positions or not goal_positions:
        return 0.0, True
    aug_starts = dict(start_positions)
    aug_goals = dict(goal_positions)
    stationary_ids = set()
    for r in robots:
        if r.robot_id in aug_starts:
            continue
        if getattr(r, "location", None) is None:
            continue
        aug_starts[r.robot_id] = r.location
        aug_goals[r.robot_id] = r.location
        stationary_ids.add(r.robot_id)
    agents = build_cbs_agents(robots, aug_starts, aug_goals)
    if not agents:
        return 0.0, True
    start_locations = [a["start"] for a in agents]
    if len(start_locations) != len(set(start_locations)):
        return 0.0, False
    env = Environment(dimension=map_dict["dimension"], agents=agents,
                      obstacles=map_dict["obstacles"])
    res = CBS(env).search()
    if not res:
        return 0.0, False
    solution, _n, _c = res
    id_to_index = {r.robot_id: idx for idx, r in enumerate(robots)}
    valid_lengths = []
    for robot_id, schedule in solution.items():
        if robot_id not in id_to_index:
            continue
        r = robots[id_to_index[robot_id]]
        if robot_id in stationary_ids:
            continue
        r.current_path = [(p["x"], p["y"]) for p in schedule]
        r.remaining_distance = max(0, len(schedule) - 1)
        if len(schedule) > 1:
            valid_lengths.append(len(schedule) - 1)
    return (float(sum(valid_lengths) / len(valid_lengths)) if valid_lengths else 0.0), True


class Robot:
    """Minimal robot stand-in: only the fields run_cbs / run_planner touch."""
    def __init__(self, rid, loc, assigned=False, current_task=None):
        self.robot_id = rid
        self.location = loc
        self.assigned = assigned
        self.current_task = current_task
        self.current_path = []
        self.remaining_distance = 0


def _empty_grid(rows, cols):
    return {"dimension": (rows, cols), "obstacles": []}


def _run_both(robots, start_positions, goal_positions, map_dict):
    """Run the frozen golden run_cbs and new run_planner('cbs',...) on
    independent copies."""
    robots_old = copy.deepcopy(robots)
    robots_new = copy.deepcopy(robots)
    old_avg, old_ok = _golden_run_cbs(robots_old, start_positions, goal_positions, map_dict)
    new_avg, new_ok = run_planner("cbs", robots_new, start_positions, goal_positions, map_dict)
    return robots_old, robots_new, old_avg, new_avg, old_ok, new_ok


def _movers(robots):
    return [r for r in robots if r.assigned and r.current_path]


def _pad(path, length):
    """Pad a path to `length` by holding the last cell (agent waits at goal)."""
    if len(path) >= length:
        return path[:length]
    return path + [path[-1]] * (length - len(path))


def _assert_valid_and_conflict_free(robots, start_positions, goal_positions):
    """Every mover path starts at start, ends at goal, and no two movers
    conflict (vertex or edge) at any timestep."""
    movers = _movers(robots)
    assert movers, "no mover paths written back"
    for r in movers:
        assert r.current_path[0] == start_positions[r.robot_id], (
            f"{r.robot_id} path doesn't start at its start: "
            f"{r.current_path[0]} != {start_positions[r.robot_id]}")
        assert r.current_path[-1] == goal_positions[r.robot_id], (
            f"{r.robot_id} path doesn't end at its goal: "
            f"{r.current_path[-1]} != {goal_positions[r.robot_id]}")
        assert r.remaining_distance == max(0, len(r.current_path) - 1), (
            f"{r.robot_id} remaining_distance={r.remaining_distance} "
            f"!= len(path)-1={len(r.current_path)-1}")

    horizon = max(len(r.current_path) for r in movers)
    padded = {r.robot_id: _pad(r.current_path, horizon) for r in movers}
    ids = list(padded)
    for t in range(horizon):
        cells = {}
        for aid in ids:
            c = padded[aid][t]
            assert c not in cells, (
                f"vertex conflict at t={t}: agents {cells[c]} and {aid} both at {c}")
            cells[c] = aid
    # edge (swap) conflict: two agents trading cells between t and t+1.
    for t in range(horizon - 1):
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if padded[a][t] == padded[b][t + 1] and padded[a][t + 1] == padded[b][t]:
                    raise AssertionError(
                        f"edge/swap conflict at t={t}: agents {a} and {b} "
                        f"swapped {padded[a][t]}<->{padded[b][t]}")


# ---------------------------------------------------------------------------
# Invariant 1: COST-PARITY + validity + conflict-freeness on no-idle instances.
# ---------------------------------------------------------------------------
def _parity_case(robots, start_positions, goal_positions, map_dict):
    (robots_old, robots_new, old_avg, new_avg,
     old_ok, new_ok) = _run_both(robots, start_positions, goal_positions, map_dict)
    assert old_ok is True, "old run_cbs failed on a no-idle fixture"
    assert new_ok is True, "new run_planner('cbs') failed on a no-idle fixture"

    # cost parity: CBS is optimal, so the optimal cost must match. Paths differ
    # nondeterministically (which robot takes a detour is a tie-break), so we
    # compare the TOTAL cost -- avg_len and the sum of remaining distances --
    # NOT per-robot values.
    assert old_avg == new_avg, (
        f"cost mismatch: old_avg={old_avg} new_avg={new_avg}")
    old_rd_sum = sum(r.remaining_distance for r in robots_old)
    new_rd_sum = sum(r.remaining_distance for r in robots_new)
    assert old_rd_sum == new_rd_sum, (
        f"sum-of-costs mismatch: old={old_rd_sum} new={new_rd_sum}")
    old_ids = {r.robot_id for r in robots_old}
    new_ids = {r.robot_id for r in robots_new}
    assert old_ids == new_ids, "robot id sets diverged"

    # both must be valid + conflict-free (the actual "CBS still works" check).
    _assert_valid_and_conflict_free(robots_old, start_positions, goal_positions)
    _assert_valid_and_conflict_free(robots_new, start_positions, goal_positions)


def test_parity_two_movers_no_idle():
    r1 = Robot("R1", (0, 0), assigned=True, current_task="T1")
    r2 = Robot("R2", (4, 0), assigned=True, current_task="T2")
    robots = [r1, r2]
    start_positions = {r.robot_id: r.location for r in robots}
    goal_positions = {"R1": (4, 4), "R2": (0, 4)}
    _parity_case(robots, start_positions, goal_positions, _empty_grid(5, 5))


def test_parity_three_movers_no_idle():
    r1 = Robot("R1", (0, 0), assigned=True, current_task="T1")
    r2 = Robot("R2", (0, 4), assigned=True, current_task="T2")
    r3 = Robot("R3", (4, 0), assigned=True, current_task="T3")
    robots = [r1, r2, r3]
    start_positions = {r.robot_id: r.location for r in robots}
    goal_positions = {"R1": (4, 4), "R2": (4, 0), "R3": (0, 4)}
    _parity_case(robots, start_positions, goal_positions, _empty_grid(5, 5))


def test_parity_opposite_swap_no_idle():
    # classic head-on swap -- forces CBS to use the wait action
    r1 = Robot("R1", (0, 2), assigned=True, current_task="T1")
    r2 = Robot("R2", (4, 2), assigned=True, current_task="T2")
    robots = [r1, r2]
    start_positions = {r.robot_id: r.location for r in robots}
    goal_positions = {"R1": (4, 2), "R2": (0, 2)}
    _parity_case(robots, start_positions, goal_positions, _empty_grid(5, 5))


# ---------------------------------------------------------------------------
# Invariant 2: COLLISION-FREE W.R.T. IDLE CELLS on with-idle instances.
# ---------------------------------------------------------------------------
def test_idle_blocker_never_occupied_by_movers():
    r1 = Robot("R1", (0, 2), assigned=True, current_task="T1")
    r2 = Robot("R2", (4, 2), assigned=True, current_task="T2")
    r3 = Robot("R3", (2, 2), assigned=False, current_task=None)  # idle blocker
    robots = [r1, r2, r3]
    start_positions = {"R1": (0, 2), "R2": (4, 2)}
    goal_positions = {"R1": (4, 2), "R2": (0, 2)}

    (_, robots_new, _, _, _, new_ok) = _run_both(
        robots, start_positions, goal_positions, _empty_grid(5, 5))
    assert new_ok is True, "new cbs_planner failed on a with-idle fixture"

    blocker_cell = (2, 2)
    for r in robots_new:
        if r.assigned:
            assert blocker_cell not in r.current_path, (
                f"mover {r.robot_id} path passes through idle blocker's cell "
                f"{blocker_cell}: {r.current_path}")


def test_idle_blocker_path_left_empty():
    r1 = Robot("R1", (0, 0), assigned=True, current_task="T1")
    r2 = Robot("R2", (2, 2), assigned=False, current_task=None)  # idle blocker
    robots = [r1, r2]
    start_positions = {"R1": (0, 0)}
    goal_positions = {"R1": (4, 4)}

    (_, robots_new, _, _, _, new_ok) = _run_both(
        robots, start_positions, goal_positions, _empty_grid(5, 5))
    assert new_ok is True
    blocker = next(r for r in robots_new if r.robot_id == "R2")
    assert blocker.current_path == [], (
        f"stationary blocker's path was overwritten: {blocker.current_path}")


if __name__ == "__main__":
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures.append((name, str(e)))
                print(f"FAIL  {name}: {e}")
            except Exception as e:  # noqa: BLE001
                failures.append((name, repr(e)))
                print(f"ERROR {name}: {e!r}")
    if failures:
        print(f"\n{len(failures)} test(s) failed.")
        sys.exit(1)
    print("\nAll CBS parity tests passed.")
