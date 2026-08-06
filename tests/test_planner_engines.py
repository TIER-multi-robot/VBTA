"""
Phase 8 engine tests: validity + conflict-freeness for all four planners.

For each planner and each fixture, assert:
  - success (planner returns ok=True) for the solvable fixtures,
  - every mover path starts at its start and ends at its goal,
  - remaining_distance == len(path)-1,
  - no vertex conflicts and no edge/swap conflicts among movers at any timestep,
  - stationary blockers are NOT given a path and movers never enter a blocker's
    cell (the new intended behavior for CBS / PP / LaCAM; PIBT-standalone is
    incomplete and may fail the blocker fixture, which is documented).

Runnable both ways:
    pytest tests/test_planner_engines.py
    python tests/test_planner_engines.py
"""

import copy
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hvbta.pathfinding.MAPF_wrapper import run_planner


class Robot:
    def __init__(self, rid, loc, assigned=False, current_task=None):
        self.robot_id = rid
        self.location = loc
        self.assigned = assigned
        self.current_task = current_task
        self.current_path = []
        self.remaining_distance = 0


GRID = {"dimension": (5, 5), "obstacles": []}


def _movers(robots):
    return [r for r in robots if r.assigned and r.current_path]


def _pad(path, length):
    if len(path) >= length:
        return path[:length]
    return path + [path[-1]] * (length - len(path))


def _assert_valid_and_conflict_free(robots, start_positions, goal_positions, blocked_cells):
    movers = _movers(robots)
    assert movers, "no mover paths written back"
    for r in movers:
        assert r.current_path[0] == start_positions[r.robot_id], (
            f"{r.robot_id} doesn't start at its start: {r.current_path[0]}")
        assert r.current_path[-1] == goal_positions[r.robot_id], (
            f"{r.robot_id} doesn't end at its goal: {r.current_path[-1]}")
        assert r.remaining_distance == max(0, len(r.current_path) - 1), (
            f"{r.robot_id} remaining_distance={r.remaining_distance} != {len(r.current_path)-1}")
        for cell in r.current_path:
            assert cell not in blocked_cells, (
                f"{r.robot_id} path enters a stationary blocker's cell {cell}: "
                f"{r.current_path}")

    horizon = max(len(r.current_path) for r in movers)
    padded = {r.robot_id: _pad(r.current_path, horizon) for r in movers}
    ids = list(padded)
    for t in range(horizon):
        cells = {}
        for aid in ids:
            c = padded[aid][t]
            assert c not in cells, (
                f"vertex conflict at t={t}: {cells[c]} and {aid} both at {c}")
            cells[c] = aid
    for t in range(horizon - 1):
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                if padded[a][t] == padded[b][t + 1] and padded[a][t + 1] == padded[b][t]:
                    raise AssertionError(
                        f"edge/swap conflict at t={t}: {a} and {b}")


def _run(planner, robots, start_positions, goal_positions, map_dict):
    rs = copy.deepcopy(robots)
    avg, ok = run_planner(planner, rs, start_positions, goal_positions, map_dict)
    return rs, avg, ok


# --- fixtures ---------------------------------------------------------------
def _swap_fixture():
    r1 = Robot("R1", (0, 2), assigned=True, current_task="T1")
    r2 = Robot("R2", (4, 2), assigned=True, current_task="T2")
    return [r1, r2], {"R1": (0, 2), "R2": (4, 2)}, {"R1": (4, 2), "R2": (0, 2)}


def _blocker_fixture():
    r1 = Robot("R1", (0, 2), assigned=True, current_task="T1")
    r2 = Robot("R2", (4, 2), assigned=True, current_task="T2")
    r3 = Robot("R3", (2, 2), assigned=False)  # idle blocker
    return ([r1, r2, r3], {"R1": (0, 2), "R2": (4, 2)},
            {"R1": (4, 2), "R2": (0, 2)}, {(2, 2)})


def _three_fixture():
    r1 = Robot("R1", (0, 0), assigned=True, current_task="T1")
    r2 = Robot("R2", (0, 4), assigned=True, current_task="T2")
    r3 = Robot("R3", (4, 0), assigned=True, current_task="T3")
    return ([r1, r2, r3],
            {"R1": (0, 0), "R2": (0, 4), "R3": (4, 0)},
            {"R1": (4, 4), "R2": (4, 0), "R3": (0, 4)})


# --- tests ------------------------------------------------------------------
# Solvers expected to succeed on the blocker fixture (complete or strong enough).
STRONG = ["cbs", "pp", "lacam"]
# All four must solve the no-blocker fixtures.
ALL = ["cbs", "pp", "lacam", "pibt"]


def test_swap_all_planners():
    robots, sp, gp = _swap_fixture()
    for p in ALL:
        rs, _, ok = _run(p, robots, sp, gp, GRID)
        assert ok, f"{p} failed the swap fixture"
        _assert_valid_and_conflict_free(rs, sp, gp, set())


def test_three_all_planners():
    robots, sp, gp = _three_fixture()
    for p in ALL:
        rs, _, ok = _run(p, robots, sp, gp, GRID)
        assert ok, f"{p} failed the three-mover fixture"
        _assert_valid_and_conflict_free(rs, sp, gp, set())


def test_blocker_strong_planners():
    robots, sp, gp, blocked = _blocker_fixture()
    for p in STRONG:
        rs, _, ok = _run(p, robots, sp, gp, GRID)
        assert ok, f"{p} failed the blocker fixture"
        _assert_valid_and_conflict_free(rs, sp, gp, blocked)
        # blocker's own path must stay empty
        blocker = next(r for r in rs if r.robot_id == "R3")
        assert blocker.current_path == [], (
            f"{p} overwrote stationary blocker's path: {blocker.current_path}")


def test_blocker_pibt_documented_incomplete():
    # PIBT is incomplete; the blocker fixture is the canonical case it can miss.
    # This test documents that: PIBT may return ok=False here, and that's
    # accepted (the sim treats it as a planner failure). We only assert that IF
    # it succeeds, the result is valid + conflict-free + avoids the blocker.
    robots, sp, gp, blocked = _blocker_fixture()
    rs, _, ok = _run("pibt", robots, sp, gp, GRID)
    if ok:
        _assert_valid_and_conflict_free(rs, sp, gp, blocked)


def test_unknown_planner_raises():
    robots, sp, gp = _swap_fixture()
    raised = False
    try:
        _run("nope", robots, sp, gp, GRID)
    except ValueError:
        raised = True
    assert raised, "unknown planner should raise ValueError"


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
    print("\nAll planner engine tests passed.")
