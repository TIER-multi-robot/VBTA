import numpy as np
from collections import deque

# Master switch for the Phase 3v2 anti-collision design: idle robots join
# every planner call as agents whose goal is their current cell (never hard
# obstacles). Exists so the pre-fix behavior can be reproduced for
# measurement; leave True.
IDLE_AGENTS_IN_PLAN = True


def _nearest_free_cell(source, obstacles, taken, rows, cols):
    """BFS over the whole reachable map from `source` for the nearest cell
    that is free (not an obstacle, not in `taken`). Returns the cell or None."""
    if source is None:
        return None
    q = deque([source])
    seen = {source}
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cell = (nr, nc)
            if cell in seen:
                continue
            seen.add(cell)
            if cell not in obstacles and cell not in taken:
                return cell
            q.append(cell)
    return None


# ---------------------------------------------------------------------------
# Shared front-matter: identical to run_cbs, factored out so every planner
# handles stationary blockers and the trivial/duplicate cases the same way.
# ---------------------------------------------------------------------------
def _prepare_agents(robots, start_positions, goal_positions, map_dict=None):
    """
    Returns (agents, stationary_ids, failure) — failure is None on success,
    otherwise a classification dict (kind: duplicate_start / duplicate_goal /
    invalid_input) describing exactly why the input is unusable. The caller
    should NOT invoke the planner when failure is not None; it logs the
    failure directly.

    agents: list of dicts {id, start, goal} for every robot that should
            occupy space this round:
              - movers (from start/goal_positions),
              - idle robots as stay-put agents whose goal is their CURRENT
                cell (not obstacles, not stationary). LaCAM/PIBT may
                displace and return them; their paths are written back.
              - EXCEPTION: an idle robot standing on a mover's goal cell
                gets a step-aside goal instead (nearest free cell not
                claimed by any agent's start or goal), searched over the
                WHOLE reachable map.

    Input validation (before any planner call):
    - After step-aside assignment, every start and goal in the call is
      claimed. If a duplicate start or goal remains, the failure dict says
      so with the agents involved — the planner is never invoked on an
      unsolvable instance.
    """
    if not start_positions or not goal_positions:
        return None, None, None  # caller returns (0.0, True): nothing to plan

    aug_starts = dict(start_positions)
    aug_goals = dict(goal_positions)
    stationary_ids = set()
    idle_ids = []
    for r in robots:
        if r.robot_id in aug_starts:
            continue
        if getattr(r, "location", None) is None:
            continue
        if not (getattr(r, "assigned", False) and getattr(r, "current_task", None) is not None):
            if IDLE_AGENTS_IN_PLAN:
                aug_starts[r.robot_id] = r.location
                aug_goals[r.robot_id] = r.location
                idle_ids.append(r.robot_id)
            continue
        aug_starts[r.robot_id] = r.location
        aug_goals[r.robot_id] = r.location
        stationary_ids.add(r.robot_id)

    # Step-aside goals for idle agents on mover goals: compute AFTER all
    # mover goals are set, claim every start and goal in the call, search
    # the whole reachable map (no radius cap).
    if idle_ids and map_dict is not None:
        obstacles = set(map_dict.get("obstacles") or [])
        rows, cols = map_dict.get("dimension", (0, 0))
        mover_goals = set(goal_positions.values())
        taken = set(aug_starts.values()) | set(aug_goals.values())
        for rid in idle_ids:
            if aug_starts[rid] in mover_goals:
                aside = _nearest_free_cell(aug_starts[rid], obstacles, taken, rows, cols)
                if aside is not None:
                    aug_goals[rid] = aside
                    taken.add(aside)
                # no reachable free cell: leave the stay-put goal; the
                # input validation below will catch the duplicate goal.

    # Input validation on the FULL agent set (starts AND goals):
    starts = list(aug_starts.values())
    goals = list(aug_goals.values())
    if len(starts) != len(set(starts)):
        dup = [s for s in set(starts) if starts.count(s) > 1]
        involved = [{"id": rid, "start": aug_starts[rid], "goal": aug_goals[rid]}
                    for rid in aug_starts if aug_starts[rid] in dup]
        return None, None, {"kind": "duplicate_start", "cells": [c for c in dup],
                            "agents": involved}
    if len(goals) != len(set(goals)):
        dup = [g for g in set(goals) if goals.count(g) > 1]
        involved = [{"id": rid, "start": aug_starts[rid], "goal": aug_goals[rid]}
                    for rid in aug_starts if aug_goals[rid] in dup]
        return None, None, {"kind": "duplicate_goal", "cells": [c for c in dup],
                            "agents": involved}

    agents = [
        {"id": rid, "start": aug_starts[rid], "goal": aug_goals[rid]}
        for rid in aug_starts
    ]
    return agents, stationary_ids, None


def _write_paths_back(robots, paths, stationary_ids):
    """
    paths: {robot_id: [(x, y), ...]}  full path including the start cell.
    Writes current_path / remaining_distance, skipping stationary blockers,
    and returns the mean path length over MOVING agents (matches run_cbs).

    Engines that return whole solutions (LaCAM, PIBT-stepwise) pad every
    agent's path to the solution makespan: an agent that reaches its goal
    early gets trailing goal-cell waits. Writing that padding as
    remaining_distance makes an agent standing ON its goal still appear to
    be 40+ steps from "arriving", and frequent replans (fleet churn) reset
    the padding before it elapses - so the agent never registers as
    arrived, never works, and tasks never complete. Strip trailing waits
    so the path ends the moment the agent settles on its final cell.
    Mid-route waits (genuine blocking) are preserved; only the tail of
    consecutive repeats of the final cell is removed.
    """
    id_to_index = {r.robot_id: i for i, r in enumerate(robots)}
    valid_lengths = []
    for rid, path in paths.items():
        if rid not in id_to_index:
            continue
        if rid in stationary_ids:
            continue  # don't clobber a blocker's (empty) path with a wait schedule
        p = [(x, y) for (x, y) in path]
        while len(p) > 1 and p[-1] == p[-2]:
            p.pop()
        r = robots[id_to_index[rid]]
        r.current_path = p
        r.remaining_distance = max(0, len(p) - 1)
        if len(p) > 1:
            valid_lengths.append(len(p) - 1)
    return float(np.mean(valid_lengths)) if valid_lengths else 0.0


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])