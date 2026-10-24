import numpy as np
from collections import deque

# Master switch for the Phase 3v2 anti-collision design: idle robots join
# every planner call as agents whose goal is their current cell (never hard
# obstacles). LaCAM/PIBT may displace and return such agents (their paths
# are written back and the timestep walks them); PP plans them like any
# agent. Exists so the pre-fix behavior can be reproduced for
# measurement; leave True.
IDLE_AGENTS_IN_PLAN = True


def _nearest_free_cell(source, obstacles, taken, max_radius=40):
    """BFS outward from `source` for the nearest cell that is free (not an
    obstacle, not in `taken`). Returns the cell or None within max_radius."""
    if source is None:
        return None
    q = deque([(source, 0)])
    seen = {source}
    while q:
        (r, c), d = q.popleft()
        if d > max_radius:
            return None
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            cell = (nr, nc)
            if cell in seen or cell in obstacles or cell in taken:
                continue
            if d + 1 > max_radius:
                continue
            seen.add(cell)
            if cell not in obstacles and cell not in taken:
                return cell
            q.append((cell, d + 1))
    return None


# ---------------------------------------------------------------------------
# Shared front-matter: identical to run_cbs, factored out so every planner
# handles stationary blockers and the trivial/duplicate cases the same way.
# ---------------------------------------------------------------------------
def _prepare_agents(robots, start_positions, goal_positions, map_dict=None):
    """
    Returns (agents, stationary_ids) or (None, None) on a fast-fail.

    agents: list of dicts {id, start:(x,y), goal:(x,y)} for every robot that
            should occupy space this round:
              - movers (from start/goal_positions),
              - idle robots, as agents whose goal is their CURRENT cell
                (not obstacles, not stationary). LaCAM/PIBT may displace
                and return them; their paths are written back and the
                timestep walks the displacement. PP plans them like any
                agent (a start==goal agent is a parked agent others route
                around). This keeps idle robots from ever being invisible
                obstacles that movers walk straight through.
              - EXCEPTION: an idle robot standing on a mover's GOAL cell
                (a task can spawn on any free cell, including one an idle
                robot occupies) would make two agents share one goal -
                provably unsolvable, and LaCAM grinds to its time limit
                proving it (the P3v2 pilot's pathological runs were exactly
                this). Such an idle gets a real goal instead: the nearest
                free cell that is no other agent's start or goal, so it
                walks out of the way.
              - assigned robots somehow missing from start_positions are
                still pinned as stationary blockers (legacy run_cbs
                behavior).

    map_dict (optional): {"dimension": (rows, cols), "obstacles": [...]} -
            needed to find the step-aside goal for an idle agent on a
            mover's goal cell.
    """
    if not start_positions or not goal_positions:
        return None, None  # caller returns (0.0, True): nothing to plan

    aug_starts = dict(start_positions)
    aug_goals = dict(goal_positions)
    stationary_ids = set()
    idle_ids_on_goal = []
    for r in robots:
        if r.robot_id in aug_starts:
            continue
        if getattr(r, "location", None) is None:
            continue
        if not (getattr(r, "assigned", False) and getattr(r, "current_task", None) is not None):
            # Unassigned (idle) robot: include it as a stay-put agent whose
            # goal is its current cell. NOT in stationary_ids - its path is
            # written back so it executes any displacement the planner asks
            # of it.
            if IDLE_AGENTS_IN_PLAN:
                aug_starts[r.robot_id] = r.location
                aug_goals[r.robot_id] = r.location
                if r.location in set(goal_positions.values()):
                    idle_ids_on_goal.append(r.robot_id)
            continue
        # Assigned robot somehow missing from start/goal: legacy stationary
        # blocker pin (paths skipped at write-back).
        aug_starts[r.robot_id] = r.location
        aug_goals[r.robot_id] = r.location
        stationary_ids.add(r.robot_id)

    # An idle agent parked on a mover's goal makes the instance unsolvable:
    # two agents can never both end on one cell. Give such an idle a real
    # "step aside" goal (nearest free cell not claimed by any other agent's
    # start or goal) so the instance stays solvable and the idle physically
    # walks out of the way.
    if idle_ids_on_goal and map_dict is not None:
        obstacles = set(map_dict.get("obstacles") or [])
        taken = set(aug_starts.values()) | set(aug_goals.values())
        for rid in idle_ids_on_goal:
            aside = _nearest_free_cell(aug_starts[rid], obstacles, taken)
            if aside is not None:
                aug_goals[rid] = aside
                taken.add(aside)
            # no free cell within reach: leave the stay-put goal; the
            # instance is genuinely unsolvable and the planner fails
            # honestly (replan-on-failure takes over).

    agents = [
        {"id": rid, "start": aug_starts[rid], "goal": aug_goals[rid]}
        for rid in aug_starts
    ]

    starts = [a["start"] for a in agents]
    if len(starts) != len(set(starts)):
        # two agents claim the same cell -- same guard run_cbs uses
        return None, None
    goals = [a["goal"] for a in agents]
    if len(goals) != len(set(goals)):
        # duplicate goals (two movers routed to the same task cell) - same
        # class of unsolvable instance; fail fast instead of letting the
        # engine grind.
        return None, None
    return agents, stationary_ids


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