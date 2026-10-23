import numpy as np

# ---------------------------------------------------------------------------
# Shared front-matter: identical to run_cbs, factored out so every planner
# handles stationary blockers and the trivial/duplicate cases the same way.
# ---------------------------------------------------------------------------
def _prepare_agents(robots, start_positions, goal_positions):
    """
    Returns (agents, stationary_ids) or (None, None) on a fast-fail.

    agents: list of dicts {id, start:(x,y), goal:(x,y)} for every robot that
            should occupy space this round -- movers (from start/goal_positions)
            plus idle robots pinned as start==goal blockers, exactly as run_cbs
            augments them so the planner routes movers around idle robots.

    Only robots that are ASSIGNED and working on a task are pinned as stationary
    blockers. Unassigned (idle) robots are NOT pinned — they should be able to
    get out of the way of movers. Pinning idle robots can disconnect the graph
    and cause the planner to fail when an idle robot is sitting in a corridor
    or choke point that a mover needs to pass through.
    """
    if not start_positions or not goal_positions:
        return None, None  # caller returns (0.0, True): nothing to plan

    aug_starts = dict(start_positions)
    aug_goals = dict(goal_positions)
    stationary_ids = set()
    for r in robots:
        if r.robot_id in aug_starts:
            continue
        if getattr(r, "location", None) is None:
            continue
        # Only pin robots that are actively working on a task — they have a
        # goal and are in transit or working. Idle robots (no task) should
        # not be blockers; they can be pushed out of the way by movers.
        if not (getattr(r, "assigned", False) and getattr(r, "current_task", None) is not None):
            continue
        aug_starts[r.robot_id] = r.location
        aug_goals[r.robot_id] = r.location
        stationary_ids.add(r.robot_id)

    agents = [
        {"id": rid, "start": aug_starts[rid], "goal": aug_goals[rid]}
        for rid in aug_starts
    ]

    starts = [a["start"] for a in agents]
    if len(starts) != len(set(starts)):
        # two agents claim the same cell -- same guard run_cbs uses
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