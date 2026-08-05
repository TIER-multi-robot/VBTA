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
    """
    id_to_index = {r.robot_id: i for i, r in enumerate(robots)}
    valid_lengths = []
    for rid, path in paths.items():
        if rid not in id_to_index:
            continue
        if rid in stationary_ids:
            continue  # don't clobber a blocker's (empty) path with a wait schedule
        r = robots[id_to_index[rid]]
        r.current_path = [(x, y) for (x, y) in path]
        r.remaining_distance = max(0, len(path) - 1)
        if len(path) > 1:
            valid_lengths.append(len(path) - 1)
    return float(np.mean(valid_lengths)) if valid_lengths else 0.0


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])