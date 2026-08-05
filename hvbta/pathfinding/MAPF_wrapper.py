"""
Pure dispatch layer for all MAPF planners.

Public entry point:

    run_planner(planner, robots, start_positions, goal_positions, map_dict, **kw)
        -> (avg_moving_path_length: float, succeeded: bool)

`planner` is a plain string ("cbs" | "pp" | "pibt" | "lacam"). The dispatcher
looks it up in PLANNERS and routes to either the NATIVE-FULL-PLAN wrapper
(run_native) or the PER-STEP / loop-to-completion wrapper (run_stepwise).

Contract out (ALL planners, identical to the old run_cbs so the outer loop is
untouched):
    - writes robot.current_path       = [(x, y), ...]  full path incl. start
    - writes robot.remaining_distance = max(0, len(path) - 1)
    - does NOT overwrite stationary blockers' paths
    - returns (mean path length over MOVING agents, success flag)

Stationary-blocker augmentation is done ONCE, in planner_common._prepare_agents,
for every planner. No engine adapter augments again (double-augmentation would
double-count idle robots). This means CBS now routes movers around idle robots
the same way PP / PIBT / LaCAM do -- which fixes the latent no-op augmentation
in the old run_cbs (build_cbs_agents filtered unassigned robots back out).

Dependency arrows (one-way, no cycles):
    planner_common  <-  each *_planner.py  <-  MAPF_wrapper  <-  main_simulation
Planner modules never import MAPF_wrapper.
"""

from hvbta.pathfinding.planner_common import _prepare_agents, _write_paths_back, _manhattan

# Registry built lazily so an unimplemented planner module (e.g. lacam) doesn't
# break importing this file. Each entry: name -> (kind, callable).
_PLANNER_CACHE = None


def _registry():
    global _PLANNER_CACHE
    if _PLANNER_CACHE is not None:
        return _PLANNER_CACHE
    from hvbta.pathfinding import cbs_planner, prioritized_planner, pibt_planner, lacam_planner
    _PLANNER_CACHE = {
        "cbs":   ("native",   cbs_planner.plan),
        "pp":    ("native",   prioritized_planner.plan),
        "lacam": ("native",   lacam_planner.plan),
        "pibt":  ("stepwise", pibt_planner.step),
    }
    return _PLANNER_CACHE


def run_planner(planner, robots, start_positions, goal_positions, map_dict, **kw):
    registry = _registry()
    if planner not in registry:
        raise ValueError(f"unknown planner {planner!r}; "
                         f"choose from {list(registry)}")
    kind, fn = registry[planner]
    if kind == "native":
        return run_native(fn, robots, start_positions, goal_positions, map_dict)
    return run_stepwise(fn, robots, start_positions, goal_positions, map_dict, **kw)


# ===========================================================================
# (1) NATIVE-FULL-PLAN wrapper  -- CBS / Prioritized Planning / LaCAM.
#     The engine returns whole paths in one call, so this is thin.
# ===========================================================================
def run_native(plan_fn, robots, start_positions, goal_positions, map_dict):
    """
    plan_fn(agents, map_dict) -> {robot_id: [(x,y), ...]} | None
        `agents` is ALREADY augmented with stationary blockers; the engine must
        not augment again. None / {} == failure.
    """
    agents, stationary_ids = _prepare_agents(robots, start_positions, goal_positions)
    if agents is None:
        # nothing to plan -- mirrors old run_cbs's trivial-success branch so
        # pending event latencies still get measured against "next moment
        # planner did not fail."
        return 0.0, True

    paths = plan_fn(agents, map_dict)
    if not paths:
        print(f"{getattr(plan_fn, '__module__', '?')}.{getattr(plan_fn, '__name__', 'plan')} "
              f"found no plan under current state.")
        return 0.0, False

    avg_len = _write_paths_back(robots, paths, stationary_ids)
    return avg_len, True


# ===========================================================================
# (2) LOOP-TO-COMPLETION wrapper  -- PIBT (per-step engine).
#     Drives a one-step engine until every mover arrives, assembling full
#     paths. Structurally heavier because of the outer loop + no-progress
#     guard; the write-back is identical.
# ===========================================================================
def run_stepwise(step_fn, robots, start_positions, goal_positions, map_dict,
                 max_internal_steps=None, no_progress_limit=None):
    """
    Wraps a per-timestep rule (PIBT) so this function returns full paths like
    run_cbs. Internally advances a private clock, calling the one-step rule
    each tick, until all movers arrive or a cap/stall fires.

    step_fn signature:
        step(pos, goal, obstacles, dimension, priority, pinned) -> {aid: cell} | None
      Returns the next cell for every agent, respecting:
        * no two agents to the same cell,
        * no head-on swaps,
        * priority inheritance down the chain,
        * pinned agents (stationary blockers) never yield.
    """
    agents, stationary_ids = _prepare_agents(robots, start_positions, goal_positions)
    if agents is None:
        return 0.0, True

    dim = map_dict["dimension"]
    if max_internal_steps is None:
        max_internal_steps = 4 * (dim[0] + dim[1])   # generous; tune per map
    if no_progress_limit is None:
        no_progress_limit = 2 * (dim[0] + dim[1])

    pos = {a["id"]: a["start"] for a in agents}
    goal = {a["id"]: a["goal"] for a in agents}
    trail = {a["id"]: [a["start"]] for a in agents}

    # PIBT requires a strict total order; ties broken by id via sorted().
    def priority(aid):
        return _manhattan(pos[aid], goal[aid])   # higher = plans first

    stationary = set(stationary_ids)

    def all_arrived():
        return all(pos[aid] == goal[aid] for aid in pos if aid not in stationary)

    prev_remaining = sum(
        1 for aid in pos if aid not in stationary and pos[aid] != goal[aid]
    )
    stall = 0

    for _tick in range(max_internal_steps):
        if all_arrived():
            break

        next_pos = step_fn(
            pos=pos, goal=goal, obstacles=map_dict["obstacles"],
            dimension=dim, priority=priority,
            pinned=stationary,       # never yield these
        )
        if next_pos is None:
            print("stepwise engine reported no valid configuration.")
            return 0.0, False

        for aid, cell in next_pos.items():
            pos[aid] = cell
            if aid not in stationary:
                trail[aid].append(cell)

        remaining = sum(
            1 for aid in pos if aid not in stationary and pos[aid] != goal[aid]
        )
        stall = stall + 1 if remaining >= prev_remaining else 0
        prev_remaining = remaining
        if stall >= no_progress_limit:
            print("stepwise engine livelock: no progress within limit -> failing out.")
            return 0.0, False
    else:
        if not all_arrived():
            print("stepwise engine hit max_internal_steps before all agents arrived.")
            return 0.0, False

    avg_len = _write_paths_back(robots, trail, stationary_ids)
    return avg_len, True
