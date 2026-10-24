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

# Per-call planner time budget (seconds). Engines that accept a time limit
# (LaCAM, CBS) get this passed; a budget hit is a FAILURE classified as
# "lacam_timeout" (or "engine_unsolved" for engines without limits) - never
# treated as proof of infeasibility. Configurable.
PLANNER_TIME_BUDGET_S = 1.0

# Registry built lazily so an unimplemented planner module (e.g. lacam) doesn't
# break importing this file. Each entry: name -> (kind, callable).
_PLANNER_CACHE = None

# Phase 3 cascade diagnostics: set on every run_native call - None on
# success, otherwise a dict describing WHY the call failed:
#   kind: "duplicate_start" | "duplicate_goal" | "lacam_no_solution" |
#         "lacam_timeout" | "engine_unsolved"
# plus, for the duplicate cases, the agents involved. Pure logging; no
# decision path reads it.
LAST_PLAN_FAILURE = None


def _classify_failure(start_positions, goal_positions, agents, plan_fn):
    """Build the LAST_PLAN_FAILURE dict for a failed native plan call."""
    starts = list(start_positions.values())
    goals = list(goal_positions.values())
    if len(starts) != len(set(starts)):
        dup = [s for s in set(starts) if starts.count(s) > 1]
        involved = [{"id": rid, "start": sp, "goal": goal_positions[rid]}
                    for rid, sp in start_positions.items()
                    if sp in dup]
        return {"kind": "duplicate_start", "cells": [c for c in dup], "agents": involved}
    if len(goals) != len(set(goals)):
        dup = [g for g in set(goals) if goals.count(g) > 1]
        involved = [{"id": rid, "start": start_positions[rid], "goal": gp}
                    for rid, gp in goal_positions.items()
                    if gp in dup]
        return {"kind": "duplicate_goal", "cells": [c for c in dup], "agents": involved}
    # engine-level failure with clean inputs: ask the engine why (lacam
    # reports its reason; other engines get the generic label)
    if getattr(plan_fn, "__module__", "").endswith("lacam_planner"):
        from hvbta.pathfinding import lacam_planner
        reason = lacam_planner.LAST_FAILURE_REASON
        if reason == "timeout":
            return {"kind": "lacam_timeout", "reason": reason}
        if reason is not None and reason.startswith("solved=0"):
            # Distinguish a budget hit (LaCAM self-terminated at its time
            # limit) from a genuine no-solution (it finished well under
            # the limit). Parse comp_time and limit from the reason string.
            import re as _re
            m_ct = _re.search(r"comp_time=(\d+)", reason)
            m_lim = _re.search(r"limit=(\d+)", reason)
            comp_time = int(m_ct.group(1)) if m_ct else -1
            limit = int(m_lim.group(1)) if m_lim else -1
            if limit > 0 and comp_time >= 0.9 * limit:
                return {"kind": "lacam_timeout", "reason": reason}
            return {"kind": "lacam_no_solution", "reason": reason}
        return {"kind": "engine_unsolved", "reason": reason}
    return {"kind": "engine_unsolved", "reason": None}


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
    import time as _time
    registry = _registry()
    if planner not in registry:
        raise ValueError(f"unknown planner {planner!r}; "
                         f"choose from {list(registry)}")
    kind, fn = registry[planner]
    _t0 = _time.perf_counter_ns()
    try:
        if kind == "native":
            return run_native(fn, robots, start_positions, goal_positions, map_dict)
        return run_stepwise(fn, robots, start_positions, goal_positions, map_dict, **kw)
    finally:
        PLANNER_TIMING["ns"] += _time.perf_counter_ns() - _t0
        PLANNER_TIMING["calls"] += 1


# Per-run planner-call wall-time accounting (Phase 3v2: planning idle
# robots as stay-put agents costs time, so measure it). Reset/read by
# main_simulation per run. agents_sum accumulates the agent count of every
# successful _prepare_agents call for mean-agents-per-call reporting.
PLANNER_TIMING = {"calls": 0, "ns": 0, "agents_sum": 0}


def reset_planner_timing():
    PLANNER_TIMING["calls"] = 0
    PLANNER_TIMING["ns"] = 0
    PLANNER_TIMING["agents_sum"] = 0


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
    global LAST_PLAN_FAILURE
    LAST_PLAN_FAILURE = None
    agents, stationary_ids, failure = _prepare_agents(robots, start_positions, goal_positions, map_dict)
    if agents is None:
        if start_positions and goal_positions:
            # Input-level failure (duplicate start/goal after repair, or
            # genuinely nothing to plan): classify from _prepare_agents,
            # which saw the FULL agent set including idle agents.
            LAST_PLAN_FAILURE = failure or {"kind": "engine_unsolved", "reason": "no_agents"}
            return 0.0, False
        # genuinely nothing to plan (empty start_positions)
        return 0.0, True

    PLANNER_TIMING["agents_sum"] += len(agents)

    # Thread the per-call time budget to LaCAM (the subprocess engine whose
    # runs are unbounded without it). CBS keeps its own 30s internal limit;
    # PP/PIBT are fast and have no limit parameter.
    if getattr(plan_fn, "__module__", "").endswith("lacam_planner"):
        paths = plan_fn(agents, map_dict, time_limit=PLANNER_TIME_BUDGET_S)
    else:
        paths = plan_fn(agents, map_dict)

    if not paths:
        print(f"{getattr(plan_fn, '__module__', '?')}.{getattr(plan_fn, '__name__', 'plan')} "
              f"found no plan under current state.")
        LAST_PLAN_FAILURE = _classify_failure(start_positions, goal_positions, agents, plan_fn)
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
                 max_internal_steps=None, no_progress_limit=None, **kw):
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
    agents, stationary_ids, failure = _prepare_agents(robots, start_positions, goal_positions, map_dict)
    if agents is None:
        if failure and start_positions and goal_positions:
            LAST_PLAN_FAILURE = failure
            return 0.0, False
        return 0.0, True

    PLANNER_TIMING["agents_sum"] += len(agents)

    dim = map_dict["dimension"]
    if max_internal_steps is None:
        # Generous: PIBT is cheap per tick, so allow plenty of steps for
        # agents to detour around each other and obstacles. 200x the map
        # diameter gives even congested instances with obstacles room.
        max_internal_steps = 200 * (dim[0] + dim[1])
    if no_progress_limit is None:
        # Stall = no agent moved for this many consecutive ticks. With
        # obstacles, agents can get stuck in corridors and need many ticks
        # of randomized PIBT to break out. 50x the diameter gives the
        # randomization time to escape corridor deadlocks.
        no_progress_limit = 50 * (dim[0] + dim[1])

    pos = {a["id"]: a["start"] for a in agents}
    goal = {a["id"]: a["goal"] for a in agents}
    trail = {a["id"]: [a["start"]] for a in agents}

    import random as _random
    from collections import deque
    # PIBT requires a strict total order; ties broken by a per-tick random
    # jitter so symmetric deadlocks (e.g. two agents oscillating in a cycle)
    # break instead of livelocking forever. Deterministic given a fixed seed.
    _rng = _random.Random(kw.get("seed", 0))

    # Tabu table: per-agent deque of recently visited cells (last TABU_LEN).
    # Prevents dead-end-corridor oscillation — the standard PIBT fix.
    TABU_LEN = 3
    tabu = {a["id"]: deque(maxlen=TABU_LEN) for a in agents}
    for a in agents:
        tabu[a["id"]].append(a["start"])

    def priority(aid):
        return _manhattan(pos[aid], goal[aid]) + _rng.random() * 0.999

    stationary = set(stationary_ids)

    def all_arrived():
        return all(pos[aid] == goal[aid] for aid in pos if aid not in stationary)

    prev_pos = dict(pos)
    stall = 0

    for _tick in range(max_internal_steps):
        if all_arrived():
            break

        tabu_sets = {aid: set(tabu[aid]) for aid in tabu}
        next_pos = step_fn(
            pos=pos, goal=goal, obstacles=map_dict["obstacles"],
            dimension=dim, priority=priority,
            pinned=stationary,       # never yield these
            rng=_rng,                # randomize PIBT tie-breaks to escape cycles
            tabu=tabu_sets,          # avoid recently-visited cells
        )
        if next_pos is None:
            print("stepwise engine reported no valid configuration.")
            return 0.0, False

        for aid, cell in next_pos.items():
            pos[aid] = cell
            if aid not in stationary:
                trail[aid].append(cell)
                tabu[aid].append(cell)

        # stall = consecutive ticks where NO agent moved. Cycling (agents
        # moving but not arriving) is NOT a stall -- randomization may break
        # it on a later tick.
        moved = any(pos[aid] != prev_pos[aid] for aid in pos if aid not in stationary)
        stall = 0 if moved else stall + 1
        prev_pos = dict(pos)
        if stall >= no_progress_limit:
            print("stepwise engine livelock: no agent moved within limit -> failing out.")
            return 0.0, False
    else:
        if not all_arrived():
            print("stepwise engine hit max_internal_steps before all agents arrived.")
            return 0.0, False

    avg_len = _write_paths_back(robots, trail, stationary_ids)
    return avg_len, True
