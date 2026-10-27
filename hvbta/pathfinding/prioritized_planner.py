"""
Prioritized Planning engine behind the NATIVE-FULL-PLAN contract.

    plan(agents, map_dict) -> {robot_id: [(x,y), ...]} | None

Algorithm: space-time A* per agent in priority order. Already-planned paths
become time-indexed reservations (vertex + edge). Stationary blockers (agents
with start == goal, i.e. idle robots augmented by _prepare_agents) reserve
their cell for ALL time so movers route around them. After a mover reaches its
goal it is "parked" -- its goal cell is reserved for all future timesteps.

PP is incomplete: a single priority order can fail when later agents can't
find conflict-free paths. To mitigate, plan() tries MULTIPLE priority orders
(descending Manhattan distance, ascending, then 8 random shuffles) and returns
the first that succeeds. Since PP's per-agent A* is fast, 10 attempts add
negligible overhead.

`agents` is ALREADY augmented with stationary blockers by _prepare_agents;
this adapter must NOT augment again. Return None on failure (all attempts
exhausted), which the wrapper reports as (0.0, False)).
"""

import heapq
import os as _os
import random as _random

# Deterministic per-call work budget (2026-10-26): total A* expansions
# allowed per plan() call, SHARED across every priority order and every
# agent search in that call. PP has no wall-clock limit by design (the
# wrapper threads time limits only to LaCAM); its pathological case is the
# exhaustive failing search - a single doomed call on a hyper-constrained
# map burns up to ~10 s (millions of expansions) before returning None.
# The budget converts that into a count-bounded early give-up: when the
# budget is exhausted, remaining searches fail immediately. Deterministic
# (same seeds, same orders, same truncation); it can only turn a would-be
# slow failure into a fast one - observed successful plan() calls use well
# under 30k expansions total on the sweep maps. None disables the budget.
# Value (2026-10-26, hyper-PP pilot measurement): 1.4x the largest expansion
# count observed for a SUCCESSFUL call (178k, lak108c 10x15); failing calls
# were observed at up to 3.3M expansions (~8 s each). Env-overridable
# (HVBTA_PP_MAX_EXPANSIONS) so the cap-sensitivity re-run (500k arm) uses
# identical code.
PP_MAX_EXPANSIONS = int(_os.environ.get("HVBTA_PP_MAX_EXPANSIONS", "250000"))

# Per-RUN expansion budget (2026-10-26): total A* expansions allowed across
# ALL plan() calls of one simulation run. The per-call budget above cannot
# alone bound a run's wall time under pool load (30 workers slow pure-Python
# search ~3x to ~100k expansions/s, so 150 failed calls x 400k expansions
# still reached the 600 s cap in the first pilot re-run). This budget is the
# hard count-based bound: once exhausted, every further plan() call fails
# immediately. Sized from the pilot re-run: normal heavy hyper runs use
# ~15-25M expansions (63-failure Murty's run: 25M at the 400k per-call cap,
# ~16M at 250k); only the >350 s tail (~1% of runs) exceeds 35M. Reset per
# run by MAPF_wrapper.reset_planner_timing. Env-overridable
# (HVBTA_PP_RUN_MAX_EXPANSIONS).
PP_RUN_MAX_EXPANSIONS = int(_os.environ.get("HVBTA_PP_RUN_MAX_EXPANSIONS",
                                           "35000000"))
PP_RUN_EXPANSIONS_USED = 0

# Optional per-call diagnostics (calibration driver only): when set to a
# list, plan() appends one dict per call (total expansions, agent count,
# success flag, budget-hit flag). Inert by default.
PP_CALL_LOG = None

# Cumulative count of plan() calls whose expansion budget was exhausted
# (work truncated). Reset/read by the caller alongside PLANNER_TIMING.
PP_EXPANSION_CAP_HITS = 0


def _space_time_astar(start, goal, rows, cols, obstacles,
                      vertex_res, edge_res, parked, H, budget=None):
    """
    Time-expanded A* over states (t, (r, c)).

    vertex_res : set of (t, (r,c)) already reserved by prior agents.
    edge_res   : set of (t, from_cell, to_cell) -- a prior agent traversed
                 from->to at time t; blocks a swap (to->from at time t).
    parked     : dict cell -> earliest_t blocked for all t >= earliest_t
                 (stationary blockers + arrived movers staying at goal).
    H          : time horizon; agent must reach a parkable goal state by t <= H.
    budget     : optional one-element list [remaining_expansions]; when it
                 reaches 0 the search gives up (returns None). Shared across
                 all searches of one plan() call.

    Returns a list of cells [(r,c), ...] start..goal, or None.
    """
    h = lambda c: abs(c[0] - goal[0]) + abs(c[1] - goal[1])

    # If a prior agent is already parked at this goal, we can never complete.
    if goal in parked:
        return None

    # Latest timestep at which the goal cell is reserved by a passing prior
    # agent. We must arrive strictly after this to be able to park.
    goal_max_res = -1
    for (tt, cc) in vertex_res:
        if cc == goal and tt > goal_max_res:
            goal_max_res = tt

    start_state = (0, start)
    g = {start_state: 0}
    came = {}
    open_heap = [(h(start), 0, 0, start)]  # (f, t, tiebreak, cell)
    counter = 0

    moves = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]  # wait + 4-connected

    while open_heap:
        f, t, _, cell = heapq.heappop(open_heap)
        if cell == goal and t > goal_max_res:
            # reachable AND parkable (no future reservation on goal). Reconstruct.
            path = [cell]
            s = (t, cell)
            while s in came:
                s = came[s]
                path.append(s[1])
            path.reverse()
            return path
        if t >= H:
            continue
        if budget is not None:
            if budget[0] <= 0:
                return None
            budget[0] -= 1
        for dr, dc in moves:
            nr, nc = cell[0] + dr, cell[1] + dc
            ncell = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if ncell in obstacles:
                continue
            nt = t + 1
            # vertex blocked?
            if (nt, ncell) in vertex_res:
                continue
            if ncell in parked and nt >= parked[ncell]:
                continue
            # edge / swap blocked? (someone moved ncell->cell at time t)
            if (t, ncell, cell) in edge_res:
                continue
            ng = g[(t, cell)] + 1
            nstate = (nt, ncell)
            if nstate in g and ng >= g[nstate]:
                continue
            g[nstate] = ng
            came[nstate] = (t, cell)
            counter += 1
            heapq.heappush(open_heap, (ng + h(ncell), nt, counter, ncell))
    return None


def _try_order(movers, blocker_paths, blocker_parked, obstacles, rows, cols, H,
               budget=None):
    """Plan all movers in the given order. Returns {id: path} or None."""
    paths = dict(blocker_paths)
    vertex_res = set()
    edge_res = set()
    parked = dict(blocker_parked)

    for a in movers:
        start, goal = a["start"], a["goal"]
        path = _space_time_astar(start, goal, rows, cols, obstacles,
                                 vertex_res, edge_res, parked, H, budget)
        if path is None:
            return None
        for t, cell in enumerate(path):
            vertex_res.add((t, cell))
        for t in range(len(path) - 1):
            edge_res.add((t, path[t], path[t + 1]))
        arrival = len(path) - 1
        parked[goal] = max(parked.get(goal, 0), arrival)
        paths[a["id"]] = path
    return paths


def plan(agents, map_dict):
    rows, cols = map_dict["dimension"]
    obstacles = set(map_dict["obstacles"])

    movers = []
    blocker_paths = {}
    blocker_parked = {}
    for a in agents:
        if a["start"] == a["goal"]:
            blocker_paths[a["id"]] = [a["start"]]
            blocker_parked[a["start"]] = 0
        else:
            movers.append(a)

    if not movers:
        return blocker_paths

    def _dist(a):
        return abs(a["start"][0] - a["goal"][0]) + abs(a["start"][1] - a["goal"][1])

    n = len(movers)
    H = max((rows + cols) * (n + 1) + 16, 64)

    # Build 4 priority orders: descending distance, ascending distance,
    # then 2 random shuffles. Reduced from 10 to 4 for the hyper-constrained
    # sweep where PP failures are expected (fewer restarts = higher failure
    # rate = more replan-on-failure events to analyze). The multi-restart
    # still provides some robustness but fails more often than the 10-restart
    # version, which is the desired behavior for testing the replan mechanism.
    rng = _random.Random(42)
    orders = [
        sorted(movers, key=_dist, reverse=True),
        sorted(movers, key=_dist),
    ]
    for _ in range(2):
        shuffled = list(movers)
        rng.shuffle(shuffled)
        orders.append(shuffled)

    # Shared per-call work budget. When PP_MAX_EXPANSIONS is None the budget
    # is unlimited (the sentinel is finite but astronomically large so the
    # counter still measures real work for PP_CALL_LOG). The per-call budget
    # is capped by the run budget's remaining allowance.
    global PP_EXPANSION_CAP_HITS, PP_RUN_EXPANSIONS_USED
    remaining_run = (PP_RUN_MAX_EXPANSIONS - PP_RUN_EXPANSIONS_USED
                     if PP_RUN_MAX_EXPANSIONS is not None else 10 ** 18)
    if PP_MAX_EXPANSIONS is not None:
        budget = [max(0, min(int(PP_MAX_EXPANSIONS), remaining_run))]
    else:
        budget = [max(0, remaining_run)]
    _start_budget = budget[0]
    if remaining_run <= 0:
        # Run budget exhausted: this call fails instantly (searches never
        # start). Counted as a cap hit like any truncation.
        PP_EXPANSION_CAP_HITS += 1
        if PP_CALL_LOG is not None:
            PP_CALL_LOG.append({"expansions": 0, "agents": len(agents),
                                "movers": len(movers), "ok": False,
                                "budget_hit": True})
        return None

    for order in orders:
        result = _try_order(order, blocker_paths, blocker_parked,
                           obstacles, rows, cols, H, budget)
        if result is not None:
            PP_RUN_EXPANSIONS_USED += _start_budget - budget[0]
            if PP_CALL_LOG is not None:
                PP_CALL_LOG.append({"expansions": _start_budget - budget[0],
                                    "agents": len(agents), "movers": len(movers),
                                    "ok": True, "budget_hit": budget[0] <= 0})
            return result
    budget_hit = budget[0] <= 0
    PP_RUN_EXPANSIONS_USED += _start_budget - budget[0]
    if budget_hit:
        PP_EXPANSION_CAP_HITS += 1
    if PP_CALL_LOG is not None:
        PP_CALL_LOG.append({"expansions": _start_budget - budget[0],
                            "agents": len(agents), "movers": len(movers),
                            "ok": False, "budget_hit": budget_hit})
    return None
