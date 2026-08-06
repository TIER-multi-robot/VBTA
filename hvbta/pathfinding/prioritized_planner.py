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
import random as _random


def _space_time_astar(start, goal, rows, cols, obstacles,
                      vertex_res, edge_res, parked, H):
    """
    Time-expanded A* over states (t, (r, c)).

    vertex_res : set of (t, (r,c)) already reserved by prior agents.
    edge_res   : set of (t, from_cell, to_cell) -- a prior agent traversed
                 from->to at time t; blocks a swap (to->from at time t).
    parked     : dict cell -> earliest_t blocked for all t >= earliest_t
                 (stationary blockers + arrived movers staying at goal).
    H          : time horizon; agent must reach a parkable goal state by t <= H.

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


def _try_order(movers, blocker_paths, blocker_parked, obstacles, rows, cols, H):
    """Plan all movers in the given order. Returns {id: path} or None."""
    paths = dict(blocker_paths)
    vertex_res = set()
    edge_res = set()
    parked = dict(blocker_parked)

    for a in movers:
        start, goal = a["start"], a["goal"]
        path = _space_time_astar(start, goal, rows, cols, obstacles,
                                 vertex_res, edge_res, parked, H)
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

    # Build 10 priority orders: descending distance, ascending distance,
    # then 8 random shuffles. Different orders produce different reservation
    # patterns; some succeed where others fail.
    rng = _random.Random(42)
    orders = [
        sorted(movers, key=_dist, reverse=True),
        sorted(movers, key=_dist),
    ]
    for _ in range(8):
        shuffled = list(movers)
        rng.shuffle(shuffled)
        orders.append(shuffled)

    for order in orders:
        result = _try_order(order, blocker_paths, blocker_parked,
                           obstacles, rows, cols, H)
        if result is not None:
            return result
    return None
