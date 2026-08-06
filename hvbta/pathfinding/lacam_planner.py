"""
LaCAM engine behind the NATIVE-FULL-PLAN contract.

    plan(agents, map_dict) -> {robot_id: [(x,y), ...]} | None

LaCAM (Lazy Constraints Addition for MAPF; Okumura 2023) is a complete,
fast planner. This implementation is a configuration-space best-first search
that uses ``pibt_planner.step`` as its successor generator. To get branching
(and therefore completeness beyond PIBT's myopia), each expanded configuration
generates SEVERAL successor configurations by running PIBT under a small set of
deterministic priority orderings -- different orderings yield different next
configurations, so the search escapes the symmetric deadlocks / oscillations
that pure PIBT falls into (e.g. two movers contesting one detour cell around a
stationary blocker).

`agents` is ALREADY augmented with stationary blockers by _prepare_agents; this
adapter must NOT augment again. Stationary agents (start == goal) are pinned so
PIBT never moves them. Returns full per-agent paths (start cell included), or
None when the node/horizon budget is exhausted (the wrapper then reports
(0.0, False)).
"""

import heapq
import random as _random


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def plan(agents, map_dict):
    from hvbta.pathfinding.pibt_planner import step as pibt_step

    rows, cols = map_dict["dimension"]
    obstacles = set(map_dict["obstacles"])
    aids = [a["id"] for a in agents]
    n = len(aids)
    starts = {a["id"]: a["start"] for a in agents}
    goals = {a["id"]: a["goal"] for a in agents}
    stationary = frozenset(a for a in aids if starts[a] == goals[a])

    init_cfg = tuple(starts[a] for a in aids)
    goal_cfg = tuple(goals[a] for a in aids)
    if init_cfg == goal_cfg:
        return {a: [starts[a]] for a in aids}

    def cfg_to_pos(cfg):
        return {aids[i]: cfg[i] for i in range(n)}

    def h(cfg):
        return sum(_manhattan(cfg[i], goal_cfg[i]) for i in range(n))

    # Branching: at each node, try B randomized PIBT calls with distinct seeds.
    # Randomized tie-breaking (both in priority order and in equal-cost move
    # selection -- see pibt_planner.step) yields DIFFERENT successors across
    # calls, which is what gives LaCAM a real branching factor and lets it
    # escape the symmetric deadlocks pure PIBT falls into.
    BRANCH = 4
    base_seed = 0x5EED ^ hash(tuple(aids))

    H = max((rows + cols) * (n + 1) + 32, 64)
    NODE_BUDGET = max(100000, 2000 * n)

    counter = 0
    # start_node = heuristic, counter, depth, config, plan
    start_node = (h(init_cfg), counter, 0, init_cfg, [init_cfg])
    open_heap = [start_node]
    best_depth = {init_cfg: 0}
    budget = 0

    while open_heap:
        f, _, depth, cfg, plan = heapq.heappop(open_heap)
        if best_depth.get(cfg, 1 << 30) < depth:
            continue
        budget += 1
        if budget > NODE_BUDGET:
            return None
        if cfg == goal_cfg:
            return _build_paths(plan, aids, stationary)
        if depth >= H:
            continue

        pos = cfg_to_pos(cfg)
        seen_succ = set()
        for b in range(BRANCH):
            rng = _random.Random(hash((base_seed, budget, b)) & 0xFFFFFFFF)
            # base priority by distance; rng handles the tie-breaking inside PIBT
            priority = lambda aid, _pos=pos: _manhattan(_pos[aid], goals[aid])
            nxt = pibt_step(
                pos=pos, goal=goals, obstacles=obstacles,
                dimension=(rows, cols), priority=priority,
                pinned=set(stationary), rng=rng,
            )
            if nxt is None:
                continue
            ncfg = tuple(nxt[a] for a in aids)
            if ncfg == cfg or ncfg in seen_succ:
                continue
            seen_succ.add(ncfg)
            nd = depth + 1
            if best_depth.get(ncfg, 1 << 30) <= nd:
                continue
            best_depth[ncfg] = nd
            counter += 1
            heapq.heappush(
                open_heap,
                (nd + h(ncfg), counter, nd, ncfg, plan + [ncfg]),
            )
    return None


def _build_paths(plan, aids, stationary):
    """plan: list of configurations (one per timestep, plan[0] = starts).
    Returns {aid: [(x,y), ...]} with the full path including the start cell.
    Stationary agents get a single-cell path (wrapper skips them anyway)."""
    paths = {}
    for i, aid in enumerate(aids):
        if aid in stationary:
            paths[aid] = [plan[0][i]]
        else:
            paths[aid] = [cfg[i] for cfg in plan]
    return paths
