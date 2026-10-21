"""
PIBT one-step engine behind the PER-STEP contract.

    step(pos, goal, obstacles, dimension, priority, pinned) -> {aid: cell} | None

PIBT (Priority Inheritance with Backtracking, Okumura 2019) is a one-timestep
rule. Agents are processed in priority order (highest first); each agent tries
to move to its preferred next cell (toward goal). If that cell is held by a
lower-priority undecided agent, priority *inherits* down to push that agent out
of the way (recursion). If an agent can't find any valid move it *backtracks*
and stays put. Pinned agents (stationary blockers) never move and have top
priority so no one can push them.

Guarantees of one tick:
  * no two agents to the same cell (vertex conflict free),
  * no head-on swaps (edge conflict free),
  * pinned agents stay put,
  * each non-pinned agent gets a next cell (possibly its current cell).

Used two ways:
  1. Driven to completion by MAPF_wrapper.run_stepwise (loop until all arrive).
  2. As the successor generator inside lacam_planner.plan.

`priority` is a callable aid -> sortable key (higher = plans first). Ties are
broken by id to give a strict total order (required so the inheritance
recursion can't loop).
"""


def step(pos, goal, obstacles, dimension, priority, pinned, rng=None, tabu=None,
         blocked=None):
    """
    rng: optional random.Random instance. When provided, ties in the priority
    order AND in the candidate-cell preference are broken randomly, so repeated
    calls on the same state can yield different successors. This is what LaCAM
    uses to get a real branching factor at each high-level node. When None,
    ties are broken deterministically (by id) -- used by the run_stepwise
    driver and for unit tests.

    tabu: optional dict {aid: set_of_cells}. Cells in an agent's tabu set are
    treated as LAST-RESORT candidates (tried after all other moves including
    wait). This prevents oscillation in dead-end corridors.

    blocked: optional dict {aid: set_of_cells}. Cells in an agent's blocked
    set are HARD constraints — the agent CANNOT move there at all. This is
    used by LaCAM's LCA mechanism to force an agent off a cell it's cycling
    on. Unlike tabu (soft preference), blocked cells are never considered.
    """
    aids = list(pos.keys())
    obs_set = set(obstacles)
    rows, cols = dimension
    pinned = set(pinned)
    tabu = tabu or {}
    blocked = blocked or {}

    # Strict total order key. Pinned get +inf so they decide first and never
    # get pushed. Ties broken by id (descending) when rng is None, else by a
    # per-call random jitter so equal-priority agents decide in random order.
    _jitter = {} if rng is None else {aid: rng.random() for aid in aids}

    def pkey(aid):
        if aid in pinned:
            return (float("inf"), _jitter.get(aid, 0.0), aid)
        return (priority(aid), _jitter.get(aid, 0.0), aid)

    order = sorted(aids, key=pkey, reverse=True)  # highest first

    decided = {}                       # aid -> next cell
    current_occupant = {c: aid for aid, c in pos.items()}  # cell -> aid there now

    def decide(a):
        if a in decided:
            return True
        if a in pinned:
            decided[a] = pos[a]
            return True
        cur = pos[a]
        g = goal[a]
        at_goal = (cur == g)

        # Candidate cells: 4-connected moves + wait. Validity filtered.
        my_blocked = blocked.get(a, set())
        cands = []
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            nr, nc = cur[0] + dr, cur[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in obs_set:
                if (nr, nc) in my_blocked:
                    continue  # hard LCA constraint — can't go here
                cands.append((nr, nc))
        if cur not in my_blocked:
            cands.append(cur)  # wait (unless blocked from staying)

        # Preference: among moves, closer to goal first. Waiting is the
        # ABSOLUTE last choice (taken only when no move works) unless the agent
        # is already at its goal -- this is the PIBT-R variant that escapes
        # head-on deadlocks in open maps by detouring rather than waiting.
        # TABU cells (recently visited) are pushed just before wait — the agent
        # avoids going back where it just came from unless there's no other
        # option. This breaks dead-end-corridor oscillation.
        # When rng is provided, equal-preference cells are randomly ordered so
        # repeated calls yield different successors (LaCAM branching).
        my_tabu = tabu.get(a, set())
        def pref(c):
            d = abs(c[0] - g[0]) + abs(c[1] - g[1])
            is_wait = (c == cur)
            is_tabu = (c in my_tabu and c != g)
            if at_goal:
                return (2 if is_tabu else 0 if is_wait else 1, d)
            return (3 if is_tabu else 1 if is_wait else 0, d)
        cands.sort(key=pref)
        if rng is not None:
            # Stable shuffle within groups of equal `pref` value, so two
            # equal-distance detours (e.g. left vs right around a blocker) are
            # picked at random instead of always the same one.
            i = 0
            while i < len(cands):
                j = i + 1
                while j < len(cands) and pref(cands[j]) == pref(cands[i]):
                    j += 1
                if j > i + 1:
                    block = cands[i:j]
                    rng.shuffle(block)
                    cands[i:j] = block
                i = j

        for c in cands:
            # vertex: no decided agent already claims c
            if any(decided[b] == c for b in decided):
                continue

            if c == cur:
                # waiting: always physically valid; the "someone higher-prio
                # already moving into cur" case can't arise because higher-prio
                # agents decided earlier and would have pushed us first.
                decided[a] = c
                return True

            # moving cur -> c
            # swap with a decided agent b whose pos is c and next is cur
            swap = False
            for b, nb in decided.items():
                if b != a and pos[b] == c and nb == cur:
                    swap = True
                    break
            if swap:
                continue

            occ = current_occupant.get(c)
            if occ is None or occ == a:
                # free cell
                decided[a] = c
                return True

            # c is occupied by `occ`
            if occ in decided:
                if decided[occ] == c:
                    continue                # occ staying at c -> blocked
                if decided[occ] == cur:
                    continue                # occ moving c->cur -> swap
                decided[a] = c              # occ left -> c is free now
                return True
            else:
                # occ undecided -- only push strictly-lower-priority agents.
                if pkey(occ) >= pkey(a):
                    continue
                ok = decide(occ)            # priority inheritance
                if not ok:
                    continue
                if decided[occ] == c:
                    continue                # occ couldn't leave c
                if decided[occ] == cur:
                    continue                # swap
                decided[a] = c
                return True

        # no candidate worked -> backtrack: stay put.
        if any(decided[b] == cur for b in decided if b != a):
            # a higher-prio agent already claimed our cell; can't stay -> fail.
            return False
        decided[a] = cur
        return True

    for a in order:
        if a not in decided:
            if not decide(a):
                return None

    # safety: any agent not decided (e.g. pinned missed by the loop) stays put
    for a in aids:
        if a not in decided:
            decided[a] = pos[a]
    return decided
