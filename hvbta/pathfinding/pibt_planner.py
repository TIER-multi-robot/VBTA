"""
PIBT one-step engine behind the PER-STEP contract.

    step(pos, goal, obstacles, dimension, priority, pinned) -> {aid: cell} | None

Stub (Phase 4 wiring). Real implementation (Phase 8): one PIBT tick with
priority inheritance. The priority-inheritance recursion is the subtle part;
unit-test in isolation: two agents contesting one cell, a yield chain of 3,
and a pinned stationary blocker that must NOT yield. Then run_planner("pibt",
...) works via the stepwise driver in MAPF_wrapper.

Contract:
  * `pos`     : {aid: (x,y)} current cells of every agent (movers + pinned).
  * `goal`    : {aid: (x,y)} goal cells.
  * `obstacles`: list of blocked cells.
  * `dimension`: (rows, cols).
  * `priority`: callable aid -> sortable key (higher plans first).
  * `pinned`  : set of aids that must NEVER move (stationary blockers).
  Returns {aid: next_cell} for every agent, or None if no conflict-free
  config exists.
"""


def step(pos, goal, obstacles, dimension, priority, pinned):
    raise NotImplementedError("pibt_planner.step not implemented yet (Phase 8).")
