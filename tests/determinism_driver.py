"""Determinism test driver (agent instructions Step 3).

Runs a small seeded config for the methods given on the command line (in
the given ORDER) and writes one CSV row per method. Launched as separate
subprocesses by tests/test_partial_observability.py, which asserts:

  1. Cross-process determinism: two subprocesses with the same order
     produce identical CSVs (excluding wall-clock timing columns).
  2. Order independence: running the methods in a different order does
     not change any method's row.

Uses PP (pure Python, no external binary) and the same seeding scheme as
the sweep: entities <- _derived_seed(schedule_seed, "entities"),
initial assignment <- _derived_seed(schedule_seed, "initial", <method>),\nper-run <- _derived_seed(schedule_seed, "run", <method>). Mask level/policy and\nobservation-noise sigma are deliberately NOT part of any run seed.

Usage: python determinism_driver.py <output_csv> <method>,<method>,...
"""
import copy
import csv
import os
import sys
import zlib

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import hvbta.generation as G
import hvbta.suitability as S
import hvbta.allocators.voting as V
import hvbta.allocators.optimizers as O
from hvbta.simulation.main_simulation import (
    benchmark_simulation, build_event_schedule, _derived_seed, _seed_rngs,
)
from hvbta.pathfinding.CBS_helper import create_obstacle_list
from hvbta.partial_observability import PartialObservabilityState

ROWS = COLS = 12
GRID = [[0] * COLS for _ in range(ROWS)]
MAP_DICT = {"dimension": (ROWS, COLS), "obstacles": create_obstacle_list(GRID)}
HYP = (ROWS ** 2 + COLS ** 2) ** 0.5

METHODS = {
    "borda": V.rank_assignments_borda,
    "cumulative": V.rank_assignments_cumulative_voting,
    "murtys": O.murtys_task_allocation,
}

NUM_ROBOTS = 6
NUM_TASKS = 6
NC = 36
SCENARIO = "calm"
REP = 0
MASK_LEVEL, MASK_POLICY = "none", "none"
MAX_STEPS = 80


def main():
    out_csv = sys.argv[1]
    order = sys.argv[2].split(",")

    schedule_seed = zlib.adler32(repr(
        ("det_grid.map", NUM_ROBOTS, NUM_TASKS, NC,
         "evaluate_suitability_balanced", REP, SCENARIO)).encode()) & 0xFFFFFFFF

    # Per-rep entity generation: seeded, so identical in every process.
    _seed_rngs(_derived_seed(schedule_seed, "entities"))
    robot_locs, robots = set(), []
    for i in range(NUM_ROBOTS):
        r = G.generate_random_robot_profile_strict(f"R{i+1}", GRID, robot_locs)
        robots.append(r)
        robot_locs.add(r.location)
    task_locs, tasks = set(), []
    for i in range(NUM_TASKS):
        t = G.generate_random_task_description_strict(f"T{i+1}", GRID, task_locs, [])
        tasks.append(t)
        task_locs.add(t.location)

    event_schedule = build_event_schedule(
        robots, tasks, GRID, MAX_STEPS,
        add_tasks=True, add_robots=True, remove_robots_flag=True, remove_tasks_flag=True,
        task_arrival_rate=0.10, robot_arrival_rate=0.01,
        robot_departure_rate=0.01, task_cancellation_rate=0.01,
        tasks_to_add=2, robots_to_add=1, robots_to_remove=1, tasks_to_remove=1,
        seed=schedule_seed,
    )

    po = PartialObservabilityState(schedule_seed, MASK_LEVEL, MASK_POLICY)
    sm = S.evaluate_suitability_balanced
    matrix = S.calculate_suitability_matrix(robots, tasks, po.masked_scorer(sm), HYP)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method"] + [f"c{i}" for i in range(58)])
        for name in order:
            meth = METHODS[name]
            # Per-method initial-assignment RNG isolation (same scheme as the
            # sweep), then a per-(group, method) run seed.
            _seed_rngs(_derived_seed(schedule_seed, "initial", name))
            if name == "murtys":
                out, score, length, per = O.assign_tasks_with_method(meth, matrix)
            else:
                out, score, length, per = V.assign_tasks_with_voting(robots, tasks, matrix, NC, meth)
            tup = benchmark_simulation(
                output=out, robots=copy.deepcopy(robots), tasks=copy.deepcopy(tasks),
                num_candidates=NC, voting_method=meth, grid=GRID, map_dict=MAP_DICT,
                suitability_method=sm, suitability_matrix=matrix,
                max_time_steps=MAX_STEPS,
                add_tasks=True, add_robots=True, remove_robots=True,
                event_schedule=event_schedule,
                mask_level=MASK_LEVEL, mask_policy=MASK_POLICY,
                schedule_seed=schedule_seed,
                run_seed=_derived_seed(schedule_seed, "run", name),
                planner="pp", scenario=SCENARIO,
            )
            w.writerow([name] + list(tup))


if __name__ == "__main__":
    main()
