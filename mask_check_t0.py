"""Mask check (2026-10-29, report-only): zero-entry fraction of the
allocator's view AT TICK 0, per PO mask condition, averaged over lak307d
groups - plus the accumulation signal (Masked_Starvation_Count /
Never_Attempted_Count from the merged PO sweep) to test whether the
late-run 99.5% sparsity comes from masked tasks accumulating unassigned
or is already present at tick 0.

For every lak307d group (9 size combos x 4 scenarios, rep 0 - the same
seeded entity generation run_group uses; conditions are not in any seed,
so all five conditions see identical entities) it builds the initial
suitability matrix the allocator sees at tick 0 under each mask
condition and reports:
  zero_frac      - fraction of matrix entries == 0
  pos_task_frac  - fraction of tasks with >= 1 positive entry (visible
                   to at least one robot)
  pos_robot_frac - fraction of robots with >= 1 positive entry

Usage: python mask_check_t0.py
"""

import os
import sys
import zlib

import numpy as np

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)


def main():
    import csv
    import collections
    import hvbta.generation as G
    import hvbta.suitability as S
    from hvbta.simulation.main_simulation import (
        build_event_schedule, _derived_seed, _seed_rngs, REGIMES,
    )
    from hvbta.pathfinding.CBS_helper import load_map, create_obstacle_list
    from hvbta.partial_observability import PartialObservabilityState

    map_name = "lak307d.map"
    grid = load_map(os.path.join("MAPF_benchmark_maps", map_name))
    HYP = (len(grid) ** 2 + len(grid[0]) ** 2) ** 0.5
    sm = S.evaluate_suitability_balanced
    CONDITIONS = (("none", "none"), ("low", "optimistic"),
                  ("low", "pessimistic"), ("high", "optimistic"),
                  ("high", "pessimistic"))
    rep = 0
    acc = collections.defaultdict(lambda: ([], [], []))
    detail = []
    for nr in (50, 75, 100):
        for nt in (50, 75, 100):
            for scenario in ("calm", "task_churn", "fleet_churn",
                             "overload"):
                nc = max(nr, nt) * 6
                seed_tuple = (map_name, nr, nt, nc,
                              "evaluate_suitability_balanced", rep, scenario)
                schedule_seed = zlib.adler32(repr(seed_tuple).encode()) \
                    & 0xFFFFFFFF
                _seed_rngs(_derived_seed(schedule_seed, "entities"))
                robot_locs, robots = set(), []
                for i in range(nr):
                    r = G.generate_random_robot_profile_strict(
                        f"R{i+1}", grid, robot_locs)
                    robots.append(r); robot_locs.add(r.location)
                task_locs, tasks = set(), []
                for i in range(nt):
                    t = G.generate_random_task_description_strict(
                        f"T{i+1}", grid, task_locs, [])
                    tasks.append(t); task_locs.add(t.location)
                # keep the generation RNG draw-parity identical to
                # run_group: build (and discard) the schedule too
                regime = REGIMES[scenario]
                schedule = build_event_schedule(
                    robots, tasks, grid, max(200, int(HYP * 3)),
                    add_tasks=True, add_robots=True, remove_robots_flag=True,
                    remove_tasks_flag=True,
                    task_arrival_rate=regime["task_arrival_rate"],
                    robot_arrival_rate=regime["robot_arrival_rate"],
                    robot_departure_rate=regime["robot_departure_rate"],
                    task_cancellation_rate=regime["task_cancellation_rate"],
                    tasks_to_add=2, robots_to_add=1, robots_to_remove=1,
                    tasks_to_remove=1, seed=schedule_seed)
                for level, policy in CONDITIONS:
                    po = PartialObservabilityState(schedule_seed, level,
                                                   policy)
                    m = np.asarray(S.calculate_suitability_matrix(
                        robots, tasks, po.masked_scorer(sm), HYP))
                    zf = float(np.mean(m == 0))
                    pt = float(np.mean(np.any(m > 0, axis=0)))
                    pr = float(np.mean(np.any(m > 0, axis=1)))
                    acc[(level, policy)][0].append(zf)
                    acc[(level, policy)][1].append(pt)
                    acc[(level, policy)][2].append(pr)
                    detail.append((nr, nt, scenario, level, policy, zf, pt))
            print(f"  {nr}x{nt} done", flush=True)

    print("\n=== tick-0 allocator view, lak307d, mean over 36 groups "
          "(9 sizes x 4 scenarios, rep 0) ===")
    print(f"{'condition':<22}{'zero_frac':>12}{'pos_task_frac':>15}"
          f"{'pos_robot_frac':>16}")
    for cond in CONDITIONS:
        zf, pt, pr = acc[cond]
        print(f"{cond[0] + '-' + cond[1]:<22}{np.mean(zf):>12.3f}"
              f"{np.mean(pt):>15.3f}{np.mean(pr):>16.3f}")

    # accumulation signal from the merged PO sweep (descriptive)
    rows = list(csv.DictReader(open(
        os.path.join("hvbta", "io", "results", "phase4_final_po_merged",
                     "phase4_simulation.csv"), newline="", encoding="utf-8")))
    lak = [r for r in rows if r["Map"] == "lak307d.map"]
    print("\n=== accumulation signal (phase4_final_po_merged, lak307d "
          "rows, mean per run) ===")
    print(f"{'condition':<22}{'Masked_Starv':>14}{'Never_Attempt':>15}"
          f"{'Tasks_Spawned':>15}{'PoolSize_mean':>15}")
    for cond in CONDITIONS:
        sel = [r for r in lak
               if r["Mask_Level"] == cond[0] and r["Mask_Policy"] == cond[1]]
        if not sel:
            continue
        print(f"{cond[0] + '-' + cond[1]:<22}"
              f"{np.mean([float(r['Masked_Starvation_Count']) for r in sel]):>14.1f}"
              f"{np.mean([float(r['Never_Attempted_Count']) for r in sel]):>15.1f}"
              f"{np.mean([float(r['Tasks_Spawned']) for r in sel]):>15.1f}"
              f"{np.mean([float(r['Mean_Task_Pool_Size']) for r in sel]):>15.1f}")

    out_dir = os.path.join("hvbta", "io", "results", "mask_check_t0")
    os.makedirs(out_dir, exist_ok=True)
    import json
    with open(os.path.join(out_dir, "mask_check_t0.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "per_condition_means": {
                f"{l}-{p}": {"zero_frac": float(np.mean(acc[(l, p)][0])),
                             "pos_task_frac": float(np.mean(acc[(l, p)][1])),
                             "pos_robot_frac": float(np.mean(acc[(l, p)][2]))}
                for l, p in CONDITIONS},
            "detail": detail}, f, indent=2)
    print(f"\nwritten: {out_dir}/mask_check_t0.json", flush=True)


if __name__ == "__main__":
    main()