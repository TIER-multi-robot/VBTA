"""Item 6 (final scope): real-time check - allocation and planner time per
call on the largest config, versus a 1-second tick.

ONE rep of lak307d 100x100, all 4 scenarios, all 8 methods, sigma_obs = 0
(masks at the none control), LaCAM planner. Runs SERIALLY in this one
process so the timings are not distorted by load - nothing else should be
running on the machine.

Instrumentation is wrapper-based (no core-code changes): the driver
monkeypatches the module attributes main_simulation looks up at call time
(MS.run_planner, V.reassign_robots_to_tasks,
O.reassign_robots_to_tasks_with_method) plus the initial-assignment
entry points (V.assign_tasks_with_voting, O.assign_tasks_with_method),
timing every call. The run itself is bit-identical to the sweep's
(wrappers only add timing around the original functions).

Group/seed/condition construction replicates phase4_sweep.run_group
exactly (verified pool == serial), so the calls measured are the sweep's
own calls for this config.

Output: hvbta/io/results/realtime_check_lak307d/realtime_report.json
plus a printed table.

Usage: python realtime_check.py
"""
import csv
import json
import os
import sys
import time
import zlib

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

# sigma_obs = 0 only; must be set before phase4-level condition lists are
# used (this driver builds its own, but keeps the env for parity).
os.environ.setdefault("HVBTA_NOISE_SIGMAS", "0")

N_CALLS = {}

# ---------------------------------------------------------------------------
# Instrumented wrappers (installed after imports; see main())
# ---------------------------------------------------------------------------
_ORIG = {}


def _time_calls(mod, name):
    fn = getattr(mod, name)

    def wrapped(*args, **kw):
        t0 = time.perf_counter_ns()
        try:
            return fn(*args, **kw)
        finally:
            N_CALLS.setdefault(_CURRENT[0], {}).setdefault(name, []).append(
                (time.perf_counter_ns() - t0) / 1000.0)

    _ORIG[(mod, name)] = fn
    setattr(mod, name, wrapped)


_CURRENT = [None]   # ("scenario|method",) tag set around each run


def main():
    import copy
    import numpy as np
    import hvbta.generation as G
    import hvbta.suitability as S
    import hvbta.allocators.voting as V
    import hvbta.allocators.optimizers as O
    import hvbta.simulation.main_simulation as MS
    from hvbta.simulation.main_simulation import (
        benchmark_simulation, build_event_schedule, horizon_for_event_count,
        _derived_seed, _seed_rngs, REGIMES,
    )
    from hvbta.pathfinding.CBS_helper import load_map, create_obstacle_list
    from hvbta.partial_observability import PartialObservabilityState

    # Install wrappers on the attributes main_simulation resolves at call
    # time, plus the initial-assignment entry points (this driver performs
    # the initial assignment itself, mirroring run_group). The replan
    # entry points carry most allocation calls.
    _time_calls(MS, "run_planner")
    _time_calls(V, "assign_tasks_with_voting")
    _time_calls(O, "assign_tasks_with_method")
    _time_calls(V, "reassign_robots_to_tasks")
    _time_calls(O, "reassign_robots_to_tasks_with_method")
    _time_calls(O, "reassign_robots_to_tasks_with_murtys")

    map_name, nr, nt = "lak307d.map", 100, 100
    grid = load_map(os.path.join("MAPF_benchmark_maps", map_name))
    HYP = (len(grid) ** 2 + len(grid[0]) ** 2) ** 0.5
    map_dict = {"dimension": (len(grid), len(grid[0])),
                "obstacles": create_obstacle_list(grid)}
    nc = max(nr, nt) * 6
    O.PERTURBED_JV_SIGMA = 0.05

    methods = [
        ("rank_assignments_borda", V.rank_assignments_borda),
        ("rank_assignments_approval", V.rank_assignments_approval),
        ("rank_assignments_majority_judgment", V.rank_assignments_majority_judgment),
        ("rank_assignments_cumulative_voting", V.rank_assignments_cumulative_voting),
        ("rank_assignments_condorcet_method", V.rank_assignments_condorcet_method),
        ("rank_assignments_range", V.rank_assignments_range),
        ("murtys_task_allocation", O.murtys_task_allocation),
        ("perturbed_jv_task_allocation", O.perturbed_jv_task_allocation),
    ]

    out_dir = os.path.join(
        "hvbta", "io", "results",
        os.environ.get("RT_OUTDIR", "realtime_check_lak307d"))
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "realtime_report.json")

    records = []
    t_all = time.perf_counter()
    for scenario in ["calm", "task_churn", "fleet_churn", "overload"]:
        rep = 0
        seed_tuple = (map_name, nr, nt, nc, "evaluate_suitability_balanced",
                      rep, scenario)
        schedule_seed = zlib.adler32(repr(seed_tuple).encode()) & 0xFFFFFFFF

        _seed_rngs(_derived_seed(schedule_seed, "entities"))
        robot_locs, robots = set(), []
        for idx in range(nr):
            r = G.generate_random_robot_profile_strict(f"R{idx+1}", grid, robot_locs)
            robots.append(r)
            robot_locs.add(r.location)
        task_locs, tasks = set(), []
        for idx in range(nt):
            t = G.generate_random_task_description_strict(
                f"T{idx+1}", grid, task_locs, [])
            tasks.append(t)
            task_locs.add(t.location)

        provisional = max(200, int(HYP * 3))
        regime = REGIMES[scenario]
        schedule = build_event_schedule(
            robots, tasks, grid, provisional,
            add_tasks=True, add_robots=True, remove_robots_flag=True,
            remove_tasks_flag=True,
            task_arrival_rate=regime["task_arrival_rate"],
            robot_arrival_rate=regime["robot_arrival_rate"],
            robot_departure_rate=regime["robot_departure_rate"],
            task_cancellation_rate=regime["task_cancellation_rate"],
            tasks_to_add=2, robots_to_add=1, robots_to_remove=1,
            tasks_to_remove=1, seed=schedule_seed,
        )
        horizon = horizon_for_event_count(schedule, 125)
        extender = 1.5 if (nt / nr) > 10 else 1.2 if (nt / nr) > 5 else 1.0
        travel_floor = int(HYP * 1.5 * extender)
        max_steps = max(200, horizon, travel_floor)
        schedule = schedule[:max_steps]

        sm = S.evaluate_suitability_balanced
        # sigma_obs = 0: noise machinery ON at sigma 0.0, masks none.
        from hvbta.partial_observability import noisy_scorer
        po_cond = PartialObservabilityState(schedule_seed, "none", "none")
        sm_for_matrix = noisy_scorer(po_cond.masked_scorer(sm), 0.0,
                                     schedule_seed, [0])
        matrix = S.calculate_suitability_matrix(robots, tasks, sm_for_matrix, HYP)

        for pretty, method in methods:
            _CURRENT[0] = f"{scenario}|{pretty}"
            N_CALLS[_CURRENT[0]] = {}
            _seed_rngs(_derived_seed(schedule_seed, "initial", pretty))
            if pretty == "murtys_task_allocation":
                out, score, length, per = O.assign_tasks_with_method(method, matrix)
            elif pretty == "perturbed_jv_task_allocation":
                init_fn = O.perturbed_jv_factory(
                    _derived_seed(schedule_seed, "run", pretty, "pjv", "initial"))
                out, score, length, per = O.assign_tasks_with_method(init_fn, matrix)
            else:
                out, score, length, per = V.assign_tasks_with_voting(
                    robots, tasks, matrix, nc, method,
                    tie_seed=_derived_seed(schedule_seed, "ties", "initial", pretty))
            t0 = time.perf_counter()
            tup = benchmark_simulation(
                output=out, robots=copy.deepcopy(robots),
                tasks=copy.deepcopy(tasks),
                num_candidates=nc, voting_method=method, grid=grid,
                map_dict=map_dict, suitability_method=sm,
                suitability_matrix=matrix,
                max_time_steps=max_steps, add_tasks=True, add_robots=True,
                remove_robots=True, event_schedule=schedule,
                mask_level="none", mask_policy="none", noise_sigma=0.0,
                schedule_seed=schedule_seed,
                run_seed=_derived_seed(schedule_seed, "run", pretty),
                planner="lacam", scenario=scenario,
                scheduled_event_count=0, horizon_binding_term="",
            )
            wall = time.perf_counter() - t0
            calls = N_CALLS[_CURRENT[0]]
            rec = {
                "scenario": scenario, "method": pretty,
                "ticks": int(tup[4]),
                "alloc_us": (calls.get("assign_tasks_with_voting", [])
                             + calls.get("assign_tasks_with_method", [])),
                # Reassignment entry points are timed inside benchmark via
                # the same N_CALLS buckets:
                "reassign_us": (calls.get("reassign_robots_to_tasks", [])
                                + calls.get("reassign_robots_to_tasks_with_method", [])
                                + calls.get("reassign_robots_to_tasks_with_murtys", [])),
                "planner_ms": [us / 1000.0 for us in calls.get("run_planner", [])],
                "run_wall_s": wall,
            }
            records.append(rec)
            print(f"  {scenario:14s} {pretty:38s} ticks={rec['ticks']:4d} "
                  f"alloc_calls={len(rec['alloc_us']) + len(rec['reassign_us'])} "
                  f"planner_calls={len(rec['planner_ms'])} "
                  f"({rec['run_wall_s']:.0f}s)", flush=True)
            _CURRENT[0] = None

    # ---- report ----
    print(f"\n=== realtime check: lak307d 100x100, rep 0, sigma_obs = 0 "
          f"(serial, idle machine; total {time.perf_counter()-t_all:.0f}s) ===")
    print(f"  {'method':38s} {'alloc p50':>10s} {'alloc p99':>10s} "
          f"{'alloc max':>10s} {'plan p50':>9s} {'plan p99':>9s} "
          f"{'plan max':>9s} {'calls':>12s}")
    summary = {}
    for pretty, _ in methods:
        alloc = []
        plan = []
        n_alloc = n_plan = 0
        for rec in records:
            if rec["method"] != pretty:
                continue
            alloc.extend(rec["alloc_us"])
            alloc.extend(rec["reassign_us"])
            plan.extend(rec["planner_ms"])
            n_alloc += len(rec["alloc_us"]) + len(rec["reassign_us"])
            n_plan += len(rec["planner_ms"])
        a = np.asarray(alloc) if alloc else np.asarray([float("nan")])
        p = np.asarray(plan) if plan else np.asarray([float("nan")])
        summary[pretty] = {
            "alloc_us_n": int(n_alloc),
            "alloc_us_p50": float(np.percentile(a, 50)),
            "alloc_us_p99": float(np.percentile(a, 99)),
            "alloc_us_max": float(a.max()),
            "alloc_us_gt_1s": int((a > 1_000_000).sum()),
            "planner_ms_n": int(n_plan),
            "planner_ms_p50": float(np.percentile(p, 50)),
            "planner_ms_p99": float(np.percentile(p, 99)),
            "planner_ms_max": float(p.max()),
        }
        s = summary[pretty]
        print(f"  {pretty:38s} {s['alloc_us_p50']:10.0f} "
              f"{s['alloc_us_p99']:10.0f} {s['alloc_us_max']:10.0f} "
              f"{s['planner_ms_p50']:9.1f} {s['planner_ms_p99']:9.1f} "
              f"{s['planner_ms_max']:9.1f} "
              f"{n_alloc:6d}/{n_plan:<5d}")
    print("\nvs a 1-second tick: 'alloc p99/max' over 1e6 us means a single")
    print("allocation call exceeds the tick budget; planner max likewise.")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"records": records, "summary": summary}, f, indent=2)
    print(f"written: {json_path}")


if __name__ == "__main__":
    main()