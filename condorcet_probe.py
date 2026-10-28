"""Item 4 (2026-10-27): probe the 20 s Condorcet allocation call.

Re-runs condorcet on lak307d 100x100, rep 0, sigma_obs = 0 (the item-6
real-time config; deterministic, so the slow call reproduces) with wrappers
that record, per voting allocation call: the tick (count of completed
simulate_time_step calls), duration, whether the call was urgent (stall_k
is not None => the stall rule applied), the effective candidate pool size,
and the unassigned counts. Reports every call >= 5 s, with details for the
worst. Report only; no fix. MUST run idle (LaCAM calls + wall-clock
measurement).

Usage: python condorcet_probe.py
"""
import json
import os
import sys
import time

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

os.environ.setdefault("HVBTA_NOISE_SIGMAS", "0")

_TICK = [0]
_RECORDS = []
_CUR = ["", 0]


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
    from hvbta.partial_observability import (
        PartialObservabilityState, noisy_scorer)

    # --- wrappers -------------------------------------------------------
    orig_step = MS.simulate_time_step

    def wrapped_step(*a, **kw):
        result = orig_step(*a, **kw)
        _TICK[0] += 1
        return result

    MS.simulate_time_step = wrapped_step

    orig_reassign = V.reassign_robots_to_tasks

    def wrapped_reassign(robots, tasks, num_candidates, voting_method,
                         suitability_source, unassigned_robots,
                         unassigned_tasks, start_positions, goal_positions,
                         map_size, **kw):
        t0 = time.perf_counter_ns()
        try:
            return orig_reassign(robots, tasks, num_candidates, voting_method,
                                 suitability_source, unassigned_robots,
                                 unassigned_tasks, start_positions,
                                 goal_positions, map_size, **kw)
        finally:
            if getattr(voting_method, "__name__", "") == "rank_assignments_condorcet_method":
                dur_s = (time.perf_counter_ns() - t0) / 1e9
                _RECORDS.append({
                    "scenario": _CUR[0], "tick": _TICK[0],
                    "duration_s": dur_s,
                    "urgent_stall": kw.get("stall_k") is not None,
                    "pool": num_candidates,
                    "threshold": kw.get("score_threshold"),
                    "n_unassigned_robots": sum(1 for r in robots if not r.assigned),
                    "n_unassigned_tasks": sum(1 for t in tasks if not t.assigned),
                })

    V.reassign_robots_to_tasks = wrapped_reassign

    # --- run the item-6 config ------------------------------------------
    map_name, nr, nt, rep = "lak307d.map", 100, 100, 0
    grid = load_map(os.path.join("MAPF_benchmark_maps", map_name))
    HYP = (len(grid) ** 2 + len(grid[0]) ** 2) ** 0.5
    map_dict = {"dimension": (len(grid), len(grid[0])),
                "obstacles": create_obstacle_list(grid)}
    nc = max(nr, nt) * 6
    method = V.rank_assignments_condorcet_method

    for scenario in ["calm", "task_churn", "fleet_churn", "overload"]:
        import zlib
        seed_tuple = (map_name, nr, nt, nc, "evaluate_suitability_balanced",
                      rep, scenario)
        schedule_seed = zlib.adler32(repr(seed_tuple).encode()) & 0xFFFFFFFF
        _CUR[0] = scenario
        _TICK[0] = 0

        _seed_rngs(_derived_seed(schedule_seed, "entities"))
        robot_locs, robots = set(), []
        for i in range(nr):
            r = G.generate_random_robot_profile_strict(f"R{i+1}", grid, robot_locs)
            robots.append(r)
            robot_locs.add(r.location)
        task_locs, tasks = set(), []
        for i in range(nt):
            t = G.generate_random_task_description_strict(f"T{i+1}", grid, task_locs, [])
            tasks.append(t)
            task_locs.add(t.location)

        regime = REGIMES[scenario]
        provisional = max(200, int(HYP * 3))
        schedule = build_event_schedule(
            robots, tasks, grid, provisional,
            add_tasks=True, add_robots=True, remove_robots_flag=True,
            remove_tasks_flag=True,
            task_arrival_rate=regime["task_arrival_rate"],
            robot_arrival_rate=regime["robot_arrival_rate"],
            robot_departure_rate=regime["robot_departure_rate"],
            task_cancellation_rate=regime["task_cancellation_rate"],
            tasks_to_add=2, robots_to_add=1, robots_to_remove=1,
            tasks_to_remove=1, seed=schedule_seed)
        horizon = horizon_for_event_count(schedule, 125)
        max_steps = max(200, horizon, int(HYP * 1.5))
        schedule = schedule[:max_steps]

        sm = S.evaluate_suitability_balanced
        O.PERTURBED_JV_SIGMA = 0.05
        po_cond = PartialObservabilityState(schedule_seed, "none", "none")
        matrix = S.calculate_suitability_matrix(
            robots, tasks, noisy_scorer(po_cond.masked_scorer(sm), 0.0,
                                       schedule_seed, [0]), HYP)

        _seed_rngs(_derived_seed(schedule_seed, "initial",
                                 "rank_assignments_condorcet_method"))
        out, score, length, per = V.assign_tasks_with_voting(
            robots, tasks, matrix, nc, method,
            tie_seed=_derived_seed(schedule_seed, "ties", "initial",
                                   "rank_assignments_condorcet_method"))
        benchmark_simulation(
            output=out, robots=copy.deepcopy(robots),
            tasks=copy.deepcopy(tasks),
            num_candidates=nc, voting_method=method, grid=grid,
            map_dict=map_dict, suitability_method=sm,
            suitability_matrix=matrix,
            max_time_steps=max_steps, add_tasks=True, add_robots=True,
            remove_robots=True, event_schedule=schedule,
            mask_level="none", mask_policy="none", noise_sigma=0.0,
            schedule_seed=schedule_seed,
            run_seed=_derived_seed(schedule_seed, "run",
                                   "rank_assignments_condorcet_method"),
            planner="lacam", scenario=scenario,
            scheduled_event_count=0, horizon_binding_term="")
        print(f"  {scenario}: {len(_RECORDS)} condorcet calls so far", flush=True)

    # --- report ----------------------------------------------------------
    durs = np.asarray([r["duration_s"] for r in _RECORDS])
    print(f"\n=== condorcet probe: {len(_RECORDS)} calls, max {durs.max():.1f} s ===")
    print("calls >= 5 s:")
    for r in _RECORDS:
        if r["duration_s"] >= 5.0:
            print(f"  scenario={r['scenario']:12s} tick={r['tick']:4d} "
                  f"dur={r['duration_s']:7.1f}s urgent={r['urgent_stall']} "
                  f"pool={r['pool']} unassigned={r['n_unassigned_robots']}r/"
                  f"{r['n_unassigned_tasks']}t")
    worst = max(_RECORDS, key=lambda r: r["duration_s"])
    print("worst call:", json.dumps(worst))
    out_dir = os.path.join("hvbta", "io", "results", "condorcet_probe")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "condorcet_probe.json"), "w",
              encoding="utf-8") as f:
        json.dump(_RECORDS, f, indent=2)
    print(f"written: {out_dir}/condorcet_probe.json")


if __name__ == "__main__":
    main()