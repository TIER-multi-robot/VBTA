"""Phase 4 pool sweep driver.

One shared process pool across all maps and configs (not one process per
map). Work unit = one (map, robots, tasks, scenario, rep) group; the
worker runs all 8 methods for the group and returns the rows to the
parent, which alone writes the CSV (no concurrent appends). Resume-skip
is keyed by run config within the output directory.

Per-sweep overrides: "hyper" runs the PP planner with the PO machinery
off and 5 reps (PHASE4_REPS=<n> overrides the rep count globally - used
for the 2-rep hyper-PP pilot re-run).

Usage:
  python phase4_sweep.py plan              # queue + projections, no run
  python phase4_sweep.py run <outdir> [sweeps...]   # run the queue
  python phase4_sweep.py verify            # parallel-vs-serial check
  python phase4_sweep.py memory            # peak worker RSS on largest map

Workers = logical cores - 2. Queue ordered longest-expected-first using
the 2026-10-25 per-map timing measurements on the v9 code. Determinism:
every run derives its seeds from the (group, method) keys exactly as the
serial sweep does; RNG isolation is unaffected by pool scheduling.
"""
import copy
import csv
import os
import sys
import time
import zlib
import multiprocessing as mp

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

N_WORKERS = max(1, (os.cpu_count() or 4) - 2)

SWEEP_CONFIGS = {
    "full": [(r"combat.map", [10, 20, 30, 40, 50, 100], [10, 20, 30, 40, 50, 100]),
             (r"combat2.map", [10, 20, 30, 40, 50, 100], [10, 20, 30, 40, 50, 100])],
    "balanced": [
        (r"lak105d.map", [10, 20], [10, 20]),
        (r"lak102d.map", [10, 20], [10, 20]),
        (r"lak107d.map", [10, 20], [10, 20]),
        (r"arena.map", [20, 30, 50], [20, 30, 50]),
        (r"isound1.map", [20, 30, 50], [20, 30, 50]),
        (r"den204d.map", [20, 30, 50], [20, 30, 50]),
        (r"lak307d.map", [50, 75, 100], [50, 75, 100]),
        (r"den001d.map", [50, 75, 100], [50, 75, 100]),
        (r"den901d.map", [50, 75, 100], [50, 75, 100]),
    ],
    "constrained": [
        (r"lak101d.map", [10, 20], [10, 20]),
        (r"den404d.map", [10, 20], [10, 20]),
        (r"hrt002d.map", [10, 20, 30], [10, 20, 30]),
        (r"den009d.map", [10, 20, 30], [10, 20, 30]),
        (r"den101d.map", [10, 20, 30], [10, 20, 30]),
    ],
    "hyper": [
        (r"hrt003d.map", [10, 15], [10, 15]),
        (r"lak103c.map", [10, 15], [10, 15]),
        (r"lak108c.map", [10, 15], [10, 15]),
    ],
    # 2026-10-26 expansion (user-approved): the PO and noise experiments
    # move to the pool driver (their map sets grew beyond serial
    # feasibility). Map lists match main_simulation.__main__'s
    # PO_SWEEP_CONFIGS / NOISE_SWEEP_CONFIGS exactly; seeding and the
    # condition axes replicate the serial __main__ flow, so rows are
    # bit-identical to it (RNG isolation is unaffected by pool
    # scheduling; mask/sigma conditions are not in any seed).
    "po": [
        (r"lak102d.map", [10, 20], [10, 20]),
        (r"arena.map",   [20, 30, 50], [20, 30, 50]),
        (r"den204d.map", [20, 30, 50], [20, 30, 50]),
        (r"lak307d.map", [50, 75, 100], [50, 75, 100]),
    ],
    "noise": [
        (r"lak102d.map", [10, 20], [10, 20]),
        (r"den204d.map", [20, 30, 50], [20, 30, 50]),
        (r"lak307d.map", [50, 75, 100], [50, 75, 100]),
    ],
}

# Condition axis per sweep: each condition is (mask_level, mask_policy,
# noise_sigma). "po" -> the 5 mask conditions (sigma None); "noise" -> the
# 5 sigma_obs levels with masks at the none control condition; everything
# else -> the single none/none control with no noise. Hyper (PP) keeps the
# PO machinery fully OFF (mask_level None), matching the pilot config.
NOISE_SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.3]


def sweep_conditions(sweep):
    if sweep == "po":
        return [("none", "none", None),
                ("low", "optimistic", None), ("low", "pessimistic", None),
                ("high", "optimistic", None), ("high", "pessimistic", None)]
    if sweep == "noise":
        return [("none", "none", s) for s in NOISE_SIGMAS]
    if sweep == "hyper":
        return [(None, None, None)]
    return [("none", "none", None)]

SCENARIOS = ["calm", "task_churn", "fleet_churn", "overload"]
REPS = 3
# Per-sweep overrides (2026-10-26): hyper runs PP (planner-failure stress)
# with the PO machinery fully off (mask_level=None, matching the
# pilot_hyper_pp_v9 configuration) and 5 reps (analysis-plan amendment).
SWEEP_PLANNER = {"hyper": "pp"}
SWEEP_REPS = {"hyper": 5}
# PHASE4_REPS overrides the rep count for every sweep (used by the 2-rep
# hyper-PP pilot re-run on the capped planner-work bounds).
if os.environ.get("PHASE4_REPS"):
    REPS = int(os.environ["PHASE4_REPS"])
    SWEEP_REPS = {}
N_METHODS = 8

# Largest-config per-run mean seconds, from the 2026-10-25 timing run
# (one 7-method rep of each map's largest config, v9 code).
MAP_TIMING = {
    "lak105d.map": 2.0, "lak102d.map": 2.2, "lak107d.map": 3.4,
    "arena.map": 4.0, "isound1.map": 11.5, "den204d.map": 15.2,
    "lak307d.map": 32.5, "den001d.map": 63.9, "den901d.map": 96.2,
    "combat.map": 114.9, "combat2.map": 112.3,
    "lak101d.map": 4.9, "den404d.map": 3.6, "hrt002d.map": 5.8,
    "den009d.map": 6.5, "den101d.map": 12.1,
    # Hyper maps: v11 pilot re-run (PP with deterministic planner-work
    # bounds, 2026-10-26) largest-config means, measured under 30-worker
    # load (replaces the stale LaCAM-pace estimates).
    "hrt003d.map": 28.5, "lak103c.map": 46.6, "lak108c.map": 67.0,
}
MAP_LARGEST_FLEET = {
    "combat.map": 100, "combat2.map": 100, "lak307d.map": 100,
    "den001d.map": 100, "den901d.map": 100, "den101d.map": 30,
    "hrt002d.map": 30, "den009d.map": 30, "den404d.map": 30,
    "arena.map": 50, "isound1.map": 50, "den204d.map": 50,
    "lak101d.map": 30, "hrt003d.map": 15, "lak103c.map": 15, "lak108c.map": 15,
    "lak105d.map": 20, "lak102d.map": 20, "lak107d.map": 20,
}

# Row layout: identity prefix + output tuple; timing columns excluded from
# equality checks: total_reassignment_time (output col 1) and Execution
# Time (appended by benchmark_simulation, last output col).
TIMING_OUTPUT_COLS = {1}          # 0-based within output tuple
ROW_HEADER = ["Map", "Num Robots", "Num Tasks", "Scenario", "Rep",
              "Method", "Output", "Map Size", "Init_Alloc_us",
              "Mask_Level", "Mask_Policy", "Obs_Noise_Sigma"]


def _cost(map_name, nr):
    base = MAP_TIMING.get(map_name, 5.0)
    big = MAP_LARGEST_FLEET.get(map_name, 100)
    return N_METHODS * base * max(0.25, nr / big)


def build_queue(sweeps):
    items = []
    for sweep in sweeps:
        n_cond = len(sweep_conditions(sweep))
        for map_name, r_sizes, t_sizes in SWEEP_CONFIGS[sweep]:
            for nr in r_sizes:
                for nt in t_sizes:
                    cost = _cost(map_name, nr) * n_cond
                    for scenario in SCENARIOS:
                        for rep in range(SWEEP_REPS.get(sweep, REPS)):
                            items.append({
                                "sweep": sweep, "map": map_name, "nr": nr,
                                "nt": nt, "scenario": scenario, "rep": rep,
                                "cost": cost,
                                "planner": SWEEP_PLANNER.get(sweep, "lacam"),
                            })
    items.sort(key=lambda it: -it["cost"])
    return items


def run_group(job):
    """Worker: all 7 methods for one group -> list of rows."""
    import hvbta.generation as G
    import hvbta.suitability as S
    import hvbta.allocators.voting as V
    import hvbta.allocators.optimizers as O
    from hvbta.simulation.main_simulation import (
        benchmark_simulation, build_event_schedule, horizon_for_event_count,
        _derived_seed, _seed_rngs, REGIMES,
    )
    from hvbta.pathfinding.CBS_helper import load_map, create_obstacle_list
    from hvbta.partial_observability import PartialObservabilityState

    TRANSIENT_FLOOR, TARGET_EVENT_COUNT_K = 200, 125
    map_name, nr, nt, scenario, rep = (job["map"], job["nr"], job["nt"],
                                       job["scenario"], job["rep"])
    planner = job.get("planner", "lacam")
    grid = load_map(os.path.join("MAPF_benchmark_maps", map_name))
    map_size = len(grid) * len(grid[0])
    HYP = (len(grid) ** 2 + len(grid[0]) ** 2) ** 0.5
    map_dict = {"dimension": (len(grid), len(grid[0])),
                "obstacles": create_obstacle_list(grid)}
    nc = max(nr, nt) * 6
    seed_tuple = (map_name, nr, nt, nc, "evaluate_suitability_balanced", rep, scenario)
    schedule_seed = zlib.adler32(repr(seed_tuple).encode()) & 0xFFFFFFFF

    _seed_rngs(_derived_seed(schedule_seed, "entities"))
    robot_locs, robots = set(), []
    for idx in range(nr):
        r = G.generate_random_robot_profile_strict(f"R{idx+1}", grid, robot_locs)
        robots.append(r); robot_locs.add(r.location)
    task_locs, tasks = set(), []
    for idx in range(nt):
        t = G.generate_random_task_description_strict(f"T{idx+1}", grid, task_locs, [])
        tasks.append(t); task_locs.add(t.location)

    provisional = max(TRANSIENT_FLOOR, int(HYP * 3))
    regime = REGIMES[scenario]
    schedule = build_event_schedule(
        robots, tasks, grid, provisional,
        add_tasks=True, add_robots=True, remove_robots_flag=True, remove_tasks_flag=True,
        task_arrival_rate=regime["task_arrival_rate"],
        robot_arrival_rate=regime["robot_arrival_rate"],
        robot_departure_rate=regime["robot_departure_rate"],
        task_cancellation_rate=regime["task_cancellation_rate"],
        tasks_to_add=2, robots_to_add=1, robots_to_remove=1, tasks_to_remove=1,
        seed=schedule_seed,
    )
    horizon = horizon_for_event_count(schedule, TARGET_EVENT_COUNT_K)
    workload = max(1.0, nt / nr)
    extender = 1.5 if workload > 10 else 1.2 if workload > 5 else 1.0
    travel_floor = int(HYP * 1.5 * extender)
    max_steps = max(TRANSIENT_FLOOR, horizon, travel_floor)
    schedule = schedule[:max_steps]
    # Provenance fields, identical to the serial sweep's __main__.
    scheduled_event_count = sum(
        len(e["add_tasks"]) + len(e["add_robots"])
        + len(e["remove_robot_ids"]) + len(e["remove_task_ids"])
        for e in schedule)
    horizon_binding_term = (
        "transient" if TRANSIENT_FLOOR >= horizon
        and TRANSIENT_FLOOR >= travel_floor
        else "events" if horizon >= travel_floor
        else "travel"
    )

    sm = S.evaluate_suitability_balanced
    O.PERTURBED_JV_SIGMA = 0.05

    methods = [
        ("rank_assignments_borda", V.rank_assignments_borda),
        ("rank_assignments_approval", V.rank_assignments_approval),
        ("rank_assignments_majority_judgment", V.rank_assignments_majority_judgment),
        ("rank_assignments_cumulative_voting", V.rank_assignments_cumulative_voting),
        ("rank_assignments_condorcet_method", V.rank_assignments_condorcet_method),
        ("rank_assignments_range", V.rank_assignments_range),
        ("murtys_task_allocation", O.murtys_task_allocation),
        # Pretty name matches the serial sweep's func_name so the
        # (group, method) seed keys are identical across drivers.
        ("perturbed_jv_task_allocation", O.perturbed_jv_task_allocation),
    ]

    # Condition axis (2026-10-26): PO sweeps iterate the 5 mask
    # conditions, noise sweeps the 5 sigma_obs levels (masks at the none
    # control), hyper keeps the PO machinery fully OFF (mask_level None,
    # matching pilot_hyper_pp_v9/v11), everything else runs the single
    # none/none control condition. Conditions are not in any seed, so
    # they replay identical schedules.
    sweep = job.get("sweep", "balanced")
    conditions = sweep_conditions(sweep)

    rows = []
    for mask_level, mask_policy, noise_sigma in conditions:
        po_cond = (PartialObservabilityState(schedule_seed, mask_level, mask_policy)
                   if mask_level is not None else None)
        sm_for_matrix = sm
        if po_cond is not None:
            sm_for_matrix = po_cond.masked_scorer(sm)
        if noise_sigma is not None:
            from hvbta.partial_observability import noisy_scorer as _noisy
            sm_for_matrix = _noisy(sm_for_matrix, noise_sigma, schedule_seed, [0])
        matrix = S.calculate_suitability_matrix(robots, tasks, sm_for_matrix, HYP)

        for pretty, method in methods:
            _seed_rngs(_derived_seed(schedule_seed, "initial", pretty))
            if pretty == "murtys_task_allocation":
                out, score, length, per = O.assign_tasks_with_method(method, matrix)
            elif pretty == "perturbed_jv_task_allocation":
                # The factory result is used ONLY for the initial solve;
                # benchmark_simulation must receive the plain function, or the
                # main loop's method dispatch (dict membership test) never
                # matches and the run never reallocates (bug found in the
                # pilot re-run: perturbed_jv completion 0.11 vs 0.72).
                init_fn = O.perturbed_jv_factory(
                    _derived_seed(schedule_seed, "run", pretty, "pjv", "initial"))
                out, score, length, per = O.assign_tasks_with_method(init_fn, matrix)
            else:
                out, score, length, per = V.assign_tasks_with_voting(
                    robots, tasks, matrix, nc, method,
                    tie_seed=_derived_seed(schedule_seed, "ties", "initial", pretty))
            tup = benchmark_simulation(
                output=out, robots=copy.deepcopy(robots), tasks=copy.deepcopy(tasks),
                num_candidates=nc, voting_method=method, grid=grid, map_dict=map_dict,
                suitability_method=sm, suitability_matrix=matrix,
                max_time_steps=max_steps, add_tasks=True, add_robots=True,
                remove_robots=True, event_schedule=schedule,
                mask_level=mask_level, mask_policy=mask_policy,
                noise_sigma=noise_sigma,
                schedule_seed=schedule_seed,
                run_seed=_derived_seed(schedule_seed, "run", pretty),
                planner=planner, scenario=scenario,
                scheduled_event_count=scheduled_event_count,
                horizon_binding_term=horizon_binding_term,
            )
            # Convert numpy scalars to plain floats so the CSV cells hold
            # clean numbers (a list's repr would otherwise embed
            # np.float64(...)). Init_Alloc_us = this method's initial-
            # assignment wall time, so the per-sweep report can compute
            # mean allocation time per call:
            # (Init_Alloc_us + total_reassignment_time)/(1 + reassignments).
            rows.append([map_name, nr, nt, scenario, rep, pretty,
                         [x.item() if hasattr(x, "item") else x for x in tup],
                         map_size, length,
                         mask_level or "", mask_policy or "",
                         (noise_sigma if noise_sigma is not None else "")])
    return rows


def resume_keys(csv_path):
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if row and row[0] == "Map":
                continue
            try:
                done.add((row[0], int(row[1]), int(row[2]), row[3], int(row[4])))
            except (ValueError, IndexError):
                continue
    return done


def rows_equal(r1, r2):
    """Full-row equality excluding timing columns of the output tuple."""
    # row: [map, nr, nt, scenario, rep, method, output(list), map_size]
    out1, out2 = r1[6], r2[6]
    if len(out1) != len(out2):
        return False
    for i, (a, b) in enumerate(zip(out1, out2)):
        if i in TIMING_OUTPUT_COLS or i == len(out1) - 1:  # timing + exec time
            continue
        if str(a) != str(b):
            return False
    return r1[:6] == r2[:6] and r1[7] == r2[7]


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "plan"
    sweeps = [a for a in sys.argv[2:] if a in SWEEP_CONFIGS]

    if cmd == "plan":
        sweeps = sweeps or list(SWEEP_CONFIGS)
        items = build_queue(sweeps)
        total = sum(it["cost"] for it in items)
        n_cond = {s: len(sweep_conditions(s)) for s in sweeps}
        print(f"workers: {N_WORKERS} (logical cores - 2)")
        print(f"queue: {len(items)} groups x {N_METHODS} methods"
              f" x (conditions: {n_cond}) = "
              f"{sum(len(sweep_conditions(it['sweep'])) for it in items) * N_METHODS} runs"
              f" over sweeps {sweeps}")
        print(f"projected serial wall time: {total/3600:.1f} h")
        print(f"projected wall time at {N_WORKERS} workers: "
              f"{total / N_WORKERS / 3600:.1f}-{total / N_WORKERS / 3600 * 1.4:.1f} h"
              f" (25-40% straggler overhead on the longest jobs)")
        for s in sweeps:
            s_items = [it for it in items if it["sweep"] == s]
            s_total = sum(it["cost"] for it in s_items)
            s_runs = sum(len(sweep_conditions(s)) for _ in s_items) * N_METHODS
            print(f"  {s:12s} {len(s_items):4d} groups {s_runs:6d} runs "
                  f"~{s_total/3600:7.1f} h serial "
                  f"(~{s_total/N_WORKERS/3600:5.1f}-{s_total/N_WORKERS/3600*1.4:5.1f} h @30)")
        print("first 10 queue entries (longest expected first):")
        for it in items[:10]:
            print("  %-10s %-12s %3dx%-3d %-12s rep%d ~%.0fs"
                  % (it["sweep"], it["map"], it["nr"], it["nt"], it["scenario"], it["rep"], it["cost"]))
        return

    if cmd == "verify":
        # Small config, parallel (pool) vs serial (in-process).
        jobs = [
            {"map": "lak102d.map", "nr": 10, "nt": 10, "scenario": s, "rep": 0, "cost": 0}
            for s in ["calm", "task_churn", "fleet_churn", "overload"]
        ]
        serial = {}
        for j in jobs:
            key = j["scenario"]
            t0 = time.perf_counter()
            serial[key] = run_group(j)
            print(f"  serial {key}: {time.perf_counter()-t0:.1f}s", flush=True)
        with mp.Pool(min(N_WORKERS, 4)) as pool:
            par = pool.map(run_group, jobs)
        murty_same, voting_diff = True, 0
        for j, rows in zip(jobs, par):
            key = j["scenario"]
            for r_par, r_ser in zip(rows, serial[key]):
                if r_par[5] == "murtys_task_allocation":
                    murty_same = murty_same and rows_equal(r_par, r_ser)
                elif not rows_equal(r_par, r_ser):
                    voting_diff += 1
        print(f"murty rows identical (excl. timing): {murty_same}")
        print(f"voting rows differing vs serial: {voting_diff}/24 "
              f"(anytime 20ms wall-clock budget noise)")

    if cmd == "memory":
        import psutil
        job = {"map": "den901d.map", "nr": 100, "nt": 100,
               "scenario": "calm", "rep": 0, "cost": 0}
        proc = psutil.Process()
        t = proc.memory_info().rss
        import threading
        peak = [t]

        def _watch():
            while not done[0]:
                peak[0] = max(peak[0], proc.memory_info().rss)
                time.sleep(0.05)
        done = [False]
        th = threading.Thread(target=_watch, daemon=True)
        th.start()
        rows = run_group(job)
        done[0] = True
        th.join(timeout=1)
        print(f"largest map (den901d 100x100) group: "
              f"peak worker RSS {peak[0]/1e6:.0f} MB")

    if cmd == "run":
        outdir = sys.argv[2]
        sweeps = sys.argv[3:] or list(SWEEP_CONFIGS)
        res_dir = os.path.join("hvbta", "io", "results", outdir)
        csv_path = os.path.join(res_dir, "phase4_simulation.csv")
        if os.path.exists(csv_path):
            raise SystemExit(f"{csv_path} already exists - use a fresh v2 directory")
        os.makedirs(res_dir, exist_ok=True)
        items = build_queue(sweeps)
        done = resume_keys(csv_path)
        todo = [it for it in items
                if (it["map"], it["nr"], it["nt"], it["scenario"], it["rep"]) not in done]
        print(f"{len(items)} groups total, {len(items)-len(todo)} done, "
              f"{len(todo)} to run; workers={N_WORKERS}")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(ROW_HEADER)
            t0 = time.perf_counter()
            with mp.Pool(N_WORKERS) as pool:
                for i, rows in enumerate(pool.imap_unordered(run_group, todo), 1):
                    for r in rows:
                        w.writerow(r)
                    f.flush()
                    if i % 10 == 0 or i == len(todo):
                        print(f"  {i}/{len(todo)} groups "
                              f"({(time.perf_counter()-t0)/60:.0f} min)", flush=True)
        print("DONE")


if __name__ == "__main__":
    main()
