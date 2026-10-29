"""Targeted re-run of individual invalid/truncated runs (2026-10-28).

Work unit = ONE run (map, nr, nt, scenario, rep, condition, method) — never a
whole group. Replicates phase4_sweep.run_group's setup code exactly (same
entities, schedule, seeds, matrix construction, per-method RNG isolation) and
executes only the requested condition + method, so the row is identical to
what run_group would have produced for that (condition, method) slot.

Jobs (all same code + seeds, HVBTA_RUN_WALL_LIMIT_S=12000, LaCAM 1 s limit
unchanged):
  - PO: the 24 phase4_final_po_uncapped replacement rows with
    Planner_Timeout_Count > 0 whose source row was truncated (invalid under
    the 2026-10-28 validity rule: a row is valid iff
    Planner_Timeout_Count = 0).
  - Balanced: the 38 phase4_final_balanced rows with Run_Timed_Out = 1.
  - 4 non-truncated balanced CONTROL rows, compared bit-for-bit against the
    originals to validate the driver's fidelity.

Output: hvbta/io/results/rerun_targeted/phase4_simulation.csv (standard
87-column pool schema, one row per ATTEMPT, appended in attempt order).
Any key whose latest attempt still has Planner_Timeout_Count > 0 is retried at
a lower worker count (8 -> 2 -> 1) until clean. Resume-safe: keys that
already have a Planner_Timeout_Count = 0 row in the output are skipped.

Usage:
  python rerun_targeted.py run
  python rerun_targeted.py status
"""

import csv
import os
import sys
import time
import multiprocessing as mp

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

TIMING_COLS = {"Planning_Time_us", "total_reassignment_time",
               "Planner_Total_ms", "Execution Time", "Planner_Timeout_Count"}
KEY_COLS = ("Map", "Num Robots", "Num Tasks", "Scenario", "Rep",
            "Mask_Level", "Mask_Policy", "Obs_Noise_Sigma", "Method")
OUT_DIR = os.path.join("hvbta", "io", "results", "rerun_targeted")
CSV_PATH = os.path.join(OUT_DIR, "phase4_simulation.csv")

CONTROLS = [
    ("den001d.map", 50, 100, "fleet_churn", 1, "none", "none", None,
     "rank_assignments_borda"),
    ("den901d.map", 50, 100, "calm", 1, "none", "none", None,
     "rank_assignments_condorcet_method"),
    ("den001d.map", 50, 75, "task_churn", 1, "none", "none", None,
     "perturbed_jv_task_allocation"),
    ("den001d.map", 50, 100, "calm", 2, "none", "none", None,
     "murtys_task_allocation"),
]


def results_csv(dir_name):
    return os.path.join("hvbta", "io", "results", dir_name,
                        "phase4_simulation.csv")


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def row_key(r):
    return tuple(r[c] for c in KEY_COLS)


def job_from_row(r, cost):
    sigma = r["Obs_Noise_Sigma"]
    return {"map": r["Map"], "nr": int(r["Num Robots"]),
            "nt": int(r["Num Tasks"]), "scenario": r["Scenario"],
            "rep": int(r["Rep"]), "mask_level": r["Mask_Level"],
            "mask_policy": r["Mask_Policy"],
            "noise_sigma": float(sigma) if sigma not in ("", None) else None,
            "method": r["Method"], "cost": cost}


def run_single(job):
    """Exactly phase4_sweep.run_group's code path, filtered to ONE condition
    and ONE method. Returns the 87-column row."""
    import copy
    import zlib
    import hvbta.generation as G
    import hvbta.suitability as S
    import hvbta.allocators.voting as V
    import hvbta.allocators.optimizers as O
    from hvbta.simulation.main_simulation import (
        benchmark_simulation, build_event_schedule, horizon_for_event_count,
        _derived_seed, _seed_rngs, REGIMES,
        build_initial_prefix, _initial_quality_fields, build_result_row,
    )
    from hvbta.pathfinding.CBS_helper import load_map, create_obstacle_list
    from hvbta.partial_observability import PartialObservabilityState, MASK_COUNTS

    TRANSIENT_FLOOR, TARGET_EVENT_COUNT_K = 200, 125
    map_name, nr, nt = job["map"], job["nr"], job["nt"]
    scenario, rep = job["scenario"], job["rep"]
    mask_level, mask_policy = job["mask_level"], job["mask_policy"]
    noise_sigma = job["noise_sigma"]
    planner = "lacam"
    grid = load_map(os.path.join("MAPF_benchmark_maps", map_name))
    HYP = (len(grid) ** 2 + len(grid[0]) ** 2) ** 0.5
    map_dict = {"dimension": (len(grid), len(grid[0])),
                "obstacles": create_obstacle_list(grid)}
    nc = max(nr, nt) * 6
    seed_tuple = (map_name, nr, nt, nc, "evaluate_suitability_balanced", rep, scenario)
    schedule_seed = zlib.adler32(repr(seed_tuple).encode()) & 0xFFFFFFFF
    HYP2 = HYP
    map_class = "Small" if HYP2 < 55 else "Medium" if HYP2 < 105 else "Large"

    _seed_rngs(_derived_seed(schedule_seed, "entities"))
    robot_locs, robots = set(), []
    for idx in range(nr):
        rob = G.generate_random_robot_profile_strict(f"R{idx+1}", grid, robot_locs)
        robots.append(rob); robot_locs.add(rob.location)
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

    methods = {
        "rank_assignments_borda": V.rank_assignments_borda,
        "rank_assignments_approval": V.rank_assignments_approval,
        "rank_assignments_majority_judgment": V.rank_assignments_majority_judgment,
        "rank_assignments_cumulative_voting": V.rank_assignments_cumulative_voting,
        "rank_assignments_condorcet_method": V.rank_assignments_condorcet_method,
        "rank_assignments_range": V.rank_assignments_range,
        "murtys_task_allocation": O.murtys_task_allocation,
        "perturbed_jv_task_allocation": O.perturbed_jv_task_allocation,
    }
    pretty = job["method"]
    method = methods[pretty]

    po_cond = (PartialObservabilityState(schedule_seed, mask_level, mask_policy)
               if mask_level is not None else None)
    sm_for_matrix = sm
    if po_cond is not None:
        sm_for_matrix = po_cond.masked_scorer(sm)
    if noise_sigma is not None:
        from hvbta.partial_observability import noisy_scorer as _noisy
        sm_for_matrix = _noisy(sm_for_matrix, noise_sigma, schedule_seed, [0])
    matrix = S.calculate_suitability_matrix(robots, tasks, sm_for_matrix, HYP)
    mask_count = (MASK_COUNTS.get(mask_level, -1) if mask_level else -1)

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
    total, per_reported = _initial_quality_fields(
        po_cond, out, score, per, robots, tasks, HYP)
    prefix13 = build_initial_prefix(
        f"{map_name}_{nr}x{nt}_{scenario}_rep{rep}", pretty,
        "evaluate_suitability_balanced", nr, nt, nc, total, length, per_reported)
    row_prefix = [prefix13[0], prefix13[1], prefix13[2], planner] + prefix13[3:]
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
    tup_clean = [x.item() if hasattr(x, "item") else x for x in tup]
    return [map_name, rep] + build_result_row(
        row_prefix, tup_clean, scenario, regime,
        mask_level or "", mask_count, mask_policy or "",
        noise_sigma, map_class)


def key_of_job(job):
    sigma = "" if job["noise_sigma"] is None else job["noise_sigma"]
    return (job["map"], str(job["nr"]), str(job["nt"]), job["scenario"],
            str(job["rep"]), job["mask_level"], job["mask_policy"],
            sigma, job["method"])


def build_jobs():
    v1 = load_rows(results_csv("phase4_final_po_uncapped"))
    po_src = load_rows(results_csv("phase4_final_po"))
    S_po = {row_key(r): r for r in po_src}
    po_jobs, po_keys = [], set()
    for r in v1:
        k = row_key(r)
        if int(r["Planner_Timeout_Count"]) > 0 and S_po[k]["Run_Timed_Out"] == "1":
            j = job_from_row(r, float(r["Execution Time"]) / 1e9)
            po_jobs.append(j)
            po_keys.add(key_of_job(j))
    bal_src = load_rows(results_csv("phase4_final_balanced"))
    by_g = {}
    for r in bal_src:
        by_g.setdefault(row_key(r)[:5], []).append(r)
    bal_jobs, bal_keys = [], set()
    for r in bal_src:
        if r["Run_Timed_Out"] == "1":
            g = row_key(r)[:5]
            t_ref = max(int(x["Total Time Steps"]) for x in by_g[g]
                        if x["Run_Timed_Out"] == "0")
            cost = 600.0 * t_ref / int(r["Total Time Steps"])
            j = job_from_row(r, cost)
            bal_jobs.append(j)
            bal_keys.add(key_of_job(j))
    ctrl_jobs, ctrl_keys = [], set()
    BS = {row_key(r): r for r in bal_src}
    for (m, nr, nt, sc, rep, ml, mp_, sg, meth) in CONTROLS:
        k = (m, str(nr), str(nt), sc, str(rep), ml, mp_,
             "" if sg is None else sg, meth)
        s = BS[k]
        assert s["Run_Timed_Out"] == "0", f"control is truncated: {k}"
        j = job_from_row(s, float(s["Execution Time"]) / 1e9)
        ctrl_jobs.append(j)
        ctrl_keys.add(key_of_job(j))
    print(f"jobs: PO {len(po_jobs)}, BAL {len(bal_jobs)}, "
          f"controls {len(ctrl_jobs)}", flush=True)
    return po_jobs + bal_jobs + ctrl_jobs, ctrl_keys


def valid_keys():
    if not os.path.exists(CSV_PATH):
        return set()
    valid, attempted = set(), set()
    for r in load_rows(CSV_PATH):
        attempted.add(row_key(r))
        if r["Planner_Timeout_Count"] == "0":
            valid.add(row_key(r))
    return valid, attempted


def verify_controls(ctrl_keys):
    ctrl_src = load_rows(results_csv("phase4_final_balanced"))
    BS = {row_key(r): r for r in ctrl_src}
    new_rows = {row_key(r): r for r in load_rows(CSV_PATH)
                if r["Planner_Timeout_Count"] == "0"}
    shared = None
    n_mis = 0
    for k in sorted(ctrl_keys):
        s = BS[k]
        n = new_rows.get(k)
        if n is None:
            print(f"control MISSING: {k}", flush=True)
            n_mis += 1
            continue
        if shared is None:
            shared = [c for c in s.keys() if c in n and c not in TIMING_COLS]
        diffs = [c for c in shared if n[c] != s[c]]
        if diffs:
            n_mis += 1
        print(f"control {k[0]} {k[1]}x{k[2]} {k[3]} rep{k[4]} "
              f"{k[8][:30]}: {'identical' if not diffs else 'MISMATCH ' + str(diffs[:6])} "
              f"(PTC={n['Planner_Timeout_Count']})", flush=True)
    return n_mis


def run_pool(todo, workers):
    print(f"--- round at {workers} workers: {len(todo)} jobs ---", flush=True)
    t0 = time.perf_counter()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        with mp.Pool(workers) as pool:
            for i, row in enumerate(pool.imap_unordered(run_single, todo), 1):
                w.writerow(row)
                f.flush()
                if i % 10 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)} "
                          f"({(time.perf_counter()-t0)/60:.0f} min)", flush=True)


def cmd_run():
    os.environ["HVBTA_RUN_WALL_LIMIT_S"] = "12000"
    import phase4_sweep as P
    from hvbta.simulation.main_simulation import RUN_WALL_LIMIT_S
    print(f"RUN_WALL_LIMIT_S={RUN_WALL_LIMIT_S}", flush=True)
    jobs, ctrl_keys = build_jobs()
    expected = {key_of_job(j) for j in jobs}
    if not os.path.exists(CSV_PATH):
        os.makedirs(OUT_DIR, exist_ok=True)
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(P.ROW_HEADER)
    valid, attempted = valid_keys()
    todo = [j for j in sorted(jobs, key=lambda x: -x["cost"])
            if key_of_job(j) not in valid]
    ctrl_todo = [j for j in todo if key_of_job(j) in ctrl_keys]
    main_todo = [j for j in todo if key_of_job(j) not in ctrl_keys]
    total = sum(j["cost"] for j in todo)
    print(f"{len(jobs)} jobs, {len(valid)} already valid, {len(todo)} to run "
          f"({len(ctrl_todo)} controls first); "
          f"estimated serial {total/3600:.1f} h", flush=True)

    if ctrl_todo:
        run_pool(ctrl_todo, 8)
        if verify_controls(ctrl_keys):
            print("CONTROL MISMATCH - the single-run driver does not "
                  "reproduce run_group; aborting before the main jobs.",
                  flush=True)
            sys.exit(1)

    for workers in (8, 2, 1):
        valid, attempted = valid_keys()
        todo = [j for j in jobs if key_of_job(j) not in valid
                and key_of_job(j) not in ctrl_keys]
        if not todo:
            break
        run_pool(sorted(todo, key=lambda x: -x["cost"]), workers)
    valid, attempted = valid_keys()
    todo = [j for j in jobs if key_of_job(j) not in valid]
    if todo:
        print(f"STILL INVALID after 1-worker round: {len(todo)}", flush=True)
        for j in todo:
            print("  ", key_of_job(j), flush=True)
        sys.exit(1)

    print(f"all {len(expected)} expected keys valid: "
          f"{len(valid & expected) == len(expected)}", flush=True)
    bad_extra = attempted - valid - expected
    if bad_extra:
        print(f"unexpected invalid extra attempts: {len(bad_extra)}", flush=True)
    n_bad_attempts = sum(1 for r in load_rows(CSV_PATH)
                         if r["Planner_Timeout_Count"] != "0")
    n_rows = sum(1 for _ in open(CSV_PATH, encoding="utf-8")) - 1
    print(f"rows written: {n_rows} "
          f"({n_bad_attempts} invalid attempts superseded)", flush=True)
    verify_controls(ctrl_keys)


def cmd_status():
    valid, attempted = valid_keys()
    print(f"attempts: {len(attempted)}, valid: {len(valid)}")
    n_bad = sum(1 for r in load_rows(CSV_PATH)
                if r["Planner_Timeout_Count"] != "0")
    print(f"invalid attempts: {n_bad}")


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "run":
        cmd_run()
    elif len(sys.argv) >= 2 and sys.argv[1] == "status":
        cmd_status()
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()