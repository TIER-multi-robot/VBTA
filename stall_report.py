"""Stall-rule report driver (2026-10-27, final-scope item 1).

Measures, on the lak102d pilot config WITHOUT reading any completion
numbers: how often the stall rule fires on urgent calls, and the
distribution of candidates ranked before stopping, by unassigned-robot
bin.

Config: the v9 PO pilot config - lak102d [10,20]x[10,20], 4 scenarios,
5 mask conditions, 3 reps, 6 voting methods (the stall rule only affects
voting; optimizers never make voting calls). V.POOL_STATS_SINK archives
one record per voting allocation call (rule, pool sizes, stall_k, fired
flag, call dims). Passive collection: rows are bit-identical to the same
config run without the sink.

Usage:
  python stall_report.py run <outdir>     # collect records (fresh dir)
  python stall_report.py report <outdir>  # aggregate + print report
"""
import copy
import json
import multiprocessing as mp
import os
import sys
import time
import zlib

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

N_WORKERS = max(1, (os.cpu_count() or 4) - 2)

CALIB_CONFIG = [
    ("lak102d.map", [10, 20], [10, 20], 3),
]
SCENARIOS = ["calm", "task_churn", "fleet_churn", "overload"]

VOTING_METHODS = [
    "rank_assignments_borda",
    "rank_assignments_approval",
    "rank_assignments_majority_judgment",
    "rank_assignments_cumulative_voting",
    "rank_assignments_condorcet_method",
    "rank_assignments_range",
]


def run_group(job):
    """Worker: all 5 PO conditions x 6 voting methods for one
    (nr, nt, scenario, rep) group, with the POOL_STATS sink on."""
    import hvbta.generation as G
    import hvbta.suitability as S
    import hvbta.allocators.voting as V
    from hvbta.simulation.main_simulation import (
        benchmark_simulation, build_event_schedule, horizon_for_event_count,
        _derived_seed, _seed_rngs, REGIMES,
    )
    from hvbta.pathfinding.CBS_helper import load_map, create_obstacle_list
    from hvbta.partial_observability import PO_CONDITIONS

    map_name, nr, nt, scenario, rep = (job["map"], job["nr"], job["nt"],
                                        job["scenario"], job["rep"])
    out_path = job["out"]
    TRANSIENT_FLOOR, TARGET_EVENT_COUNT_K = 200, 125
    grid = load_map(os.path.join("MAPF_benchmark_maps", map_name))
    HYP = (len(grid) ** 2 + len(grid[0]) ** 2) ** 0.5
    map_dict = {"dimension": (len(grid), len(grid[0])),
                "obstacles": create_obstacle_list(grid)}
    nc = max(nr, nt) * 6
    sm_name = "evaluate_suitability_balanced"
    seed_tuple = (map_name, nr, nt, nc, sm_name, rep, scenario)
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

    regime = REGIMES[scenario]
    provisional = max(TRANSIENT_FLOOR, int(HYP * 3))
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
    max_steps = max(TRANSIENT_FLOOR, horizon, int(HYP * 1.5 * extender))
    schedule = schedule[:max_steps]

    sm = S.evaluate_suitability_balanced
    records = []
    for mask_level, mask_policy in PO_CONDITIONS:
        from hvbta.partial_observability import PartialObservabilityState
        po_cond = PartialObservabilityState(schedule_seed, mask_level, mask_policy)
        matrix = S.calculate_suitability_matrix(
            robots, tasks, po_cond.masked_scorer(sm), HYP)
        for method_name in VOTING_METHODS:
            method_fn = getattr(V, method_name)
            _seed_rngs(_derived_seed(schedule_seed, "initial", method_name))
            out, score, length, per = V.assign_tasks_with_voting(
                robots, tasks, matrix, nc, method_fn,
                tie_seed=_derived_seed(schedule_seed, "ties", "initial", method_name))
            # Archive this run's voting-call records via the sink.
            V.POOL_STATS_SINK = []
            try:
                benchmark_simulation(
                    output=out, robots=copy.deepcopy(robots),
                    tasks=copy.deepcopy(tasks),
                    num_candidates=nc, voting_method=method_fn, grid=grid,
                    map_dict=map_dict, suitability_method=sm,
                    suitability_matrix=matrix,
                    max_time_steps=max_steps, add_tasks=True, add_robots=True,
                    remove_robots=True, event_schedule=schedule,
                    mask_level=mask_level, mask_policy=mask_policy,
                    schedule_seed=schedule_seed,
                    run_seed=_derived_seed(schedule_seed, "run", method_name),
                    planner="lacam", scenario=scenario,
                    scheduled_event_count=0, horizon_binding_term="",
                )
            finally:
                calls = V.POOL_STATS_SINK or []
                V.POOL_STATS_SINK = None
            base = {"nr": nr, "nt": nt, "scenario": scenario, "rep": rep,
                    "mask_level": mask_level, "mask_policy": mask_policy,
                    "method": method_name}
            for c in calls:
                rec = dict(base)
                rec.update({k: c[k] for k in (
                    "rule", "full", "ranked", "threshold", "stall_k",
                    "stall_fired", "budget_fired", "n_robots", "n_tasks")})
                records.append(rec)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return out_path, len(records)


def _bin_label(n):
    for lo, hi in [(1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128)]:
        if lo <= n <= hi:
            return f"{lo}-{hi}"
    return "129+"


def run_cmd(outdir):
    res_dir = os.path.join("hvbta", "io", "results", outdir)
    if os.path.exists(res_dir):
        raise SystemExit(f"{res_dir} exists - use a fresh directory")
    os.makedirs(res_dir, exist_ok=True)
    jobs = []
    for map_name, r_sizes, t_sizes, reps in CALIB_CONFIG:
        for nr in r_sizes:
            for nt in t_sizes:
                for scenario in SCENARIOS:
                    for rep in range(reps):
                        jobs.append({
                            "map": map_name, "nr": nr, "nt": nt,
                            "scenario": scenario, "rep": rep, "out":
                            os.path.join(res_dir,
                                         f"stall_{nr}x{nt}_{scenario}_rep{rep}.jsonl"),
                        })
    print(f"{len(jobs)} groups; workers={min(N_WORKERS, len(jobs))}")
    t0 = time.perf_counter()
    with mp.Pool(min(N_WORKERS, len(jobs))) as pool:
        for i, (path, n) in enumerate(pool.imap_unordered(run_group, jobs), 1):
            print(f"  [{i}/{len(jobs)}] {os.path.basename(path)} ({n} calls, "
                  f"{time.perf_counter()-t0:.0f}s)", flush=True)
    print("DONE - records in", res_dir)


def report_cmd(outdir):
    import numpy as np
    res_dir = os.path.join("hvbta", "io", "results", outdir)
    records = []
    for fn in sorted(os.listdir(res_dir)):
        if fn.startswith("stall_") and fn.endswith(".jsonl"):
            with open(os.path.join(res_dir, fn), encoding="utf-8") as f:
                for line in f:
                    records.append(json.loads(line))
    print(f"loaded {len(records)} voting-call records "
          f"(completion numbers never read)")

    urgent = [r for r in records if r["stall_k"] is not None]
    non_urgent = [r for r in records if r["stall_k"] is None]

    def fmt(a):
        a = np.asarray(a, dtype=float)
        if not a.size:
            return "n/a"
        return (f"n={a.size} mean={a.mean():.1f} sd={a.std():.1f} "
                f"min={a.min():.0f} p25={np.percentile(a,25):.0f} "
                f"p50={np.percentile(a,50):.0f} p75={np.percentile(a,75):.0f} "
                f"p95={np.percentile(a,95):.0f} max={a.max():.0f}")

    fired = sum(1 for r in urgent if r["stall_fired"])
    print(f"\n== URGENT calls (stall_k=5) ==")
    print(f"  total: {len(urgent)}   stall fired: {fired} "
          f"({fired/len(urgent)*100 if urgent else 0:.1f}%)   "
          f"pool exhausted: {len(urgent)-fired}")
    print(f"  candidates ranked per call: {fmt([r['ranked'] for r in urgent])}")
    print(f"  full pool size per call:    {fmt([r['full'] for r in urgent])}")
    print(f"  ranked among FIRED calls:   "
          f"{fmt([r['ranked'] for r in urgent if r['stall_fired']])}")

    print(f"\n== by unassigned-robot bin (n_robots = call's unassigned robots) ==")
    bins = {}
    for r in urgent:
        bins.setdefault(_bin_label(r["n_robots"]), []).append(r)
    for b in sorted(bins, key=lambda s: int(s.split("-")[0])):
        g = bins[b]
        f = sum(1 for r in g if r["stall_fired"])
        print(f"  robots {b:8s} calls={len(g):5d} fired={f:5d} "
              f"({f/len(g)*100:5.1f}%)  ranked: "
              f"{fmt([r['ranked'] for r in g])}")

    print(f"\n== sanity: NON-URGENT calls (full pool by construction) ==")
    full_ranked = sum(1 for r in non_urgent if r["ranked"] == r["full"])
    print(f"  total: {len(non_urgent)}   ranked==full: {full_ranked} "
          f"({full_ranked/len(non_urgent)*100 if non_urgent else 0:.1f}%)")

    out_json = os.path.join(res_dir, "stall_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "n_records": len(records), "n_urgent": len(urgent),
            "stall_fired": fired,
            "fire_rate": fired / len(urgent) if urgent else None,
            "urgent_ranked_stats": {
                "mean": float(np.mean([r["ranked"] for r in urgent])),
                "sd": float(np.std([r["ranked"] for r in urgent])),
                "percentiles": {str(p): float(np.percentile(
                    [r["ranked"] for r in urgent], p)) for p in (5, 25, 50, 75, 95)},
            },
            "by_robot_bin": {
                b: {"n": len(g), "fired": sum(1 for r in g if r["stall_fired"]),
                    "ranked_mean": float(np.mean([r["ranked"] for r in g])),
                    "ranked_p50": float(np.percentile([r["ranked"] for r in g], 50))}
                for b, g in bins.items()},
        }, f, indent=2)
    print("report JSON:", out_json)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "pilot_stall_report"
    if cmd == "run":
        run_cmd(outdir)
    elif cmd == "report":
        report_cmd(outdir)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
