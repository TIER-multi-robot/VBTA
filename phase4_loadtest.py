"""LaCAM timeout-under-load check (user step 1b): run 30 concurrent
groups on the largest configs (combat, combat2, den901d at 100 robots)
at the full worker count and report failure kinds - especially
lacam_timeout at the 1 s limit."""
import copy, os, sys, time, multiprocessing as mp

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

import phase4_sweep as P4


def run_group_with_counts(job):
    import hvbta.pathfinding.MAPF_wrapper as MW
    rows = P4.run_group(job)
    kinds = dict(MW.FAILURE_KIND_COUNTS)
    total_pf = sum(r[6][31] for r in rows)  # Plan_Failure_Count col
    return {"kinds": kinds, "plan_failures": total_pf, "map": job["map"]}


def main():
    limit_s = float(sys.argv[1]) if len(sys.argv) > 1 else None
    if limit_s is not None:
        import hvbta.pathfinding.MAPF_wrapper as MW
        MW.PLANNER_TIME_BUDGET_S = limit_s
        # workers import their own copy; set via env for spawn
        os.environ["HVBTA_LACAM_LIMIT_S"] = str(limit_s)

    jobs = []
    for m in ["combat.map", "combat2.map", "den901d.map"]:
        for scenario in ["calm", "task_churn", "fleet_churn", "overload"]:
            for rep in range(3 if m != "den901d.map" else 2):
                jobs.append({"map": m, "nr": 100, "nt": 100,
                             "scenario": scenario, "rep": rep, "cost": 0})
    jobs = jobs[:30]
    print(f"{len(jobs)} groups at {P4.N_WORKERS} workers, "
          f"LaCAM limit = {os.environ.get('HVBTA_LACAM_LIMIT_S', '1.0')}s")
    t0 = time.perf_counter()
    with mp.Pool(P4.N_WORKERS) as pool:
        results = pool.map(run_group_with_counts, jobs)
    wall = time.perf_counter() - t0
    agg = {}
    for r in results:
        for k, v in r["kinds"].items():
            agg[k] = agg.get(k, 0) + v
    print(f"wall: {wall/60:.1f} min")
    print("failure kinds across all workers:", agg)
    print("timeout count:", agg.get("lacam_timeout", 0))


if __name__ == "__main__":
    main()
