"""Full-sweep calibration probe (2026-10-28 amendment, R2 launch rule).

Direct combat voting single-runs at the R2 grid's heavy combos —
(50,100), (100,50), (100,100), calm, rep 0, borda — run at <= 8 workers
on an idle machine, replacing the 1.8x den001d proxy with direct
measurement. Prints each run's wall time and the calibrated R2 estimate
at 8 workers (serial cost = per-combo measured borda cost scaled to 6
voting rules + 2 optimizer runs at the den001d-combat optimizer ratio,
times 4 scenarios x 3 reps x 2 maps; printed with the raw borda-only
serial as the lower bound). Same code/seeds as the sweep: reuses
rerun_targeted.run_single (HVBTA_RUN_WALL_LIMIT_S = 12000).

Usage: python probe_combat.py
"""

import os
import sys
import time

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

os.environ["HVBTA_RUN_WALL_LIMIT_S"] = "12000"

PROBES = [(100, 100), (50, 100), (100, 50)]


def main():
    import multiprocessing as mp
    import rerun_targeted as T
    import phase4_sweep as P
    from hvbta.simulation.main_simulation import RUN_WALL_LIMIT_S
    print(f"RUN_WALL_LIMIT_S={RUN_WALL_LIMIT_S}", flush=True)
    jobs = []
    for nr, nt in PROBES:
        jobs.append({"map": "combat.map", "nr": nr, "nt": nt,
                     "scenario": "calm", "rep": 0,
                     "mask_level": "none", "mask_policy": "none",
                     "noise_sigma": None,
                     "method": "rank_assignments_borda", "cost": 0})
    exec_i = P._COL_IDX["Execution Time"]
    times = {}
    with mp.Pool(min(8, len(jobs))) as pool:
        for job, row in zip(jobs, pool.imap_unordered(T.run_single, jobs)):
            key = (job["nr"], job["nt"])
            exec_s = float(row[exec_i]) / 1e9
            times[key] = exec_s
            print(f"combat {job['nr']}x{job['nt']} calm rep0 borda: "
                  f"{exec_s:.0f} s (per-run)", flush=True)

    # Calibrated R2 estimate at 8 workers:
    # group cost = 6 voting rules x measured borda + 2 optimizers
    # (optimizer/combat estimate: 1.8x den001d ~ 55 s; small vs voting).
    den = {(50, 100): 282, (100, 100): 185, (100, 50): 146, (50, 50): 82,
           (75, 75): 128}
    ratio = {}
    for k, v in times.items():
        if k in den:
            ratio[k] = v / den[k]
    r = sorted(ratio.values())
    use = r[len(r) // 2] if r else 1.8
    print(f"combat/den001d measured ratios: "
          + ", ".join(f"{k}: {v:.2f}" for k, v in sorted(ratio.items()))
          + f" -> median {use:.2f}", flush=True)
    den_scale = {k: v * use for k, v in den.items()}
    # R2 combos: measured probes + scaled analogs for the light diagonal
    r2 = {(50, 100): times.get((50, 100)),
          (100, 50): times.get((100, 50)),
          (100, 100): times.get((100, 100)),
          (50, 50): den_scale[(50, 50)],
          (30, 30): den_scale[(50, 50)] * (30 / 50),
          (20, 20): den_scale[(50, 50)] * (20 / 50),
          (10, 10): den_scale[(50, 50)] * (10 / 50)}
    serial = 0.0
    heaviest = max(r2.values())
    for (nr, nt), vote_s in r2.items():
        group = 6 * vote_s + 2 * (use * 55)
        serial += group * 4 * 3 * 2
    print(f"calibrated R2 serial: {serial/3600:.1f} h; "
          f"wall at 8 workers ~ {serial/8/3600:.1f} h "
          f"(floor = heaviest group ~ {6 * heaviest / 60:.0f} min)",
          flush=True)
    print(f"LAUNCH RULE: {'LAUNCH R2' if serial/8/3600 <= 12 else 'STOP - OVER 12 h'}",
          flush=True)


if __name__ == "__main__":
    main()