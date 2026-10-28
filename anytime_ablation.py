"""Item 5 (final scope): anytime ablation arm — full pool on urgent calls.

Six voting methods with NO early stopping (HVBTA_STALL=0 disables the
stall rule), lak102d, 3 reps, sigma_obs = 0 only (HVBTA_NOISE_SIGMAS=0).
The comparison arm is the main noise sweep's sigma_obs = 0 voting rows;
configs are identical except the stopping rule. run_group also produces
Murty's + perturbed-JV rows, which are invariant to the stopping rule —
they double as a free config-match check (must be bit-identical to the
noise sweep's sigma = 0 optimizer rows).

Output: hvbta/io/results/ablation_anytime_fullpool/phase4_simulation.csv
(pool format, readable by sweep_report.py).

Usage: python anytime_ablation.py
"""
import csv
import os
import sys

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

# Must be set BEFORE importing phase4_sweep (NOISE_SIGMAS is read at
# import time). HVBTA_STALL is read per call inside main_simulation.
os.environ["HVBTA_STALL"] = "0"
os.environ["HVBTA_NOISE_SIGMAS"] = "0"

import phase4_sweep as P


def main():
    out_dir = os.path.join("hvbta", "io", "results", "ablation_anytime_fullpool")
    csv_path = os.path.join(out_dir, "phase4_simulation.csv")
    prof_path = os.path.join(out_dir, "phase4_profiles.csv")
    if os.path.exists(csv_path):
        raise SystemExit(f"{csv_path} already exists - use a fresh directory")
    os.makedirs(out_dir, exist_ok=True)

    jobs = [{"sweep": "noise", "map": "lak102d.map", "nr": n, "nt": n,
             "scenario": s, "rep": rep, "cost": 0}
            for n in (10, 20)
            for s in P.SCENARIOS
            for rep in range(3)]
    conds = P.sweep_conditions("noise")
    assert [c[2] for c in conds] == [0.0], f"expected sigma=0 only, got {conds}"

    with open(csv_path, "w", newline="", encoding="utf-8") as f, \
         open(prof_path, "w", newline="", encoding="utf-8") as pf:
        w = csv.writer(f)
        pw = csv.writer(pf)
        w.writerow(P.ROW_HEADER)
        pw.writerow(P.PROFILES_HEADER)
        for i, job in enumerate(jobs, 1):
            rows, prof_row = P.run_group(job)
            for r in rows:
                w.writerow(r)
            pw.writerow(prof_row)
            f.flush()
            pf.flush()
            print(f"  {i}/{len(jobs)} groups done "
                  f"({job['map']} {job['nr']}x{job['nt']} "
                  f"{job['scenario']} rep{job['rep']})", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
