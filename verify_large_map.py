"""Item 2 (2026-10-27): verify the noise sweep's reconstruction on a large map.

Runs ONE lak307d group set (50x50, rep 0, all 4 scenarios, sigma_obs = 0
only) SERIALLY (in-process, one job at a time - no pool) through the now-
unified pool runner, and compares every non-timing column of the new
directly-written rows against phase4_final_noise's expanded rows (whose
initial-quality columns were regenerated from seeds by expand_results.py).
This validates the reconstruction on a large map, not just lak102d.

Also compares the profiles rows for those groups.

Runs alongside the hyper sweep: results are deterministic, only the
excluded wall-clock timing columns are load-sensitive.

Usage: python verify_large_map.py
"""
import csv
import os
import sys

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

# sigma_obs = 0 only; read by phase4_sweep at import time.
os.environ["HVBTA_NOISE_SIGMAS"] = "0"

import phase4_sweep as P

TIMING_COLS = {"Planning_Time_us", "total_reassignment_time",
               "Planner_Total_ms", "Execution Time"}


def main():
    noise_dir = os.path.join("hvbta", "io", "results", "phase4_final_noise")
    exp_path = os.path.join(noise_dir, "phase4_expanded.csv")
    prof_path = os.path.join(noise_dir, "phase4_profiles.csv")
    with open(exp_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == P.ROW_HEADER, (
            "expanded header != unified pool header")
        exp_rows = list(reader)
    sel = [r for r in exp_rows
           if r["Map"] == "lak307d.map" and r["Num Robots"] == "50"
           and r["Num Tasks"] == "50" and r["Rep"] == "0"
           and r["Obs_Noise_Sigma"] == "0.0"]
    print(f"expanded rows for (lak307d 50x50, rep 0, sigma 0): {len(sel)} "
          f"(expect 32 = 4 scenarios x 8 methods)")
    assert len(sel) == 32, len(sel)

    jobs = [{"sweep": "noise", "map": "lak307d.map", "nr": 50, "nt": 50,
             "scenario": s, "rep": 0, "cost": 0}
            for s in P.SCENARIOS]
    conds = P.sweep_conditions("noise")
    assert [c[2] for c in conds] == [0.0], conds

    idx = {name: i for i, name in enumerate(P.ROW_HEADER)}
    shared = [name for name in P.ROW_HEADER if name not in TIMING_COLS]
    exp_by_key = {}
    for r in sel:
        key = (r["Method"], r["Scenario"])
        assert key not in exp_by_key, key
        exp_by_key[key] = r

    diffs = []
    new_prof = {}
    for job in jobs:
        rows, prof_row = P.run_group(job)
        new_prof[prof_row[0]] = prof_row
        for pr in rows:
            key = (str(pr[idx["Method"]]), str(pr[idx["Scenario"]]))
            er = exp_by_key.get(key)
            if er is None:
                diffs.append((job["scenario"], key, "MISSING EXPANDED ROW", "", ""))
                continue
            for name in shared:
                pv, ev = str(pr[idx[name]]), er[name]
                if pv != ev:
                    try:
                        if abs(float(pv) - float(ev)) <= 1e-9 * max(1.0, abs(float(ev))):
                            continue
                    except ValueError:
                        pass
                    diffs.append((job["scenario"], key, name, pv, ev))
        print(f"  compared group {job['scenario']} "
              f"(cumulative mismatches: {len(diffs)})", flush=True)
    print(f"compared 32 rows x {len(shared)} non-timing columns: "
          f"{len(diffs)} mismatches")
    for d in diffs[:10]:
        print(f"  MISMATCH {d}")

    # Profiles rows for the four groups.
    with open(prof_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        exp_prof = {r[0]: r for r in reader}
    prof_diffs = []
    for rid, pr in new_prof.items():
        er = exp_prof.get(rid)
        if er is None:
            prof_diffs.append((rid, "missing in expanded profiles"))
        elif er != [str(x) for x in pr]:
            prof_diffs.append((rid, "profile content differs"))
    print(f"profiles comparison for the 4 groups: {len(prof_diffs)} mismatches")

    ok = not diffs and not prof_diffs
    print("\nRESULT:",
          "RECONSTRUCTION VERIFIED on lak307d 50x50 (non-timing columns identical)"
          if ok else "MISMATCHES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()