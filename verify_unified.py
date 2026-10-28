"""Schema-unification validation (item 1, 2026-10-27).

Runs ONE small config (lak102d 10x10, rep 0, 4 scenarios, all 5 noise
sigmas, 8 methods = 160 runs) through BOTH entry points:
  - the serial runner (main_simulation.__main__, fresh directory)
  - the pool runner (phase4_sweep.run_group, in-process, fresh directory)
and asserts:
  - identical CSV rows on every shared column EXCEPT the wall-clock
    timing columns (Planning_Time_us, total_reassignment_time,
    Planner_Total_ms, Execution Time) - including the deterministic
    Run ID;
  - identical profiles files (all 7 columns, keyed by Run ID);
  - the expected row widths (86 for the pool schema, 84 for the serial
    schema) and row counts (160 simulation rows, 4 profiles rows).

Usage: python verify_unified.py
"""
import csv
import glob
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, REPO)
os.chdir(REPO)

import phase4_sweep as P

TIMING_COLS = {"Planning_Time_us", "total_reassignment_time",
               "Planner_Total_ms", "Execution Time"}
N_METHODS = 8
N_SIGMAS = 5
N_SCEN = 4
N_ROWS = N_METHODS * N_SIGMAS * N_SCEN   # 160


def run_serial():
    serial_dir = "verify_unified_serial"
    res_dir = os.path.join("hvbta", "io", "results", serial_dir)
    if not os.path.isdir(res_dir):
        env = dict(os.environ)
        env["PATH"] = r"C:\msys64\ucrt64\bin;" + env.get("PATH", "")
        env["PYTHONPATH"] = "."
        env["HVBTA_OUTDIR"] = serial_dir
        env["HVBTA_REPS"] = "1"
        env["HVBTA_SIZES"] = "10"
        env["HVBTA_NOISE"] = "1"
        env["HVBTA_PJV_SIGMA"] = "0.05"
        env.pop("HVBTA_PO", None)
        env.pop("HVBTA_PLANNER", None)
        env.pop("HVBTA_STALL", None)
        env.pop("HVBTA_NOISE_SIGMAS", None)
        print("running the serial runner (fresh dir)...")
        r = subprocess.run(
            [sys.executable, "-m", "hvbta.simulation.main_simulation"],
            env=env, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            raise SystemExit(f"serial run failed:\n{r.stderr[-4000:]}")
    sim = sorted(glob.glob(os.path.join(res_dir, "*simulation_*.csv")))
    prof = sorted(glob.glob(os.path.join(res_dir, "*profiles_*.csv")))
    assert len(sim) == 1 and len(prof) == 1, (sim, prof)
    with open(sim[0], newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        s_header = next(reader)
        s_rows = list(reader)
    with open(prof[0], newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        s_prof = list(reader)
    return s_header, s_rows, s_prof


def run_pool():
    pool_dir = os.path.join("hvbta", "io", "results", "verify_unified_pool")
    csv_path = os.path.join(pool_dir, "phase4_simulation.csv")
    if os.path.exists(csv_path):
        raise SystemExit(f"{csv_path} already exists - use a fresh directory")
    os.makedirs(pool_dir, exist_ok=True)
    jobs = [{"sweep": "noise", "map": "lak102d.map", "nr": 10, "nt": 10,
             "scenario": s, "rep": 0, "cost": 0}
            for s in P.SCENARIOS]
    p_rows, p_prof = [], []
    for job in jobs:
        rows, prof_row = P.run_group(job)
        p_rows.extend(rows)
        p_prof.append(prof_row)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(P.ROW_HEADER)
        for r in p_rows:
            w.writerow(r)
    prof_path = os.path.join(pool_dir, "phase4_profiles.csv")
    with open(prof_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(P.PROFILES_HEADER)
        for r in p_prof:
            w.writerow(r)
    return p_rows, p_prof


def main():
    s_header, s_rows, s_prof = run_serial()
    p_rows, p_prof = run_pool()

    print(f"serial rows: {len(s_rows)} (header {len(s_header)} cols); "
          f"pool rows: {len(p_rows)} (header {len(P.ROW_HEADER)} cols)")
    assert len(s_header) == 84, f"serial header is {len(s_header)} cols, expected 84"
    assert all(len(r) == 86 for r in p_rows), "pool row width != 86"
    assert all(len(r) == 84 for r in s_rows), "serial row width != 84"
    assert len(s_rows) == len(p_rows) == N_ROWS, (len(s_rows), len(p_rows))

    # The pool schema = Map + Rep + the serial 84 columns.
    assert P.ROW_HEADER == ["Map", "Rep"] + s_header, "header mismatch"

    # Key rows by (Method, Scenario, Obs_Noise_Sigma); compare every shared
    # column except wall-clock timing - INCLUDING the deterministic Run ID.
    idx = {name: i for i, name in enumerate(P.ROW_HEADER)}
    shared = [name for name in s_header if name not in TIMING_COLS]
    serial_by_key = {}
    for r in s_rows:
        sidx = {name: i for i, name in enumerate(s_header)}
        key = (r[sidx["Method"]], r[sidx["Scenario"]],
               r[sidx["Obs_Noise_Sigma"]])
        assert key not in serial_by_key, f"duplicate serial key {key}"
        serial_by_key[key] = r
    diffs = []
    for pr in p_rows:
        # Pool rows are in-memory (ints/floats); serial rows come from the
        # CSV (strings). Normalize the key through str() so like values
        # match ("0.0" == str(0.0)).
        key = (str(pr[idx["Method"]]), str(pr[idx["Scenario"]]),
               str(pr[idx["Obs_Noise_Sigma"]]))
        sr = serial_by_key.get(key)
        if sr is None:
            diffs.append((key, "MISSING SERIAL ROW", "", ""))
            continue
        for name in shared:
            pv, sv = pr[idx[name]], sr[sidx[name]]
            if str(pv) != str(sv):
                try:
                    if abs(float(pv) - float(sv)) <= 1e-9 * max(1.0, abs(float(sv))):
                        continue
                except ValueError:
                    pass
                diffs.append((key, name, pv, sv))
    print(f"compared {len(p_rows)} rows x {len(shared)} shared columns "
          f"(timing excluded): {len(diffs)} mismatches")
    for d in diffs[:10]:
        print(f"  MISMATCH {d[0]} col {d[1]}: pool={d[2]!r} serial={d[3]!r}")

    # Profiles: identical files (all 7 columns).
    print(f"serial profiles rows: {len(s_prof)}; pool profiles rows: {len(p_prof)}")
    assert len(s_prof) == len(p_prof) == N_SCEN, (len(s_prof), len(p_prof))
    prof_diffs = []
    pool_prof_by_id = {str(r[0]): [str(x) for x in r] for r in p_prof}
    for sr in s_prof:
        pr = pool_prof_by_id.get(sr[0])
        if pr is None:
            prof_diffs.append((sr[0], "missing in pool", "", ""))
        elif sr != pr:
            for i, (a, b) in enumerate(zip(sr, pr)):
                if a != b:
                    prof_diffs.append((sr[0], P.PROFILES_HEADER[i], b, a))
    print(f"profiles comparison (all 7 columns): {len(prof_diffs)} mismatches")
    for d in prof_diffs[:10]:
        print(f"  PROFILE MISMATCH run={d[0]} field={d[1]}: "
              f"pool={d[2]!r} serial={d[3]!r}")

    ok = not diffs and not prof_diffs
    print("\nRESULT:",
          "IDENTICAL (excluding timing columns) - unification validated"
          if ok else "MISMATCHES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
