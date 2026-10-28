"""Item 7 comparison: hyper sweep 250k vs 500k PP expansion caps.
Completion, Plan_Failure_Count, Planner_Cap_Hits, planner-dead by method."""
import csv
import numpy as np


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


for base, tag in [("phase4_final_hyper", "250k"), ("phase4_hyper_500k", "500k")]:
    rows = load(rf"hvbta\io\results\{base}\phase4_simulation.csv")
    timed = [(r["Map"], r["Num Robots"], r["Scenario"], r["Rep"], r["Method"])
             for r in rows if r["Run_Timed_Out"] == "1"]
    print(f"--- {tag} ({base}) ---")
    print("wall-stop rows:", timed)
    hdr = f"{'method':38s} {'compl':>7s} {'fails':>7s} {'cap Hits':>9s} {'dead':>9s}"
    print(hdr)
    for m in sorted({r["Method"] for r in rows}):
        g = [r for r in rows if r["Method"] == m]
        compl = np.mean([float(r["Task_Completion_Fraction"]) for r in g])
        fails = np.mean([float(r["Plan_Failure_Count"]) for r in g])
        caps = np.mean([float(r["Planner_Cap_Hits"]) for r in g])
        dead = sum(1 for r in g if r["Planner_Dead_Tick"] != "-1")
        print(f"{m:38s} {compl:7.3f} {fails:7.1f} {caps:9.1f} {dead:4d}/{len(g):<4d}")