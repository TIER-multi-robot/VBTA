"""Preregistered H2 analysis (recovery) - DATA MAPPING PENDING APPROVAL.

Implements docs/ANALYSIS_PLAN.md H2 exactly, with the data mapping
PROPOSED (not yet approved - DO NOT RUN before the user approves):

  Data: phase4_final_hyper (the PP planner-failure stress sweep) - the
  only v2 sweep with nonzero Plan_Failure_Count (noise, PO, balanced and
  constrained all have 0 failures), so "voting has fewer plan failures"
  is testable only there. Group = (map, nr, nt, scenario, rep); the
  sweep's single condition is (None, None, None). Planner-dead runs (85
  of 1,920) are included - they are the phenomenon under test.
  Comparisons: Murty's vs EACH of the 6 voting rules (H2 says "voting",
  not the ordinal subset - that restriction is H1-specific).
  Metrics and directions:
    Plan_Failure_Count        diff = murty - rule (positive = voting fewer)
    Adaptation_Latency_Mean   diff = murty - rule (positive = voting lower)
    Task_Completion_Fraction  diff = rule - murty; the claim is "no worse":
      supported (better)   if Holm p < 0.05 and median > 0
      supported (no worse) if not significantly negative
      not supported (worse) if Holm p < 0.05 and median < 0
  Holm across the 6 comparisons within each (metric, reporting level);
  reported per scenario and pooled. Conventions: analysis/paired_stats.py
  (two-sided Wilcoxon, median + bootstrap 95% CI, zero-pair counts).
"""

import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0])

from paired_stats import (ALPHA, group_key, holm, index_rows, load,
                          paired_diffs, report_family, verdict,
                          wilcoxon_result)

DATA = "hvbta/io/results/phase4_final_hyper/phase4_simulation.csv"
GCOLS = ("Map", "Num Robots", "Num Tasks", "Scenario", "Rep")
SCENARIOS = ("calm", "task_churn", "fleet_churn", "overload")
MURTY = "murtys_task_allocation"
VOTING = ("rank_assignments_borda", "rank_assignments_approval",
          "rank_assignments_majority_judgment",
          "rank_assignments_cumulative_voting",
          "rank_assignments_condorcet_method", "rank_assignments_range")
LABEL = {"rank_assignments_borda": "Borda",
         "rank_assignments_approval": "Approval",
         "rank_assignments_majority_judgment": "Majority Judgment",
         "rank_assignments_cumulative_voting": "Cumulative",
         "rank_assignments_condorcet_method": "Condorcet",
         "rank_assignments_range": "Range",
         "perturbed_jv_task_allocation": "Perturbed JV"}


def completion_verdict(median, adj_p):
    """H2's completion component is 'no worse', never an equivalence claim:
    'not supported' iff voting is significantly lower; otherwise the claim
    is 'not shown worse' (which asserts nothing about equality)."""
    if adj_p is not None and adj_p < ALPHA and median < 0:
        return "not supported (worse)"
    return "not shown worse"


def run_metric(index, groups, col, rule_is_a, label_prefix="h2"):
    results = []
    for rule in VOTING:
        a, b = (rule, MURTY) if rule_is_a else (MURTY, rule)
        d = paired_diffs(index, groups, col, a, b)
        r = wilcoxon_result(d, f"{label_prefix}:{col}:{rule}")
        r["label"] = f"Murty's vs {LABEL[rule]}"
        results.append(r)
    return results


def apply_holm(fam):
    valid = [i for i, r in enumerate(fam) if r["p"] is not None]
    if valid:
        for i, a in zip(valid, holm([fam[i]["p"] for i in valid])):
            fam[i]["adj_p"] = a
    for r in fam:
        r.setdefault("adj_p", None)
    return fam


def main():
    rows = load(DATA)
    index = index_rows(rows, GCOLS)
    groups = sorted({group_key(r, GCOLS) for r in rows})
    print("H2 (preregistered) -- recovery [mapping approved]")
    print(f"data: {DATA} ({len(rows)} rows, {len(groups)} groups)")
    print("comparisons: Murty's vs each of the 6 voting rules; "
          "Holm across 6 per (metric, level); per scenario + pooled")
    print("completion component: 'not supported (worse)' iff Holm p<0.05 "
          "and median<0; otherwise 'not shown worse' (no equivalence "
          "claimed)")
    print("H2 overall per rule = supported only if ALL THREE components "
          "hold (failures sig. lower, latency sig. lower, completion not "
          "shown worse)")
    print()
    dead = {}
    for m in VOTING + (MURTY, "perturbed_jv_task_allocation"):
        dead[LABEL.get(m, m)] = sum(
            1 for r in rows
            if r["Method"] == m and r["Planner_Dead_Tick"] != "-1")
    print("planner-dead runs (hit the 150 failed-call budget), per "
          f"method, included in the tests: {dead}")
    print(f"total planner-dead runs: {sum(dead.values())} of {len(rows)}")
    print()

    levels = [("Pooled", groups)] + \
        [(f"Scenario: {sc}", [g for g in groups if g[3] == sc])
         for sc in SCENARIOS]
    for level_name, sub in levels:
        fams = {}
        for title, col, rule_is_a, vfn in (
                ("Plan_Failure_Count (fewer = voting better)",
                 "Plan_Failure_Count", False, verdict),
                ("Adaptation_Latency_Mean (lower = voting better)",
                 "Adaptation_Latency_Mean", False, verdict),
                ("Task_Completion_Fraction (no worse = voting ok)",
                 "Task_Completion_Fraction", True, completion_verdict)):
            fams[col] = apply_holm(run_metric(index, sub, col, rule_is_a))
            report_family(title, f"{level_name} (n = {len(sub)})",
                          fams[col], vfn)
        print(f"H2 OVERALL | {level_name}")
        for i, rule in enumerate(VOTING):
            f, l, c = (fams["Plan_Failure_Count"][i],
                       fams["Adaptation_Latency_Mean"][i],
                       fams["Task_Completion_Fraction"][i])
            ok_f = (f["adj_p"] is not None and f["adj_p"] < ALPHA
                    and f["median"] > 0)
            ok_l = (l["adj_p"] is not None and l["adj_p"] < ALPHA
                    and l["median"] > 0)
            ok_c = (completion_verdict(c["median"], c["adj_p"])
                    != "not supported (worse)")
            print(f"  {LABEL[rule]:<18} failures={ok_f} latency={ok_l} "
                  f"completion_ok={ok_c} "
                  f"-> {'SUPPORTED' if ok_f and ok_l and ok_c else 'not supported'}")
        print()


if __name__ == "__main__":
    main()