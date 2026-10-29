"""Preregistered H3 analysis (aggregation vs randomness) - DATA MAPPING
PENDING APPROVAL.

Implements docs/ANALYSIS_PLAN.md H3 exactly, with the data mapping
PROPOSED (not yet approved - DO NOT RUN before the user approves):

  H3: "Voting's recovery advantage over Murty's also holds against
  perturbed JV. If perturbed JV matches voting, the advantage comes from
  randomness, not aggregation."
  Data: phase4_final_hyper (same rationale as H2 - the only sweep with
  nonzero Plan_Failure_Count). Group = (map, nr, nt, scenario, rep).
  Operationalization: the same recovery metrics and directions as H2,
  with perturbed JV in Murty's place: perturbed JV vs EACH of the 6
  voting rules.
  Metrics and directions (diff = perturbed JV - rule unless noted):
    Plan_Failure_Count        positive = voting fewer  (voting advantage)
    Adaptation_Latency_Mean   positive = voting lower
    Task_Completion_Fraction  diff = rule - perturbed JV; "matches voting"
      reads out as the standard verdict on this metric: significant
      positive = voting better than the randomized baseline (aggregation
      story); not significant = perturbed JV matches voting (randomness
      story); significant negative = perturbed JV better.
  Holm across the 6 comparisons within each (metric, reporting level);
  reported per scenario and pooled. Conventions: analysis/paired_stats.py.
"""

import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0])

from paired_stats import (group_key, index_rows, load, paired_diffs,
                          report_family, verdict, wilcoxon_result)

DATA = "hvbta/io/results/phase4_final_hyper/phase4_simulation.csv"
GCOLS = ("Map", "Num Robots", "Num Tasks", "Scenario", "Rep")
SCENARIOS = ("calm", "task_churn", "fleet_churn", "overload")
PJV = "perturbed_jv_task_allocation"
VOTING = ("rank_assignments_borda", "rank_assignments_approval",
          "rank_assignments_majority_judgment",
          "rank_assignments_cumulative_voting",
          "rank_assignments_condorcet_method", "rank_assignments_range")
LABEL = {"rank_assignments_borda": "Borda",
         "rank_assignments_approval": "Approval",
         "rank_assignments_majority_judgment": "Majority Judgment",
         "rank_assignments_cumulative_voting": "Cumulative",
         "rank_assignments_condorcet_method": "Condorcet",
         "rank_assignments_range": "Range"}


def run_metric(index, groups, col, rule_is_a):
    results = []
    for rule in VOTING:
        a, b = (rule, PJV) if rule_is_a else (PJV, rule)
        d = paired_diffs(index, groups, col, a, b)
        r = wilcoxon_result(d, f"h3:{col}:{rule}")
        r["label"] = f"pJV vs {LABEL[rule]}"
        results.append(r)
    return results


def main():
    rows = load(DATA)
    index = index_rows(rows, GCOLS)
    groups = sorted({group_key(r, GCOLS) for r in rows})
    print("H3 (preregistered) -- aggregation vs randomness "
          "[mapping approved]")
    print(f"data: {DATA} ({len(rows)} rows, {len(groups)} groups)")
    print("comparisons: perturbed JV vs each of the 6 voting rules; "
          "Holm across 6 per (metric, level); per scenario + pooled")
    print("verdicts are the standard rule; a non-significant result is "
          "'not supported (no advantage shown)' and NEVER an equivalence "
          "claim - the plan's aggregation-vs-randomness reading is "
          "interpretation only, reported alongside the numbers")
    print()
    for title, col, rule_is_a in (
            ("Plan_Failure_Count (positive = voting fewer)",
             "Plan_Failure_Count", False),
            ("Adaptation_Latency_Mean (positive = voting lower)",
             "Adaptation_Latency_Mean", False),
            ("Task_Completion_Fraction (positive = voting better)",
             "Task_Completion_Fraction", True)):
        report_family(title, f"Pooled (n groups = {len(groups)})",
                      run_metric(index, groups, col, rule_is_a))
        for sc in SCENARIOS:
            sub = [g for g in groups if g[3] == sc]
            report_family(title, f"Scenario: {sc} (n = {len(sub)})",
                          run_metric(index, sub, col, rule_is_a))


if __name__ == "__main__":
    main()