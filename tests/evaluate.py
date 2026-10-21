"""
VBTA Evaluation Script
======================
Runs the full claim-by-claim evaluation from CLAIMS_AND_EVIDENCE.md.

Usage:
    python evaluate.py --dataset maps_balanced2
    python evaluate.py --dataset Full-Sweep
    python evaluate.py --dataset maps_balanced2 --output results_eval/

Outputs:
    - CSV summary tables (one per claim)
    - PNG figures (Pareto front, scaling curves, regime validation)
    - Console report with statistical tests

Datasets:
    maps_balanced2 : 9 maps, balanced fleet pairing, tuned task_churn (0.15/0.08)
                     Best for Claims 1-3, 6 (quality, fairness, adaptation, regimes)
    Full-Sweep     : 9 maps, all fleet sizes on all maps, aggressive task_churn (0.30/0.15)
                     Best for Claim 4 (scaling — needs the full fleet-size axis)
    maps_balanced  : INCONSISTENT — mixed task_churn rates across timestamps. Use with caution.
"""

import argparse
import glob
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")  # non-interactive
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join("hvbta", "io", "results")
DATASETS = {
    "maps_balanced2": os.path.join(RESULTS_DIR, "maps_balanced2"),
    "Full-Sweep": os.path.join(RESULTS_DIR, "Full-Sweep"),
    "maps_balanced": os.path.join(RESULTS_DIR, "maps_balanced"),
}

# Method grouping
VOTING_METHODS = [
    "rank_assignments_borda",
    "rank_assignments_approval",
    "rank_assignments_majority_judgment",
    "rank_assignments_cumulative_voting",
    "rank_assignments_condorcet_method",
    "rank_assignments_range",
]
OPTIMIZATION_METHODS = [
    "cbba_task_allocation",
    "ssia_task_allocation",
    "ilp_task_allocation",
    "jv_task_allocation",
]
ALL_METHODS = VOTING_METHODS + OPTIMIZATION_METHODS

# Pretty names for tables/figures
METHOD_LABELS = {
    "rank_assignments_borda": "Borda",
    "rank_assignments_approval": "Approval",
    "rank_assignments_majority_judgment": "Majority Judge",
    "rank_assignments_cumulative_voting": "Cumulative",
    "rank_assignments_condorcet_method": "Condorcet",
    "rank_assignments_range": "Range",
    "cbba_task_allocation": "CBBA",
    "ssia_task_allocation": "SSIA",
    "ilp_task_allocation": "ILP",
    "jv_task_allocation": "JV",
}

SCENARIO_ORDER = ["calm", "task_churn", "fleet_churn", "overload"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_dataset(dataset_name):
    """Load all simulation CSVs from a dataset directory into one DataFrame."""
    data_dir = DATASETS[dataset_name]
    if not os.path.exists(data_dir):
        print(f"ERROR: dataset directory not found: {data_dir}")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(data_dir, "*simulation*.csv")))
    if not files:
        print(f"ERROR: no simulation CSVs in {data_dir}")
        sys.exit(1)

    dfs = []
    for f in files:
        map_name = os.path.basename(f).split("simulation_")[-1].replace(".csv", "")
        df = pd.read_csv(f)
        df["Map"] = map_name
        dfs.append(df)
        print(f"  loaded {os.path.basename(f)}: {len(df)} rows")

    df = pd.concat(dfs, ignore_index=True)
    print(f"  total: {len(df)} rows, {len(df.columns)} columns")

    # Add derived columns
    df["Method_Label"] = df["Method"].map(METHOD_LABELS).fillna(df["Method"])
    df["Is_Voting"] = df["Method"].isin(VOTING_METHODS)
    df["Is_Optimization"] = df["Method"].isin(OPTIMIZATION_METHODS)

    # latency per event (rate-normalized across regimes)
    df["Latency_Per_Event"] = df.apply(
        lambda r: r["Adaptation_Latency_Mean"] / max(r["Scheduled_Event_Count"], 1), axis=1
    )

    # attempt ratio
    df["Attempt_Ratio"] = df.apply(
        lambda r: r["Tasks_Attempted"] / max(r["Tasks_Spawned"], 1), axis=1
    )

    return df


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------
def bootstrap_ci(data, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval of the mean."""
    if len(data) < 2:
        return np.nan, np.nan
    boot_means = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return lo, hi


def paired_wilcoxon(voting_vals, opt_vals):
    """Paired Wilcoxon signed-rank test. Returns (statistic, p_value)."""
    if len(voting_vals) < 5 or len(opt_vals) < 5:
        return np.nan, np.nan
    min_len = min(len(voting_vals), len(opt_vals))
    try:
        stat, p = stats.wilcoxon(voting_vals[:min_len], opt_vals[:min_len])
        return stat, p
    except Exception:
        return np.nan, np.nan


def aggregate_by_method(df, metric, groupby=None):
    """Compute mean + bootstrap CI for each method."""
    if groupby is None:
        groupby = ["Method_Label"]
    results = []
    for keys, grp in df.groupby(groupby):
        vals = grp[metric].dropna().values
        mean = np.mean(vals) if len(vals) else np.nan
        lo, hi = bootstrap_ci(vals)
        row = dict(zip(groupby if isinstance(groupby, list) else [groupby], keys if isinstance(keys, tuple) else [keys]))
        row.update({f"{metric}_mean": mean, f"{metric}_ci_lo": lo, f"{metric}_ci_hi": hi, "n": len(vals)})
        results.append(row)
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Claim 1: Voting ~= optimization on quality
# ---------------------------------------------------------------------------
def claim1_quality(df, out_dir):
    print("\n" + "=" * 70)
    print("CLAIM 1: Voting ~= optimization on solution quality")
    print("=" * 70)

    metrics = ["Total Score", "Cumulative_Reassignment_Quality", "Mean_Assignment_Suitability"]
    tables = []

    for scenario in SCENARIO_ORDER:
        sdf = df[df["Scenario"] == scenario]
        if sdf.empty:
            continue
        print(f"\n  Scenario: {scenario}")
        for metric in metrics:
            voting = sdf[sdf["Is_Voting"]][metric].dropna().values
            opt = sdf[sdf["Is_Optimization"]][metric].dropna().values
            v_mean = np.mean(voting) if len(voting) else 0
            o_mean = np.mean(opt) if len(opt) else 0
            v_lo, v_hi = bootstrap_ci(voting)
            o_lo, o_hi = bootstrap_ci(opt)
            stat, p = paired_wilcoxon(voting, opt)
            ratio = v_mean / o_mean if o_mean > 0 else 0
            print(f"    {metric:40s}: voting={v_mean:8.3f} [{v_lo:.3f}, {v_hi:.3f}]  "
                  f"opt={o_mean:8.3f} [{o_lo:.3f}, {o_hi:.3f}]  ratio={ratio:.3f}  p={p:.4f}" if not np.isnan(p) else
                  f"    {metric:40s}: voting={v_mean:8.3f}  opt={o_mean:8.3f}  ratio={ratio:.3f}")
            tables.append({
                "Scenario": scenario, "Metric": metric,
                "Voting_Mean": v_mean, "Voting_CI_lo": v_lo, "Voting_CI_hi": v_hi,
                "Opt_Mean": o_mean, "Opt_CI_lo": o_lo, "Opt_CI_hi": o_hi,
                "Ratio": ratio, "Wilcoxon_p": p,
            })

    table = pd.DataFrame(tables)
    table.to_csv(os.path.join(out_dir, "claim1_quality.csv"), index=False)
    return table


# ---------------------------------------------------------------------------
# Claim 2: Voting is more fair
# ---------------------------------------------------------------------------
def claim2_fairness(df, out_dir):
    print("\n" + "=" * 70)
    print("CLAIM 2: Voting is more fair than optimization")
    print("=" * 70)

    metrics = ["Init_Jains_Index", "Avg Reassignment Jains Index", "Avg_Reass_Gini",
               "Workload_Variance", "Attempted_Workload_Variance"]
    tables = []

    for scenario in SCENARIO_ORDER:
        sdf = df[df["Scenario"] == scenario]
        if sdf.empty:
            continue
        print(f"\n  Scenario: {scenario}")
        for metric in metrics:
            voting = sdf[sdf["Is_Voting"]][metric].dropna().values
            opt = sdf[sdf["Is_Optimization"]][metric].dropna().values
            v_mean = np.mean(voting) if len(voting) else 0
            o_mean = np.mean(opt) if len(opt) else 0
            stat, p = paired_wilcoxon(voting, opt)
            # For fairness metrics: Jain's higher=better, Gini/Variance lower=better
            better = "voting" if ("Jains" in metric and v_mean > o_mean) or \
                                  ("Gini" in metric and v_mean < o_mean) or \
                                  ("Variance" in metric and v_mean < o_mean) else "opt"
            print(f"    {metric:40s}: voting={v_mean:8.4f}  opt={o_mean:8.4f}  better={better}  p={p:.4f}" if not np.isnan(p) else
                  f"    {metric:40s}: voting={v_mean:8.4f}  opt={o_mean:8.4f}  better={better}")
            tables.append({
                "Scenario": scenario, "Metric": metric,
                "Voting_Mean": v_mean, "Opt_Mean": o_mean,
                "Better": better, "Wilcoxon_p": p,
            })

    table = pd.DataFrame(tables)
    table.to_csv(os.path.join(out_dir, "claim2_fairness.csv"), index=False)

    # Pareto front: score vs Jain's index
    fig, ax = plt.subplots(figsize=(8, 6))
    for method_label in METHOD_LABELS.values():
        mdf = df[df["Method_Label"] == method_label]
        if mdf.empty:
            continue
        x = mdf["Avg Reassignment Jains Index"].mean()
        y = mdf["Cumulative_Reassignment_Quality"].mean()
        color = "blue" if method_label in [METHOD_LABELS[m] for m in VOTING_METHODS] else "red"
        marker = "o" if color == "blue" else "s"
        ax.scatter(x, y, label=method_label, c=color, marker=marker, s=60, zorder=3)
    ax.set_xlabel("Avg Reassignment Jains Index (higher = more fair)")
    ax.set_ylabel("Cumulative Reassignment Quality (higher = better)")
    ax.set_title("Quality vs Fairness Pareto Front")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "claim2_pareto.png"), dpi=150)
    plt.close(fig)

    return table


# ---------------------------------------------------------------------------
# Claim 3: Voting adapts faster under dynamic churn
# ---------------------------------------------------------------------------
def claim3_adaptation(df, out_dir):
    print("\n" + "=" * 70)
    print("CLAIM 3: Voting adapts faster under dynamic churn")
    print("=" * 70)

    metrics = ["Adaptation_Latency_Mean", "Adaptation_Latency_Max",
               "Throughput_Var", "total_reassignment_time", "Latency_Per_Event"]
    tables = []

    for scenario in SCENARIO_ORDER:
        sdf = df[df["Scenario"] == scenario]
        if sdf.empty:
            continue
        print(f"\n  Scenario: {scenario}")
        for metric in metrics:
            voting = sdf[sdf["Is_Voting"]][metric].dropna().values
            opt = sdf[sdf["Is_Optimization"]][metric].dropna().values
            v_mean = np.mean(voting) if len(voting) else 0
            o_mean = np.mean(opt) if len(opt) else 0
            stat, p = paired_wilcoxon(voting, opt)
            # For latency/variance/time: lower = better
            better = "voting" if v_mean < o_mean else "opt"
            print(f"    {metric:40s}: voting={v_mean:10.4f}  opt={o_mean:10.4f}  better={better}  p={p:.4f}" if not np.isnan(p) else
                  f"    {metric:40s}: voting={v_mean:10.4f}  opt={o_mean:10.4f}  better={better}")
            tables.append({
                "Scenario": scenario, "Metric": metric,
                "Voting_Mean": v_mean, "Opt_Mean": o_mean,
                "Better": better, "Wilcoxon_p": p,
            })

    table = pd.DataFrame(tables)
    table.to_csv(os.path.join(out_dir, "claim3_adaptation.csv"), index=False)

    # Bar chart: adaptation latency by scenario (voting vs optimization)
    fig, ax = plt.subplots(figsize=(10, 6))
    scenarios = [s for s in SCENARIO_ORDER if s in df["Scenario"].unique()]
    x = np.arange(len(scenarios))
    width = 0.35
    v_vals = [df[(df["Scenario"] == s) & df["Is_Voting"]]["Adaptation_Latency_Mean"].mean() for s in scenarios]
    o_vals = [df[(df["Scenario"] == s) & df["Is_Optimization"]]["Adaptation_Latency_Mean"].mean() for s in scenarios]
    ax.bar(x - width/2, v_vals, width, label="Voting (mean)", color="steelblue")
    ax.bar(x + width/2, o_vals, width, label="Optimization (mean)", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Adaptation Latency Mean (timesteps)")
    ax.set_title("Adaptation Latency by Scenario: Voting vs Optimization")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "claim3_latency_by_scenario.png"), dpi=150)
    plt.close(fig)

    return table


# ---------------------------------------------------------------------------
# Claim 4: Voting scales better with problem size
# ---------------------------------------------------------------------------
def claim4_scaling(df, out_dir):
    print("\n" + "=" * 70)
    print("CLAIM 4: Voting scales better with problem size")
    print("=" * 70)

    metrics = ["total_reassignment_time", "Execution Time", "Planning_Time_us"]
    tables = []

    for metric in metrics:
        print(f"\n  Metric: {metric}")
        for nr in sorted(df["Num Robots"].unique()):
            sdf = df[df["Num Robots"] == nr]
            voting = sdf[sdf["Is_Voting"]][metric].dropna().values
            opt = sdf[sdf["Is_Optimization"]][metric].dropna().values
            v_mean = np.mean(voting) if len(voting) else 0
            o_mean = np.mean(opt) if len(opt) else 0
            ratio = v_mean / o_mean if o_mean > 0 else 0
            print(f"    {nr:4d} robots: voting={v_mean:12.1f}  opt={o_mean:12.1f}  ratio={ratio:.3f}")
            tables.append({
                "Metric": metric, "Num_Robots": nr,
                "Voting_Mean": v_mean, "Opt_Mean": o_mean, "Ratio": ratio,
            })

    table = pd.DataFrame(tables)
    table.to_csv(os.path.join(out_dir, "claim4_scaling.csv"), index=False)

    # Scaling curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for method_label in ["Borda", "Approval", "Range", "CBBA", "SSIA", "ILP", "JV"]:
            mdf = df[df["Method_Label"] == method_label]
            if mdf.empty:
                continue
            grp = mdf.groupby("Num Robots")[metric].mean()
            color = "blue" if method_label in [METHOD_LABELS[m] for m in VOTING_METHODS] else "red"
            ax.plot(grp.index, grp.values, marker="o", label=method_label, color=color, markersize=4)
        ax.set_xlabel("Num Robots")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Scaling: Execution Time vs Fleet Size (blue=voting, red=optimization)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "claim4_scaling.png"), dpi=150)
    plt.close(fig)

    return table


# ---------------------------------------------------------------------------
# Claim 6: Regimes expose where voting wins vs loses
# ---------------------------------------------------------------------------
def claim6_regimes(df, out_dir):
    print("\n" + "=" * 70)
    print("CLAIM 6: Named regimes expose conditional advantages")
    print("=" * 70)

    metrics = ["Task_Completion_Fraction", "Workload_Variance",
               "Throughput_Var", "Attempted_Completion_Rate", "Pool_Size_Trend"]
    tables = []

    # Headline table: Method x Scenario
    print("\n  Headline table (mean across all maps/sizes/reps):")
    print(f"  {'Method':<25} ", end="")
    for s in SCENARIO_ORDER:
        print(f"{'  '+s:>18}", end="")
    print()

    for method in ALL_METHODS:
        label = METHOD_LABELS.get(method, method)
        mdf = df[df["Method"] == method]
        print(f"  {label:<25} ", end="")
        for scenario in SCENARIO_ORDER:
            smdf = mdf[mdf["Scenario"] == scenario]
            val = smdf["Task_Completion_Fraction"].mean() if not smdf.empty else 0
            print(f"{val:>18.4f}", end="")
        print()

    # Detailed per-metric breakdown
    for scenario in SCENARIO_ORDER:
        sdf = df[df["Scenario"] == scenario]
        if sdf.empty:
            continue
        print(f"\n  Scenario: {scenario}")
        for metric in metrics:
            voting = sdf[sdf["Is_Voting"]][metric].dropna().values
            opt = sdf[sdf["Is_Optimization"]][metric].dropna().values
            v_mean = np.mean(voting) if len(voting) else 0
            o_mean = np.mean(opt) if len(opt) else 0
            stat, p = paired_wilcoxon(voting, opt)
            print(f"    {metric:35s}: voting={v_mean:10.4f}  opt={o_mean:10.4f}  p={p:.4f}" if not np.isnan(p) else
                  f"    {metric:35s}: voting={v_mean:10.4f}  opt={o_mean:10.4f}")
            tables.append({
                "Scenario": scenario, "Metric": metric,
                "Voting_Mean": v_mean, "Opt_Mean": o_mean, "Wilcoxon_p": p,
            })

    table = pd.DataFrame(tables)
    table.to_csv(os.path.join(out_dir, "claim6_regimes.csv"), index=False)

    # Pool_Size_Trend by scenario (regime validation)
    fig, ax = plt.subplots(figsize=(8, 5))
    scenarios = [s for s in SCENARIO_ORDER if s in df["Scenario"].unique()]
    trends = [df[df["Scenario"] == s]["Pool_Size_Trend"].mean() for s in scenarios]
    ax.bar(scenarios, trends, color=["green", "orange", "blue", "red"][:len(scenarios)])
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_ylabel("Pool_Size_Trend (mean)")
    ax.set_title("Regime Validation: Pool Size Trend by Scenario\n(near 0 = steady state, positive = overload)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "claim6_regime_validation.png"), dpi=150)
    plt.close(fig)

    # Completion by method x scenario heatmap
    pivot = df.pivot_table(
        values="Task_Completion_Fraction",
        index="Method_Label",
        columns="Scenario",
        aggfunc="mean",
    )
    # Reorder columns
    pivot = pivot.reindex(columns=[s for s in SCENARIO_ORDER if s in pivot.columns])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.values, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Task Completion Fraction: Method x Scenario")
    fig.colorbar(im, ax=ax, label="Completion Fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "claim6_completion_heatmap.png"), dpi=150)
    plt.close(fig)

    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VBTA Evaluation Script")
    parser.add_argument("--dataset", default="maps_balanced2",
                        choices=list(DATASETS.keys()),
                        help="Which dataset to evaluate")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: <dataset>_eval)")
    args = parser.parse_args()

    out_dir = args.output or f"{args.dataset}_eval"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nVBTA Evaluation")
    print(f"Dataset: {args.dataset}")
    print(f"Output:  {out_dir}")
    print(f"\nLoading data...")

    df = load_dataset(args.dataset)

    # Data summary
    print(f"\nData summary:")
    print(f"  Rows: {len(df)}")
    print(f"  Maps: {df['Map'].nunique()} ({sorted(df['Map'].unique())})")
    print(f"  Methods: {df['Method'].nunique()}")
    print(f"  Scenarios: {sorted(df['Scenario'].unique())}")
    print(f"  Suitability methods: {sorted(df['Suitability Method'].unique())}")
    print(f"  Robot sizes: {sorted(df['Num Robots'].unique())}")
    print(f"  Task sizes: {sorted(df['Num Tasks'].unique())}")
    print(f"  Reps: {df['Run ID'].nunique()}")

    # Run claims
    t1 = claim1_quality(df, out_dir)
    t2 = claim2_fairness(df, out_dir)
    t3 = claim3_adaptation(df, out_dir)
    t4 = claim4_scaling(df, out_dir)
    t6 = claim6_regimes(df, out_dir)

    print(f"\n\nDone. Results saved to {out_dir}/")
    print(f"  claim1_quality.csv          — voting vs optimization quality")
    print(f"  claim2_fairness.csv         — fairness comparison + Pareto front PNG")
    print(f"  claim3_adaptation.csv       — adaptation latency + bar chart PNG")
    print(f"  claim4_scaling.csv          — scaling curves PNG")
    print(f"  claim6_regimes.csv          — regime x method + heatmap + validation PNG")


if __name__ == "__main__":
    main()
