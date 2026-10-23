# Analysis Plan

Written and committed before running the sweeps it covers. Changes go in a
dated Amendments section at the end; the original text is never edited.

## Data used

Only the v2 sweeps count toward these tests: the noise experiment,
hyper-constrained, the PO sweep, constrained, maps_balanced2, and Full-Sweep,
each re-run into a new `<name>_v2_<timestamp>/` directory. Pilot runs are for
validation only.

## Methods compared

Six voting rules (Borda, Approval, Majority Judgment, Cumulative, Condorcet,
Range), Murty's K-best (built on JV), and perturbed JV.

Perturbed JV's noise level σ_pjv is chosen by this rule, fixed now: on the
pilot map, try σ_pjv ∈ {0.02, 0.05, 0.1, 0.2} and use the smallest value whose
duplicate-retry rate is below 10%. Completion results are not looked at
during selection. The chosen value and the duplicate-retry rates are added in
an Amendment before any sweep counted here is run.

Ordinal/median rules: Borda, Majority Judgment, Condorcet.
Cardinal methods: Range, Murty's, perturbed JV.

## Hypotheses and predictions

H1, noise robustness (primary). As observation noise σ_obs increases,
completion degrades faster for cardinal methods than for ordinal/median
rules. Prediction: the drop in Task_Completion_Fraction from σ_obs = 0 to
σ_obs = 0.3 is larger for Murty's than for each of Borda, Majority Judgment,
and Condorcet.

H2, recovery. Voting has fewer plan failures and lower adaptation latency
than Murty's, with completion no worse than Murty's.

H3, aggregation vs. randomness. Voting's recovery advantage over Murty's also
holds against perturbed JV. If perturbed JV matches voting, the advantage
comes from randomness, not aggregation.

H-PO1. Voting's advantage over Murty's grows as masking increases.
H-PO2. Optimistic masking raises mis-assignment rate and wasted travel;
pessimistic masking raises starvation above the Never_Attempted baseline.
H-PO3. Voting and perturbed JV have lower Mis_Assignment_Recovery_Latency
than Murty's.

Quality and fairness claims (completion, Jain's index, Workload_Variance) are
re-tested on the v2 data as secondary results.

## Metrics

Primary: Task_Completion_Fraction, Adaptation_Latency_Mean,
Plan_Failure_Count.
Secondary: Mis_Assignment_Rate, Wasted_Travel_Fraction,
Masked_Starvation_Count minus Never_Attempted_Count,
Mis_Assignment_Recovery_Latency, True_Assignment_Suitability, Jain's index,
Workload_Variance, Throughput_Mean.

## Statistical tests

- All method comparisons are paired. A pair is two methods on the same group
  (same map, robot count, task count, rep, scenario, and condition), which
  share robots, tasks, schedule, masks, and noise.
- Test: Wilcoxon signed-rank on per-group differences.
- Effect size: median paired difference with a bootstrap 95% confidence
  interval (10,000 resamples).
- H1 is tested on the per-group degradation (value at σ_obs = 0 minus value
  at σ_obs = 0.3), compared between Murty's and each ordinal rule.
- Multiple comparisons: Holm-Bonferroni within each hypothesis family.
  α = 0.05.
- Results are reported per scenario and pooled.

## Reporting

Every hypothesis above is reported, whether supported or not. No hypothesis
is dropped or replaced after the data is seen. Exploratory analyses are
labeled as exploratory.

## Amendments

(none yet)
