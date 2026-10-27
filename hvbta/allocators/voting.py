import random
import time, numpy as np
from typing import List, Tuple, Callable
from .assignments import generate_random_assignments, generate_high_suitability_assignments
from hvbta.models import CapabilityProfile, TaskDescription
from hvbta.suitability import (
    calculate_total_suitability, check_zero_suitability, calculate_suitability_matrix,
    apply_priority_weights, priority_factors,
)
from .misc_assignment import extract_submatrix

# Config flag (default True). When False, assign_tasks_with_voting behaves
# EXACTLY like the pre-fix voting.py: no zero-pair filtering of candidates,
# the check_zero_suitability skip loop and best_ranking reset, and the
# "keep the random assignment" all-zero else-branch. This lets pre-fix and
# post-fix voting run from the same code with the same seeds (ablation).
VOTING_ZERO_PAIR_FILTER = True

# Config flag (default True). When False, the anytime score_threshold uses
# the old mean-per-pair semantics (which the zero-pair filter broke: the
# filtered mean crosses the threshold on the first candidate, so voting
# ranked pools of size 1 under urgency). When True (2.8):
#   - the threshold score is coverage-normalized quality: candidate total
#     raw score divided by max_feasible = min(#robots with >=1 positive
#     task, #tasks with >=1 positive robot) in the current matrix;
#   - max_feasible == 0 means the threshold never fires;
#   - the threshold may only fire after MIN_POOL_BEFORE_THRESHOLD
#     candidates have accumulated (or the whole pool if it is smaller).
VOTING_COVERAGE_THRESHOLD = True
MIN_POOL_BEFORE_THRESHOLD = 5

# Optional pool-diagnostics collector: set to a list to record one dict per
# assign_tasks_with_voting call (pool size before the anytime loop, pool
# size at ranking time, and the active threshold/budget). Used to measure
# how often the anytime early termination fires. Inert by default.
POOL_STATS = None

# Budget-calibration collector (2026-10-25, deterministic-budget change):
# when set to a list, one dict per anytime-budgeted call records how many
# candidates were ranked before the wall-clock budget cut the loop. Used
# once, serially, to calibrate the candidate-count replacement; inert after.
BUDGET_CALIBRATION = None

# Deterministic urgent budget (2026-10-25): the old 20 ms wall-clock budget
# made urgent-call results depend on machine load. Calibrated replacement:
# cap the ranked pool at this many candidates on urgent calls. The value
# is the MEDIAN ranked-candidate count measured over 20,303 urgent calls
# in a serial 1-rep run of the v9 pilot config (lak102d [10,20]x[10,20],
# 6 voting rules, 4 scenarios, 5 mask conditions): median 3 (p25 1, p75 8,
# p95 17, mean 5.6). See docs/checkins/2026-10-25-deterministic-budgets.md.
# 4a (2026-10-26): OFF in the main configuration - under urgency voting
# stops via the good-enough threshold (with the MIN_POOL_BEFORE_THRESHOLD
# floor) or pool exhaustion. Re-enabled only for the cap-ablation arm
# (HVBTA_URGENT_CAP=1).
VOTING_URGENT_CANDIDATE_BUDGET = 3

# 4d (2026-10-26): the urgent good-enough threshold as a fraction of the
# call's upper bound UB = min(UB_robot, UB_task) (see _upper_bounds): stop
# when a candidate's coverage-normalized quality >= f * UB. f is either a
# single float or a callable (num_robots, num_tasks) -> float. None (until
# the user sets it, via HVBTA_UB_F) keeps the legacy fixed threshold
# (0.7) that the main loop passes via score_threshold.
# 2026-10-27: UNUSED BY THE MAIN CONFIGURATION - the anytime stopping rule
# under urgency is now the stall rule below. The machinery stays in place
# for potential ablations; it is inert unless a caller passes
# score_threshold (the main loop no longer does).
VOTING_UB_THRESHOLD_F = None

# Anytime stall rule (2026-10-27; replaces the f*UB threshold in the main
# configuration): under urgency, voting ranks candidates in pool order and
# stops once the best candidate so far has not improved for k consecutive
# candidates - but only after at least MIN_POOL_BEFORE_THRESHOLD candidates
# have been ranked - or when the pool is exhausted. k = 5, equal to the
# min-pool floor (no new fitted constant; a tiered f was rejected as too
# many fitted constants with thin support in large calls - the calibration
# showed median winner/UB = 1.0 in small calls but 0.4-0.6 in large
# balanced calls, where per-bin sample sizes are thin). Non-urgent calls
# keep the full pool. HVBTA_STALL=0 disables the rule (the anytime
# ablation arm: full pool on urgent calls).
VOTING_URGENT_STALL_K = 5

# Optional archive for per-call pool statistics: when set to a list,
# main_simulation appends every run's POOL_STATS records to it just
# before clearing them at run end (diagnostic drivers only; inert by
# default).
POOL_STATS_SINK = None

# 4c calibration collector (2026-10-26): when set to a list, one dict per
# assign_tasks_with_voting call records the call's dimensions, the upper
# bounds, the winner's coverage-normalized quality, and the positive pair
# scores the allocator saw (the call's actual view). Used by the
# calibration driver only; sweeps never enable it.
UB_CALIBRATION = None


def _upper_bounds(suitability_matrix):
    """Per-call upper bounds on achievable total score (4b, 2026-10-26),
    computed from the scores the allocator sees.

    max_feasible = min(#robots with >=1 positive task, #tasks with >=1
    positive robot) - the maximum number of simultaneously-assignable
    positive pairs. UB_robot = mean of the top max_feasible values among
    each robot's best feasible score; UB_task likewise over tasks. Each is
    an upper bound on the mean per-pair score any assignment could achieve
    at that coverage; a total-score interpretation is max_feasible * UB.

    Returns (ub_robot, ub_task, max_feasible); (None, None, 0) when
    nothing is feasible.
    """
    _m = np.asarray(suitability_matrix)
    pos_rows = int(np.sum(np.any(_m > 0, axis=1)))
    pos_cols = int(np.sum(np.any(_m > 0, axis=0)))
    max_feasible = min(pos_rows, pos_cols)
    if max_feasible <= 0:
        return None, None, 0
    row_best = _m.max(axis=1)  # each robot's best feasible score (0 rows sort last)
    col_best = _m.max(axis=0)  # each task's best feasible score
    ub_robot = float(np.mean(np.sort(row_best)[-max_feasible:]))
    ub_task = float(np.mean(np.sort(col_best)[-max_feasible:]))
    return ub_robot, ub_task, max_feasible

# Per-rule top-2 tie counter: {rule_name: [calls, top2_ties]}. Accumulated
# across calls; reset/read by the caller (2.5 reporting). Inert by default.
TIE_STATS = {}


def _tie_break_ranking(scores, rng):
    """
    Indices sorted by score, descending. Ties are broken by the provided RNG
    (shuffle-then-stable-sort) so pool order (seed / hall-of-fame first) no
    longer systematically wins ties. rng=None keeps deterministic index order
    (the pre-2.5 behavior) for direct/legacy calls.
    """
    order = list(range(len(scores)))
    if rng is not None:
        rng.shuffle(order)
    order.sort(key=lambda i: scores[i], reverse=True)
    return order


def _weighted_median(grade_weight_pairs):
    """Priority-weighted median: smallest grade whose cumulative weight
    reaches half the total weight (2.2 MJ)."""
    if not grade_weight_pairs:
        return 0
    pairs = sorted(grade_weight_pairs, key=lambda x: x[0])
    total = sum(w for _, w in pairs)
    cum = 0.0
    for v, w in pairs:
        cum += w
        if cum >= total / 2.0:
            return v
    return pairs[-1][0]


def _filter_zero_pairs(candidate, suitability_matrix):
    """
    Drop every pair whose RAW (unweighted) suitability is 0 from a candidate
    assignment. Dropped robots/tasks go to the candidate's unassigned lists.

    Returns the fixed (pairs, unassigned_robots, unassigned_tasks) candidate,
    or None when no pair survives (candidate is empty after filtering).

    Applied to EVERY candidate before it enters the pool - generated,
    warm-start seed, and hall-of-fame alike - so that:
      - voting rules never rank (and never execute) a zero-score pair,
        matching Murty's (a zero pair is an infeasible pairing that would
        just travel and mis-fire at arrival);
      - the executed candidate is always the voting rule's actual winner
        (assignment_ranking[0]), not the first zero-free candidate;
      - coverage bias (Borda ranking zero pairs above unassigned robots,
        MJ ignoring unassigned voters) can no longer elect all-zero
        candidates.
    """
    pairs, unr, unt = candidate
    kept_pairs = []
    dropped_robots = set()
    dropped_tasks = set()
    for r, t in pairs:
        if suitability_matrix[r][t] > 0:
            kept_pairs.append((int(r), int(t)))
        else:
            dropped_robots.add(int(r))
            dropped_tasks.add(int(t))
    if not kept_pairs:
        return None
    if not dropped_robots:
        return (kept_pairs, list(unr or []), list(unt or []))
    new_unr = sorted(set(unr or []) | dropped_robots)
    new_unt = sorted(set(unt or []) | dropped_tasks)
    return (kept_pairs, new_unr, new_unt)

def suitability_all_zero(suitability_matrix):
    return all(value == 0 for row in suitability_matrix for value in row)

def rank_assignments_range(assignments: List[List[Tuple[int, int]]], weighted_matrix: List[List[float]],
                           raw_matrix=None, priority_factors=None, rng=None) -> Tuple[List[float], List[int]]:
    """
    Ranks the given assignments based on their total (priority-weighted)
    suitability scores. raw_matrix / priority_factors accepted for a uniform
    rule signature (Range is cardinal: it only uses the weighted matrix).

    Returns:
        (total_scores, ranking)
    """
    total_scores = []

    # Calculate total suitability scores for each assignment
    for assignment in assignments:
        total_suitability = sum(weighted_matrix[robot][task] for robot, task in assignment[0])
        total_scores.append(total_suitability)

    ranking = _tie_break_ranking(total_scores, rng)
    return total_scores, ranking

def rank_assignments_borda(assignments: List[List[Tuple[int, int]]], weighted_matrix: List[List[float]],
                           raw_matrix=None, priority_factors=None, rng=None) -> Tuple[List[float], List[int]]:
    """
    Borda count with FRACTIONAL (average-rank) tie handling (2.4): tied
    candidates share the average of the Borda points their positions span,
    so pool order (seed / hall-of-fame first) can no longer hand tied
    candidates unequal points. Each robot ranks the candidates by its
    (priority-weighted) suitability for its task; unassigned ranks lowest
    (-1 sentinel score).

    Returns:
        (borda_scores, ranked_assignments)
    """
    num_assignments = len(assignments)
    num_robots = len(weighted_matrix)

    # Initialize Borda scores for each assignment
    borda_scores = [0] * num_assignments

    # For each robot, rank the assignments based on its suitability rating
    for robot in range(num_robots):
        # Collect suitability scores for this robot across all assignments
        robot_scores = []

        for assignment_index, assignment in enumerate(assignments):
            # Find the task assigned to this robot in the current assignment
            assigned_task = next((task for r, task in assignment[0] if r == robot), None)

            # If the robot has no task assigned in this assignment, assign it the lowest score
            if assigned_task is None:
                robot_scores.append((assignment_index, -1))  # Using -1 to rank unassigned lower
            else:
                robot_scores.append((assignment_index, weighted_matrix[robot][assigned_task]))

        # Sort robot's scores for all assignments: unassigned (score -1) at the bottom, then by suitability
        robot_scores.sort(key=lambda x: x[1], reverse=True)

        # Fractional Borda points: candidates tied on score share the average
        # of the points their positions span (n-1 points for first place).
        n = len(robot_scores)
        i = 0
        while i < n:
            j = i
            while j + 1 < n and robot_scores[j + 1][1] == robot_scores[i][1]:
                j += 1
            if j > i:
                avg_pts = sum(n - 1 - p for p in range(i, j + 1)) / (j - i + 1)
                for k in range(i, j + 1):
                    borda_scores[robot_scores[k][0]] += avg_pts
            else:
                borda_scores[robot_scores[i][0]] += n - 1 - i
            i = j + 1

    ranked_assignments = _tie_break_ranking(borda_scores, rng)

    return borda_scores, ranked_assignments

def rank_assignments_approval(assignments: List[List[Tuple[int, int]]], weighted_matrix: List[List[float]],
                              raw_matrix=None, priority_factors=None, rng=None, threshold: float = 0.5) -> Tuple[List[float], List[int]]:
    """
    Approval voting (2.1 + 2.2): a robot approves an assignment when its RAW
    suitability for its task exceeds the threshold (the old code compared
    against the priority-weighted matrix, where low-priority tasks could
    never exceed 0.5, and `if assigned_task` dropped task index 0). Each
    approval counts the task's priority factor instead of 1, so priority
    adjusts vote weight rather than gating approval.

    Returns:
        (approval_scores, ranked_assignments)
    """
    num_assignments = len(assignments)
    num_robots = len(weighted_matrix)
    if raw_matrix is None:
        raw_matrix = weighted_matrix

    # Initialize approval scores for each assignment
    approval_scores = [0.0] * num_assignments

    # For each robot, evaluate each assignment based on the raw-suitability threshold
    for robot in range(num_robots):
        for assignment_index, assignment in enumerate(assignments):
            # Find the task assigned to this robot in the current assignment
            assigned_task = next((task for r, task in assignment[0] if r == robot), None)

            # 2.1: `assigned_task is not None` (task index 0 is falsy and used
            # to be dropped). 2.2: compare RAW suitability.
            if assigned_task is not None and raw_matrix[robot][assigned_task] > threshold:
                weight = 1.0
                if priority_factors is not None:
                    weight = float(priority_factors[assigned_task])
                approval_scores[assignment_index] += weight  # approval counts the task's priority factor

    ranked_assignments = _tie_break_ranking(approval_scores, rng)

    return approval_scores, ranked_assignments

def rank_assignments_majority_judgment(assignments: List[List[Tuple[int, int]]], weighted_matrix: List[List[float]],
                                       raw_matrix=None, priority_factors=None, rng=None) -> Tuple[List[float], List[int]]:
    """
    Majority Judgment (2.2 + 2.3):
      - Grades are computed from RAW suitability (the old code graded the
        priority-weighted matrix, capping low-priority tasks at Fair).
      - Every robot with at least one positive-score task in the current
        matrix is a VOTER; a voter that is unassigned in a candidate grades
        it Poor (the old median-over-assigned-pairs ignored unassigned
        voters, so one excellent pair beat ten good ones).
      - The median is priority-weighted: each grade's weight is the priority
        factor of the voter's assigned task (1.0 for unassigned voters).

    Returns:
        (assignment_ratings, ranked_assignments)
    """
    if raw_matrix is None:
        raw_matrix = weighted_matrix

    # Define qualitative rating categories mapped to raw-suitability ranges
    rating_scale = {"Excellent": 3, "Good": 2, "Fair": 1, "Poor": 0}
    rating_thresholds = [(.85, "Excellent"), (.7, "Good"), (.5, "Fair"), (0, "Poor")]

    # 2.3: voters are robots with at least one feasible option; robots with
    # no positive-score task are indifferent (they'd grade everything Poor,
    # which only creates ties).
    voters = [r for r in range(len(raw_matrix))
              if any(raw_matrix[r][t] > 0 for t in range(len(raw_matrix[r])))]

    assignment_ratings = []

    for assignment in assignments:
        assigned_by_robot = {r: t for r, t in assignment[0]}
        grades = []
        for robot in voters:
            task = assigned_by_robot.get(robot)
            if task is None:
                # unassigned voter grades the candidate Poor
                grades.append((rating_scale["Poor"], 1.0))
            else:
                score = raw_matrix[robot][task]
                grade = next(rating_scale[rating]
                             for threshold, rating in rating_thresholds
                             if score >= threshold)
                weight = 1.0
                if priority_factors is not None:
                    weight = float(priority_factors[task])
                grades.append((grade, weight))

        # Priority-weighted median over all voters
        assignment_ratings.append(_weighted_median(grades))

    ranked_assignments = _tie_break_ranking(assignment_ratings, rng)

    return assignment_ratings, ranked_assignments

def rank_assignments_cumulative_voting(assignments: List[List[Tuple[int, int]]], weighted_matrix: List[List[float]],
                                       raw_matrix=None, priority_factors=None, rng=None, total_votes: int = 10) -> Tuple[List[float], List[int]]:
    """
    Cumulative voting (cardinal: uses the priority-weighted matrix; raw /
    factors accepted for the uniform rule signature).
    """
    num_assignments = len(assignments)
    cumulative_votes = [0] * num_assignments
    if total_votes < num_assignments/2:
        total_votes = num_assignments/2

    for robot in range(len(weighted_matrix)):
        # Collect suitability scores for this robot across all assignments
        robot_votes = []

        for assignment_index, assignment in enumerate(assignments):
            # Find the task assigned to this robot in the current assignment
            assigned_task = next((task for r, task in assignment[0] if r == robot), None)

            # If robot is unassigned in this assignment, give it a suitability score of 0
            suitability_score = weighted_matrix[robot][assigned_task] if assigned_task is not None else 0
            robot_votes.append((assignment_index, suitability_score))

        # Sort robot votes based on suitability scores in descending order
        robot_votes.sort(key=lambda x: x[1], reverse=True)

        # Distribute the total votes among assignments proportional to suitability scores
        total_score = sum(score for _, score in robot_votes if score > 0)
        for assignment_index, score in robot_votes:
            if total_score > 0:
                cumulative_votes[assignment_index] += total_votes * (score / total_score)

    ranked_assignments = _tie_break_ranking(cumulative_votes, rng)

    return cumulative_votes, ranked_assignments

def rank_assignments_condorcet_method(assignments: List[List[Tuple[int, int]]], weighted_matrix: List[List[float]],
                                      raw_matrix=None, priority_factors=None, rng=None) -> Tuple[List[int], List[int]]:
    """
    Condorcet method (ordinal: uses the priority-weighted matrix; raw /
    factors accepted for the uniform rule signature).
    """
    num_assignments = len(assignments)
    num_robots = len(weighted_matrix)

    # Initialize a matrix to track pairwise wins for each assignment comparison
    pairwise_wins = [0] * num_assignments

    # Compare each pair of assignments
    for i in range(num_assignments):
        for j in range(i + 1, num_assignments):
            # Count the number of robots that prefer assignment i over assignment j and vice versa
            i_wins, j_wins = 0, 0

            for robot in range(num_robots):
                # Find the task assigned to the robot in both assignments
                task_i = next((task for r, task in assignments[i][0] if r == robot), None)
                task_j = next((task for r, task in assignments[j][0] if r == robot), None)

                # Get suitability scores (or 0 if unassigned)
                suitability_i = weighted_matrix[robot][task_i] if task_i is not None else 0
                suitability_j = weighted_matrix[robot][task_j] if task_j is not None else 0

                # Determine which assignment the robot prefers
                if suitability_i > suitability_j:
                    i_wins += 1
                elif suitability_j > suitability_i:
                    j_wins += 1

            # Update pairwise win counts based on robot preferences
            if i_wins > j_wins:
                pairwise_wins[i] += 1
            elif j_wins > i_wins:
                pairwise_wins[j] += 1

    ranked_assignments = _tie_break_ranking(pairwise_wins, rng)

    return pairwise_wins, ranked_assignments

def assign_tasks_with_voting(
        robots: List[CapabilityProfile],
        tasks: List[TaskDescription],
        suitability_matrix: np.ndarray,
        num_candidates: int,
        voting_method: Callable,
        score_threshold: float = None,
        time_budget_ms: float = None,
        candidate_budget: int = None,
        stall_k: int = None,
        generator: Callable = None,
        seed_assignment: List[Tuple[int, int]] = None,
        diversity_threshold: float = 0.8,
        extra_candidates: List = None,
        top_k_out: List = None,
        top_k: int = 3,
        tie_seed: int = None,
        ) -> Tuple[Tuple[List[Tuple[int, int]], List[int], List[int]], float, float, List[float]]:
    """
    Assigns tasks to robots using random assignment and ranks the assignments using the specified voting method.

    Parameters:
        robots: List of robot profiles.
        tasks: List of task descriptions.
        suitability_matrix: A 2D numpy array with suitability scores for each robot-task pair.
        num_candidates: Upper bound on candidate assignments to generate.
        voting_method: The name of the voting function.
        score_threshold: If not None, a float in [0, 1]. When
            VOTING_COVERAGE_THRESHOLD is on (2.8), stops accumulating the
            pool once a candidate's coverage-normalized quality (total raw
            score / max_feasible) reaches this value, and only after at
            least MIN_POOL_BEFORE_THRESHOLD candidates (or the whole pool).
            When the flag is off, the old mean-per-pair semantics apply.
            2026-10-27: the main loop no longer passes a threshold under
            urgency (the stall rule below replaced it); this parameter is
            kept for ablations of the old semantics.
        time_budget_ms: If not None, stop generating candidates once wall-clock
            elapsed (from function entry) reaches this many milliseconds.
            DEPRECATED for the main loop (2026-10-25): wall-clock budgets
            make results load-dependent; the main loop now passes the
            deterministic candidate_budget instead. Kept for back-compat.
        candidate_budget: If not None, rank at most this many candidates
            from the pool (deterministic anytime budget, calibrated from
            the old 20 ms budget's behavior; see
            VOTING_URGENT_CANDIDATE_BUDGET).
        stall_k: If not None (2026-10-27, the anytime stopping rule for
            urgent calls), rank candidates in pool order and stop once the
            best candidate so far (by coverage-normalized quality) has not
            improved for stall_k consecutive candidates - but only after
            at least MIN_POOL_BEFORE_THRESHOLD candidates have been ranked
            - or when the pool is exhausted. Deterministic; no fitted
            constants beyond k = VOTING_URGENT_STALL_K = 5, equal to the
            min-pool floor.
        generator: Optional candidate generator with signature
            (num_robots, num_tasks, num_assignments) -> List[candidate]. If
            None, the biased generate_high_suitability_assignments is used
            when the matrix has any positive entry, otherwise
            generate_random_assignments. Callers can pass a specific
            generator to override the auto-selection.
        seed_assignment: Optional warm-start seed as list of (local_r_idx,
            local_t_idx). Prepended to the pool as candidate #0.
        extra_candidates: Optional list of pre-built candidates (each a local-
            index 3-tuple (pairs, unassigned_robots, unassigned_tasks)) injected
            into the pool right after the seed - used for the hall-of-fame cache.
            They count against num_candidates and seed the diversity filter, so
            generated candidates explore away from them.
        top_k_out: Optional mutable list. If provided, after ranking it is
            populated with the top-`top_k` candidates (local-index 3-tuples) so
            the caller can cache them.
        top_k: How many top candidates to write into top_k_out (default 3).
        tie_seed: Optional seed (2.5) for the tie-break RNG passed to the
            voting rule, derived by the caller from the schedule seed +
            replan index.
        diversity_threshold: Float in [0, 1] or None. Reject a newly-generated
            candidate if its pair-overlap with any already-accepted candidate
            (|A ∩ B| / |A|) exceeds this value. 0.8 means "must have >= 20% of
            pairs different from every prior candidate". None or >= 1.0
            disables the filter. Retries are capped at 5 * num_candidates so
            small state spaces don't loop forever.
        When budgets are both None, behavior matches the previous full-pool version.

    Returns:
        (best_assignment, best_score, length, per_agent_scores): The best assignment, its suitability score, and the time taken for the voting process.
    """
    num_robots = len(robots)
    num_tasks = len(tasks)

    start = time.perf_counter_ns()

    # Choose a generator. Default to the biased generator when the matrix has
    # any positive score (which is the case when this function is reached at
    # all - suitability_all_zero() is routed elsewhere upstream). Fall back to
    # random when we truly have nothing to bias toward.
    if generator is None:
        matrix_has_positive = np.any(np.asarray(suitability_matrix) > 0)
        if matrix_has_positive:
            def generator(nr, nt, k):
                return generate_high_suitability_assignments(nr, nt, suitability_matrix, k)
        else:
            generator = generate_random_assignments

    # Pre-materialize the pool with three combined concerns:
    #   1. Warm-start seed (if any) sits at index 0.
    #   2. Generator fills the rest via the biased/random generator.
    #   3. Diversity filter rejects candidates whose pair-overlap with any
    #      already-accepted candidate exceeds diversity_threshold (bounded
    #      retry loop; degrades gracefully in tiny state spaces).
    # When VOTING_ZERO_PAIR_FILTER is on, every candidate (seed,
    # hall-of-fame, generated) is filtered through _filter_zero_pairs
    # FIRST: zero-score (infeasible) pairs are dropped before ranking, with
    # the dropped robots/tasks moved to the candidate's unassigned lists.
    # This re-scores the warm-start / hall-of-fame entries against the
    # CURRENT matrix (no stored scores). With the flag off, the pre-fix
    # unfiltered pool construction is used verbatim.
    full_pool = []
    if seed_assignment:
        seed_pairs = [(int(r), int(t)) for r, t in seed_assignment]
        seed_robot_ids = {r for r, _ in seed_pairs}
        seed_task_ids = {t for _, t in seed_pairs}
        seed_unr = [r for r in range(num_robots) if r not in seed_robot_ids]
        seed_unt = [t for t in range(num_tasks) if t not in seed_task_ids]
        seed_cand = (seed_pairs, seed_unr, seed_unt)
        if VOTING_ZERO_PAIR_FILTER:
            seed_cand = _filter_zero_pairs(seed_cand, suitability_matrix)
        if seed_cand is not None:
            full_pool.append(seed_cand)

    # Hall-of-fame extras (already validated/translated by the caller). Add as
    # many as fit; they earned their place so they bypass the diversity filter,
    # but they DO seed accepted_pair_sets so generated candidates avoid them.
    if extra_candidates:
        for cand in extra_candidates:
            if len(full_pool) >= num_candidates:
                break
            if VOTING_ZERO_PAIR_FILTER:
                fixed = _filter_zero_pairs(cand, suitability_matrix)
                if fixed is None:
                    continue
                full_pool.append(fixed)
            else:
                full_pool.append(cand)

    diversity_active = diversity_threshold is not None and diversity_threshold < 1.0
    accepted_pair_sets = [set(c[0]) for c in full_pool]
    tries = 0
    max_tries = max(1, 5 * num_candidates)
    while len(full_pool) < num_candidates and tries < max_tries:
        tries += 1
        need = num_candidates - len(full_pool)
        # Ask the generator for a batch. Smaller batches keep the retry loop
        # responsive; larger batches amortize generator overhead. A batch
        # size of `need` is a reasonable compromise.
        batch = generator(num_robots, num_tasks, need)
        for candidate in batch:
            if len(full_pool) >= num_candidates:
                break
            if VOTING_ZERO_PAIR_FILTER:
                fixed = _filter_zero_pairs(candidate, suitability_matrix)
                if fixed is None:
                    continue
            else:
                fixed = candidate
            if not diversity_active:
                full_pool.append(fixed)
                continue
            # Overlap is computed on the FILTERED pairs (never empty here -
            # _filter_zero_pairs returns None for empty candidates).
            new_pairs = set(fixed[0])
            too_similar = False
            for accepted in accepted_pair_sets:
                if not accepted:
                    continue
                overlap = len(new_pairs & accepted) / len(new_pairs)
                if overlap > diversity_threshold:
                    too_similar = True
                    break
            if not too_similar:
                full_pool.append(fixed)
                accepted_pair_sets.append(new_pairs)

    # If the retry cap was hit before we filled the pool, we still proceed
    # with what we have. When the zero-pair filter is on and it emptied the
    # entire pool (every candidate's pairs were all infeasible), assign
    # NOTHING this round - same behavior as Murty's on an all-infeasible
    # matrix. Voting must never execute a zero-score pair. (With the flag
    # off, the pre-fix code always had >= 1 candidate.)
    if not full_pool:
        length = (time.perf_counter_ns() - start) / 1000.0
        empty = ([], list(range(num_robots)), list(range(num_tasks)))
        print(f"Best assignment in voting {empty} (pool empty after zero-pair filter)")
        return empty, 0.0, length, []

    # Anytime accumulation over the pre-generated pool. When neither budget is
    # set the loop consumes the whole pool identically to the old flow.
    # 2.8 (VOTING_COVERAGE_THRESHOLD): the old mean-per-pair check broke under
    # the zero-pair filter - the filtered mean crosses the threshold on the
    # very first candidate, so voting ranked pools of size 1 under urgency.
    # Coverage-normalized semantics instead: candidate total raw score
    # divided by max_feasible = min(#robots with >=1 positive task,
    # #tasks with >=1 positive robot), and the threshold may only fire after
    # MIN_POOL_BEFORE_THRESHOLD candidates (or the whole pool if smaller).
    # 4b (2026-10-26): the call's upper bounds are computed from the same
    # scores the allocator sees; 4d: when VOTING_UB_THRESHOLD_F is set, the
    # effective threshold is f * UB (f a float or a function of the call's
    # robot/task counts) instead of the fixed score_threshold.
    # Stall rule (2026-10-27, replaces the threshold in the main config):
    # with stall_k set, candidates are ranked in pool order and the loop
    # stops once the best coverage-normalized quality so far has not
    # improved for stall_k consecutive candidates (after the min-pool
    # floor), or when the pool is exhausted.
    ub_robot, ub_task, ub_max_feasible = _upper_bounds(suitability_matrix)
    ub = min(ub_robot, ub_task) if ub_robot is not None else None
    effective_threshold = score_threshold
    if score_threshold is not None and VOTING_UB_THRESHOLD_F is not None:
        _f = VOTING_UB_THRESHOLD_F
        if callable(_f):
            _f = _f(num_robots, num_tasks)
        effective_threshold = (_f * ub) if (ub is not None and ub > 0) else None
    min_pool = min(MIN_POOL_BEFORE_THRESHOLD, len(full_pool))
    if VOTING_COVERAGE_THRESHOLD:
        max_feasible = ub_max_feasible
    random_assignments = []
    budget_fired = False
    threshold_fired = False
    stall_fired = False
    best_quality = -1.0          # best coverage-normalized quality seen
    candidates_since_improvement = 0
    for one in full_pool:
        random_assignments.append(one)

        if stall_k is not None:
            _q = (sum(suitability_matrix[r][t] for r, t in one[0]) / ub_max_feasible
                  if ub_max_feasible > 0 else 0.0)
            if _q > best_quality:
                best_quality = _q
                candidates_since_improvement = 0
            else:
                candidates_since_improvement += 1
            if (len(random_assignments) >= min_pool
                    and candidates_since_improvement >= stall_k):
                stall_fired = True
                break
        if effective_threshold is not None:
            if VOTING_COVERAGE_THRESHOLD:
                if max_feasible > 0 and len(random_assignments) >= min_pool:
                    candidate_score = sum(
                        suitability_matrix[r][t] for r, t in one[0]) / max_feasible
                else:
                    # not enough pool yet, or nothing feasible at all: never fire
                    candidate_score = 0.0
            else:
                # flag off: old mean-per-pair semantics over filtered pairs
                pairs = one[0]
                if pairs:
                    candidate_score = sum(suitability_matrix[r][t] for r, t in pairs) / len(pairs)
                else:
                    candidate_score = 0.0
            if candidate_score >= effective_threshold:
                threshold_fired = True
                break
        if time_budget_ms is not None:
            if (time.perf_counter_ns() - start) / 1e6 >= time_budget_ms:
                budget_fired = True
                break
        if candidate_budget is not None and len(random_assignments) >= candidate_budget:
            budget_fired = True
            break

    # Priority-weighted view for voting ranking ONLY. Everything else in this
    # function (anytime score_threshold above, best_score reporting) stays
    # on the raw matrix to preserve [0, 1] semantics and cross-run
    # comparability of reported scores. The weighted matrix may have entries
    # up to 2.0 (urgent factor), which is fine for internal ranking but would
    # inflate reported metrics.
    priority_weighted_matrix = apply_priority_weights(suitability_matrix, tasks)
    factors = priority_factors(tasks)

    # 2.5: seeded tie-break RNG (schedule seed + replan index, derived by the
    # caller). None keeps deterministic index-order tie-breaks for legacy calls.
    tie_rng = random.Random(tie_seed) if tie_seed is not None else None

    if POOL_STATS is not None:
        POOL_STATS.append({
            "rule": getattr(voting_method, "__name__", str(voting_method)),
            "full": len(full_pool),
            "ranked": len(random_assignments),
            "threshold": effective_threshold,
            "budget_ms": time_budget_ms,
            "candidate_budget": candidate_budget,
            # Stall-rule diagnostics (2026-10-27): n_robots / n_tasks are
            # the call's dimensions = the CURRENT unassigned counts.
            "stall_k": stall_k,
            "stall_fired": stall_fired,
            "budget_fired": budget_fired,
            "n_robots": num_robots,
            "n_tasks": num_tasks,
        })
    if BUDGET_CALIBRATION is not None and time_budget_ms is not None:
        BUDGET_CALIBRATION.append({
            "ranked": len(random_assignments),
            "full": len(full_pool),
            "budget_fired": budget_fired,
            "threshold_fired": threshold_fired,
        })

    # Every candidate was zero-pair-filtered before entering the pool, so no
    # candidate contains a zero-score pair and no post-hoc zero handling is
    # needed: the executed candidate is the voting rule's actual winner,
    # assignment_ranking[0].
    total_scores, assignment_ranking = voting_method(
        random_assignments, priority_weighted_matrix,
        raw_matrix=suitability_matrix, priority_factors=factors, rng=tie_rng)

    # 2.5 reporting: how often the top two candidates tie, per rule.
    if len(assignment_ranking) >= 2:
        rule_name = getattr(voting_method, "__name__", str(voting_method))
        cell = TIE_STATS.setdefault(rule_name, [0, 0])
        cell[0] += 1
        if total_scores[assignment_ranking[0]] == total_scores[assignment_ranking[1]]:
            cell[1] += 1

    # 4c calibration collector: one record per call with the call's
    # dimensions, upper bounds, winner quality, and the positive pair
    # scores the allocator saw. Winner quality uses the same
    # coverage-normalized semantics as the anytime threshold (winner =
    # assignment_ranking[0], the executed candidate under the zero-pair
    # filter - the collector assumes the flag-on main configuration).
    if UB_CALIBRATION is not None:
        _winner = random_assignments[assignment_ranking[0]]
        _winner_quality = (
            sum(suitability_matrix[r][t] for r, t in _winner[0]) / ub_max_feasible
            if ub_max_feasible > 0 else 0.0)
        _m = np.asarray(suitability_matrix)
        UB_CALIBRATION.append({
            "rule": getattr(voting_method, "__name__", str(voting_method)),
            "urgent": (score_threshold is not None) or (stall_k is not None),
            "full": len(full_pool),
            "ranked": len(random_assignments),
            "budget_fired": budget_fired,
            "threshold_fired": threshold_fired,
            "n_robots": num_robots, "n_tasks": num_tasks,
            "max_feasible": ub_max_feasible,
            "ub_robot": ub_robot, "ub_task": ub_task, "ub": ub,
            "winner_quality": float(_winner_quality),
            "pairs": _m[_m > 0].tolist(),
        })
    end = time.perf_counter_ns()
    length = (end - start) / 1000.0

    # Hall of fame: hand the top-`top_k` ranked candidates back to the caller.
    if top_k_out is not None:
        for rank_idx in assignment_ranking[:top_k]:
            top_k_out.append(random_assignments[rank_idx])

    if VOTING_ZERO_PAIR_FILTER:
        # Every candidate was zero-pair-filtered before entering the pool, so
        # no candidate contains a zero-score pair and no post-hoc zero handling
        # is needed: the executed candidate is the voting rule's actual winner,
        # assignment_ranking[0].
        best_assignment = random_assignments[assignment_ranking[0]]
        assigned_pairs = list(best_assignment[0])
        unassigned_robots = list(best_assignment[1] or [])
        unassigned_tasks = list(best_assignment[2] or [])
    else:
        # Pre-fix behavior, verbatim: advance past candidates containing a
        # zero pair (so the executed candidate is NOT necessarily the rule's
        # winner), reset to index 0 when the loop runs out, and fall back to
        # keeping the random assignment when the chosen candidate is all-zero.
        best_ranking = 0
        # check for zero suitability in the best assignment and move to the next best if so
        while(check_zero_suitability(random_assignments[assignment_ranking[best_ranking]][0], suitability_matrix) and best_ranking < len(assignment_ranking)-1):
            best_ranking += 1
        if best_ranking == len(random_assignments) - 1:
            best_ranking = 0

        best_assignment = random_assignments[assignment_ranking[best_ranking]]
        assigned_pairs = list(best_assignment[0])
        unassigned_robots = list(best_assignment[1] or [])
        unassigned_tasks = list(best_assignment[2] or [])

        has_positive = any(suitability_matrix[r][t] > 0 for r, t in assigned_pairs)
        if has_positive:
            kept_pairs = []
            for r, t in assigned_pairs:
                if suitability_matrix[r][t] > 0:
                    kept_pairs.append((r, t))
                else:
                    if r not in unassigned_robots:
                        unassigned_robots.append(r)
                    if t not in unassigned_tasks:
                        unassigned_tasks.append(t)
            assigned_pairs = kept_pairs
        else:
            # All pairs scored 0 – keep the random assignment so the simulator has work to do.
            if best_assignment[1] is None:
                assigned_ids = {r for r, _ in assigned_pairs}
                unassigned_robots = [idx for idx in range(num_robots) if idx not in assigned_ids]
            if best_assignment[2] is None:
                assigned_task_ids = {t for _, t in assigned_pairs}
                unassigned_tasks = [idx for idx in range(num_tasks) if idx not in assigned_task_ids]

    filtered_best_assignments = (
        assigned_pairs,
        unassigned_robots,
        unassigned_tasks,
    )
    print(f"Best assignment in voting {filtered_best_assignments}")

    best_score = calculate_total_suitability(filtered_best_assignments[0], suitability_matrix)

    # Extract per-agent scores for fairness calculation (only assigned agents)
    per_agent_scores = [float(suitability_matrix[r][t]) for r, t in filtered_best_assignments[0]]

    return filtered_best_assignments, best_score, length, per_agent_scores

def assign_tasks_randomly(
        robots: List[CapabilityProfile], 
        tasks: List[TaskDescription], 
        suitability_matrix: np.ndarray, num_candidates: int
    ) -> Tuple[Tuple[List[Tuple[int, int]], List[int], List[int]], float, float, List[float]]:
    """
    Assigns tasks to robots using random assignment and random assignments.
    for use with the all-zero suitability matrix case.
    
    Parameters:
        robots: List of robot profiles.
        tasks: List of task descriptions.
        suitability_matrix: A 2D numpy array with suitability scores for each robot-task pair.
        num_candidates: Number of candidate assignments to generate.
    
    Returns:
        (best_assignment, best_score, length, per_agent_scores): The best assignment, its suitability score, and the time taken for the voting process.
    """
    num_robots = len(robots)
    num_tasks = len(tasks)
    
    random_assignments = generate_random_assignments(num_robots, num_tasks, num_candidates)
    
    start = time.perf_counter_ns()
    # total_scores, assignment_ranking = voting_method(random_assignments, suitability_matrix)
    # assignment ranking is a list of integers usually from 0 to num_candidates-1 that ranks the assignments
    # but here we just pick a random assignment since they are all equally bad
    k = np.random.randint(0, num_candidates)
    final_pairs, final_unr_idx, final_unt_idx = random_assignments[k]
    # for i in range(num_candidates):
    #     pairs, unr_idx, unt_idx = random_assignments[i]
    #     if len(pairs) > len(final_pairs):
    #         final_pairs, final_unr_idx, final_unt_idx = pairs, unr_idx, unt_idx
    #         break
    
    end = time.perf_counter_ns()
    length = (end - start) / 1000.0
    total_scores = 0.0
 
    assigned_pairs = []
    unassigned_robots = []
    unassigned_tasks = []

    print(f"ASSIGNED PAIRS: {final_pairs} \n UNASSIGNED ROBOTS: {final_unr_idx} \n UNASSIGNED TASKS: {final_unt_idx}")

    for robot_id, task_id in final_pairs:
        assigned_pairs.append((robot_id, task_id))

    if final_unr_idx is None or final_unt_idx is None:
        assigned_r = {r for r, _ in assigned_pairs}
        assigned_t = {t for _, t in assigned_pairs}
        unassigned_robots = [i for i in range(num_robots) if i not in assigned_r]
        unassigned_tasks = [j for j in range(num_tasks) if j not in assigned_t]
    else:
        unassigned_robots = list(final_unr_idx)
        unassigned_tasks = list(final_unt_idx)

    filtered_best_assignments = (assigned_pairs, unassigned_robots, unassigned_tasks)

    # print(f"Best assignment in voting {filtered_best_assignments}")

    # best_score = calculate_total_suitability(filtered_best_assignments[0], suitability_matrix)

    # Extract per-agent scores for fairness calculation (only assigned agents)
    per_agent_scores = [float(suitability_matrix[r][t]) for r, t in assigned_pairs]

    return filtered_best_assignments, total_scores, length, per_agent_scores

def reassign_robots_to_tasks(
        robots: List[CapabilityProfile],
        tasks: List[TaskDescription],
        num_candidates: int,
        voting_method: Callable,
        suitability_source,  # Can be Callable (scorer) OR tuple (matrix, r_idx, t_idx) for LLM
        unassigned_robots: List[str],
        unassigned_tasks: List[str],
        start_positions: dict,
        goal_positions: dict,
        map_size: int,
        inertia_threshold: float = 0.1,
        score_threshold: float = None,
        time_budget_ms: float = None,
        candidate_budget: int = None,
        stall_k: int = None,
        generator: Callable = None,
        seed_assignment_ids: List[Tuple[str, str]] = None,
        diversity_threshold: float = 0.8,
        cached_candidate_ids: List = None,
        top_k_out_ids: List = None,
        hall_of_fame_k: int = 3,
        tie_seed: int = None):
    """
    Reassigns unassigned robots to unassigned tasks using a voting method.
    Parameters:
        robots: List of all robot profiles.
        tasks: List of all task descriptions.
        num_candidates: Number of candidate assignments to generate.
        voting_method: The name of the voting function to use for ranking assignments.
        suitability_source: Either a callable scorer(robot, task)->float, OR a tuple of
                           (full_matrix, robot_id_to_idx, task_id_to_idx) for direct matrix lookup.
        unassigned_robots: List of unassigned robot IDs.
        unassigned_tasks: List of unassigned task IDs.
        start_positions: Dictionary mapping robot IDs to their start positions.
        goal_positions: Dictionary mapping robot IDs to their goal positions.
        Inertia threshold: minimum improvement in suitability required to steal an already‐assigned task.

    Returns:
        return_assignments: Dictionary mapping robot IDs to assigned task IDs.
        unassigned_robots: List of unassigned robot IDs after reassignment.
        unassigned_tasks: List of unassigned task IDs after reassignment.
        score: Total suitability score of the best assignment.
        length: Time taken for the voting process in microseconds.
    """
    urobots = [robot for robot in robots if not robot.assigned]
    utasks = [task for task in tasks if not task.assigned]

    # Early outs
    if not urobots or not utasks:
        return {}, unassigned_robots, unassigned_tasks, 0.0, 0.0, []

    # Check if suitability_source is a matrix tuple or a callable scorer
    if isinstance(suitability_source, tuple) and len(suitability_source) == 3:
        # Direct matrix lookup path (fast - for LLM)
        full_matrix, robot_id_to_idx, task_id_to_idx = suitability_source
        suitability_matrix = extract_submatrix(full_matrix, urobots, utasks, robot_id_to_idx, task_id_to_idx)
        use_matrix_lookup = True
    else:
        # Callable scorer path (for non-LLM methods)
        suitability_matrix = suitability_source(urobots, utasks, map_size=map_size)
        suitability_matrix = np.clip(suitability_matrix, 0.0, 1.0)
        use_matrix_lookup = False

    if suitability_all_zero(suitability_matrix):
        print("All suitability scores are zero, randomly assigning tasks to robots.")
        output, score, length, per_agent_scores = reassign_robots_to_tasks_randomly(robots, tasks, num_candidates, unassigned_robots, unassigned_tasks)
    else:
        # ID -> local index maps for the current unassigned pools. Shared by the
        # seed translation and the hall-of-fame cache translation below.
        rid_to_local = {r.robot_id: i for i, r in enumerate(urobots)}
        tid_to_local = {t.task_id: j for j, t in enumerate(utasks)}

        def _translate_id_pairs(id_pairs):
            """ID pairs -> local (r_idx, t_idx) pairs, dropping any pair whose
            robot or task is no longer among the unassigned pool. Enforces one
            robot / one task per candidate (first valid wins)."""
            seen_r, seen_t = set(), set()
            local = []
            for rid, tid in id_pairs:
                if (rid in rid_to_local and tid in tid_to_local
                        and rid not in seen_r and tid not in seen_t):
                    local.append((rid_to_local[rid], tid_to_local[tid]))
                    seen_r.add(rid)
                    seen_t.add(tid)
            return local

        # Warm-start seed (single candidate).
        seed_local = _translate_id_pairs(seed_assignment_ids) if seed_assignment_ids else None
        if not seed_local:
            seed_local = None  # nothing survived, don't wedge an empty seed into the pool

        # Hall-of-fame cache (multiple candidates). Wrap each surviving pair set
        # into the local 3-tuple candidate shape the generators produce.
        extra_local = None
        if cached_candidate_ids:
            extra_local = []
            for cand_ids in cached_candidate_ids:
                local_pairs = _translate_id_pairs(cand_ids)
                if not local_pairs:
                    continue
                used_r = {r for r, _ in local_pairs}
                used_t = {t for _, t in local_pairs}
                unr = [i for i in range(len(urobots)) if i not in used_r]
                unt = [j for j in range(len(utasks)) if j not in used_t]
                extra_local.append((local_pairs, unr, unt))
            if not extra_local:
                extra_local = None

        # Local accumulator for this round's top-K, translated back to IDs after.
        local_top_k = [] if top_k_out_ids is not None else None

        output, score, length, per_agent_scores = assign_tasks_with_voting(
            urobots, utasks, suitability_matrix, num_candidates, voting_method,
            score_threshold=score_threshold, time_budget_ms=time_budget_ms,
            candidate_budget=candidate_budget,
            stall_k=stall_k,
            generator=generator, seed_assignment=seed_local,
            diversity_threshold=diversity_threshold,
            extra_candidates=extra_local, top_k_out=local_top_k, top_k=hall_of_fame_k,
            tie_seed=tie_seed)

        # Translate the top-K candidates back to ID pairs for the caller's cache.
        if top_k_out_ids is not None and local_top_k:
            for cand in local_top_k:
                id_pairs = [(urobots[i].robot_id, utasks[j].task_id) for i, j in cand[0]]
                if id_pairs:
                    top_k_out_ids.append(id_pairs)

    # this assigned_pairs only contains the unassigned robots and tasks, I may have to pass in the actual assigned_pairs to update it
    assigned_pairs = output[0] # list of tuples
    return_assignments = {}
    unassigned_robots = [urobots[i].robot_id for i in output[1]]
    unassigned_tasks = [utasks[j].task_id for j in output[2]]

    for (robot_idx, task_idx) in assigned_pairs:
        # print(f"UROBOT: {urobots[robot_idx].robot_id} | UTASK: {utasks[task_idx].task_id}")
        pair_score = suitability_matrix[robot_idx][task_idx]
        r = urobots[robot_idx]
        t = utasks[task_idx]
        r.current_task = t
        r.tasks_attempted += 1
        t.assigned_robot = r
        r.assigned = True
        t.assigned = True
        r.current_task_suitability = pair_score
        t.current_suitability = pair_score
        # update start and goal positions when robots are assigned
        start_positions[r.robot_id] = r.location
        goal_positions[r.robot_id] = t.location
        return_assignments[r.robot_id] = t.task_id
        # return_assignments.update({urobots[robot_idx].robot_id : utasks[task_idx].task_id})

    # Check for stealing tasks from already assigned robots
    free_robots = [r for r in robots if not r.assigned]
    if free_robots:
        for task in tasks:
            current = task.assigned_robot
            if current is None:
                continue

            # Use direct matrix lookup if available (LLM path), otherwise use scorer
            if use_matrix_lookup:
                current_suitability = full_matrix[robot_id_to_idx[current.robot_id], task_id_to_idx[task.task_id]]
            else:
                current_suitability = float(suitability_source([current], [task], map_size=map_size)[0, 0])
            # print(f"Better suitability in reassigning: {current_suitability}")
            # find the best free robot for this task
            best, best_suit = None, current_suitability
            for r in free_robots:
                if use_matrix_lookup:
                    s = full_matrix[robot_id_to_idx[r.robot_id], task_id_to_idx[task.task_id]]
                else:
                    s = float(suitability_source([r], [task], map_size=map_size)[0, 0])
                if s > best_suit:
                    # print(f"Better suitability in reassigning: {s}")
                    best, best_suit = r, s
                    
            # Priority-weighted inertia check: URGENT tasks halve the required delta,
            # LOW tasks double it, so a robot can be pulled off a low-priority job for
            # an urgent one even when the raw suitability gain is modest.
            _priority_factor = {"low": 0.5, "medium": 1.0, "high": 1.5, "urgent": 2.0}.get(
                getattr(task, "priority_level", "medium"), 1.0
            )
            effective_threshold = inertia_threshold / _priority_factor
            if best and (best_suit - current_suitability) >= effective_threshold:
                # unassign the current robot from the task
                current.current_task = None
                current.assigned = False
                current.current_task_suitability = None
                if current.robot_id not in unassigned_robots:
                    unassigned_robots.append(current.robot_id)

                # update the task's assigned robot
                best.current_task = task
                best.assigned = True
                best.tasks_attempted += 1
                best.current_task_suitability = best_suit
                task.assigned_robot = best
                task.current_suitability = best_suit

                start_positions[best.robot_id] = best.location
                goal_positions[best.robot_id] = task.location

                # remove from free list and unassigned robots
                free_robots.remove(best)
                if best.robot_id in unassigned_robots:
                    unassigned_robots.remove(best.robot_id)
    
    print(f"Reassign Score: {score}, Reassign Length: {length}")

    # Recalculate per_agent_scores after potential stealing - get scores for all assigned robots
    final_per_agent_scores = []
    for r in robots:
        if r.assigned and r.current_task is not None:
            final_per_agent_scores.append(float(r.current_task_suitability))

    return return_assignments, unassigned_robots, unassigned_tasks, score, length, final_per_agent_scores

def reassign_robots_to_tasks_randomly(
        robots: List[CapabilityProfile],
        tasks: List[TaskDescription],
        num_candidates: int,
        unassigned_robots: List[str],
        unassigned_tasks: List[str],

) ->Tuple[Tuple[List[Tuple[int, int]], List[int], List[int]], float, float]:
    """
    Reassigns unassigned robots to unassigned tasks using random assignments.
    for use with the all-zero suitability matrix case.
    
    Parameters:
        robots: List of all robot profiles.
        tasks: List of all task descriptions.
        num_candidates: Number of candidate assignments to generate.
        unassigned_robots: List of unassigned robot IDs.
        unassigned_tasks: List of unassigned task IDs.
    
    Returns:
        return_assignments: Dictionary mapping robot IDs to assigned task IDs.
        unassigned_robots: List of unassigned robot IDs after reassignment.
        unassigned_tasks: List of unassigned task IDs after reassignment.
        score: Total suitability score of the best assignment (always 0.0).
        length: Time taken for the random assignment process in microseconds.
    """
    urobots = [robot for robot in robots if not robot.assigned]
    utasks = [task for task in tasks if not task.assigned]
    num_robots = len(urobots)
    num_tasks = len(utasks)

    random_assignments = generate_random_assignments(num_robots, num_tasks, num_candidates)
    
    start = time.perf_counter_ns()
    # total_scores, assignment_ranking = voting_method(random_assignments, suitability_matrix)
    # assignment ranking is a list of integers usually from 0 to num_candidates-1 that ranks the assignments
    # but here we just pick a random assignment since they are all equally bad
    k = np.random.randint(0, num_candidates)
    final_pairs, final_unr_idx, final_unt_idx = random_assignments[k]
    # for i in range(num_candidates):
    #     pairs, unr_idx, unt_idx = random_assignments[i]
    #     if len(pairs) > len(final_pairs):
    #         final_pairs, final_unr_idx, final_unt_idx = pairs, unr_idx, unt_idx
    #         break
    
    end = time.perf_counter_ns()
    length = (end - start) / 1000.0
    total_scores = 0.0
 
    assigned_pairs = []
    unassigned_robots = []
    unassigned_tasks = []

    print(f"ASSIGNED PAIRS: {final_pairs} \n UNASSIGNED ROBOTS: {final_unr_idx} \n UNASSIGNED TASKS: {final_unt_idx}")

    for robot_id, task_id in final_pairs:
        assigned_pairs.append((robot_id, task_id))

    if final_unr_idx is None or final_unt_idx is None:
        assigned_r = {r for r, _ in assigned_pairs}
        assigned_t = {t for _, t in assigned_pairs}
        unassigned_robots = [i for i in range(num_robots) if i not in assigned_r]
        unassigned_tasks = [j for j in range(num_tasks) if j not in assigned_t]
    else:
        unassigned_robots = list(final_unr_idx)
        unassigned_tasks = list(final_unt_idx)

    filtered_best_assignments = (assigned_pairs, unassigned_robots, unassigned_tasks)

    # print(f"Best assignment in voting {filtered_best_assignments}")

    # best_score = calculate_total_suitability(filtered_best_assignments[0], suitability_matrix)

    # Per-agent scores are all 0.0 for random assignment (all-zero suitability case)
    per_agent_scores = [0.0] * len(assigned_pairs)

    return filtered_best_assignments, total_scores, length, per_agent_scores

