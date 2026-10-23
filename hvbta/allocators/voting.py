import time, numpy as np
from typing import List, Tuple, Callable
from .assignments import generate_random_assignments, generate_high_suitability_assignments
from hvbta.models import CapabilityProfile, TaskDescription
from hvbta.suitability import calculate_total_suitability, calculate_suitability_matrix, apply_priority_weights
from .misc_assignment import extract_submatrix


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

def rank_assignments_range(assignments: List[List[Tuple[int, int]]], suitability_matrix: List[List[float]]) -> Tuple[List[float], List[int]]:
    """
    Ranks the given assignments based on their total suitability scores.
    
    Parameters:
        assignments: A list of assignments, where each assignment is a list of (robot, task) pairs.
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability of robot i for task j.
    
    Returns:
        (total_scores, ranking): A tuple where
      1. A list of total suitability scores for each assignment.
      2. A list of indices representing the ranking of the assignments, where the first index corresponds to the
         assignment with the highest total score.
    """
    total_scores = []

    # Calculate total suitability scores for each assignment
    for assignment in assignments:
        total_suitability = sum(suitability_matrix[robot][task] for robot, task in assignment[0])
        total_scores.append(total_suitability)

    # Get the ranking of assignments based on the total scores (higher score is better)
    ranking = sorted(range(len(total_scores)), key=lambda i: total_scores[i], reverse=True)
    #***print (f"ranking: {ranking}")
    return total_scores, ranking

def rank_assignments_borda(assignments: List[List[Tuple[int, int]]], suitability_matrix: List[List[float]]) -> Tuple[List[float], List[int]]:
    """
    Ranks each assignment from the perspective of each robot based on its suitability for its task in each assignment,
    and uses the Borda count method to aggregate these rankings.
    
    Parameters:
        assignments: A list of assignments, where each assignment is a list of (robot, task) pairs.
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability of robot i for task j.
    
    Returns:
        (borda_scores, ranked_assignments): A tuple containing:
      1. A list of Borda scores for each assignment.
      2. A list of indices representing the ranking of the assignments based on Borda scores, where the first index corresponds
         to the assignment with the highest total Borda score.
    """
    num_assignments = len(assignments)
    num_robots = len(suitability_matrix)

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
                robot_scores.append((assignment_index, suitability_matrix[robot][assigned_task]))

        # Sort robot's scores for all assignments: unassigned (score -1) at the bottom, then by suitability
        robot_scores.sort(key=lambda x: x[1], reverse=True)

        # Assign Borda points based on the ranking
        for rank, (assignment_index, _) in enumerate(robot_scores):
            borda_points = len(robot_scores) - rank - 1  # Borda count: higher ranks get more points
            borda_scores[assignment_index] += borda_points

    # Rank assignments based on total Borda scores (higher score is better)
    ranked_assignments = sorted(range(num_assignments), key=lambda i: borda_scores[i], reverse=True)

    return borda_scores, ranked_assignments

def rank_assignments_approval(assignments: List[List[Tuple[int, int]]], suitability_matrix: List[List[float]], threshold: float = 0.5) -> Tuple[List[int], List[int]]:
    """
    Uses approval voting where each robot gives 1 point to assignments where its suitability for its task is above a threshold.
    If a robot's suitability is below the threshold or it is unassigned, it gives 0 points for that assignment.
    
    Parameters:
        assignments: A list of assignments, where each assignment is a list of (robot, task) pairs.
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability of robot i for task j.
        threshold: The suitability threshold above which a robot will approve an assignment (default is 10).
    
    Returns:
        (approval_scores, ranked_assignments): A tuple containing:
      1. A list of approval scores for each assignment.
      2. A list of indices representing the ranking of the assignments based on approval scores.
    """
    num_assignments = len(assignments)
    num_robots = len(suitability_matrix)

    # Initialize approval scores for each assignment
    approval_scores = [0] * num_assignments

    # For each robot, evaluate each assignment based on the suitability threshold
    for robot in range(num_robots):
        for assignment_index, assignment in enumerate(assignments):
            # Find the task assigned to this robot in the current assignment
            assigned_task = next((task for r, task in assignment[0] if r == robot), None)
            
            # Check if robot's suitability rating for this task is above the threshold
            if assigned_task  and suitability_matrix[robot][assigned_task] > threshold:
                approval_scores[assignment_index] += 1  # Robot approves this assignment

    # Rank assignments based on approval scores (higher score is better)
    ranked_assignments = sorted(range(num_assignments), key=lambda i: approval_scores[i], reverse=True)

    return approval_scores, ranked_assignments

def rank_assignments_majority_judgment(assignments: List[List[Tuple[int, int]]], suitability_matrix: List[List[float]]) -> Tuple[List[float], List[int]]:
    """
    Uses majority judgment to rank assignments based on robot suitability for assigned tasks.
    
    Parameters:
        assignments: A list of assignments, where each assignment is a list of (robot, task) pairs.
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability of robot i for task j.
    
    Returns:
        (assignment_ratings, ranked_assignments): A tuple containing:
      1. A list of median scores for each assignment based on majority judgment.
      2. A list of indices representing the ranking of assignments based on median scores.
    """
    # Define qualitative rating categories mapped to suitability score ranges
    rating_scale = {"Excellent": 3, "Good": 2, "Fair": 1, "Poor": 0}
    rating_thresholds = [(.85, "Excellent"), (.7, "Good"), (.5, "Fair"), (0, "Poor")]

    assignment_ratings = []

    for assignment in assignments:
        ratings = []
        
        for robot, task in assignment[0]:
            # Get the suitability score for the robot-task pair
            suitability_score = suitability_matrix[robot][task]

            # Convert suitability score to qualitative rating
            for threshold, rating in rating_thresholds:
                if suitability_score >= threshold:
                    ratings.append(rating_scale[rating])
                    break
        
        # Calculate the median rating for this assignment
        median_rating = np.median(ratings) if ratings else 0
        assignment_ratings.append(median_rating)

    # Rank assignments based on median ratings
    ranked_assignments = sorted(range(len(assignment_ratings)), key=lambda i: assignment_ratings[i], reverse=True)

    return assignment_ratings, ranked_assignments

def rank_assignments_cumulative_voting(assignments: List[List[Tuple[int, int]]], suitability_matrix: List[List[float]], total_votes: int = 10) -> Tuple[List[float], List[int]]:
    """
    Uses cumulative voting to rank assignments, where each robot distributes a fixed number of votes
    across assignments based on suitability scores.
    
    Parameters:
        assignments: A list of assignments, where each assignment is a list of (robot, task) pairs.
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability of robot i for task j.
        total_votes: The total number of votes each robot has to distribute (default is 10).
    
    Returns:
        (cumulative_votes, ranked_assignments): A tuple containing:
      1. A list of cumulative votes for each assignment.
      2. A list of indices representing the ranking of assignments based on cumulative votes.
    """
    num_assignments = len(assignments)
    cumulative_votes = [0] * num_assignments
    if total_votes < num_assignments/2:
        total_votes = num_assignments/2
    
    for robot in range(len(suitability_matrix)):
        # Collect suitability scores for this robot across all assignments
        robot_votes = []
        
        for assignment_index, assignment in enumerate(assignments):
            # Find the task assigned to this robot in the current assignment
            assigned_task = next((task for r, task in assignment[0] if r == robot), None)
            
            # If robot is unassigned in this assignment, give it a suitability score of 0
            suitability_score = suitability_matrix[robot][assigned_task] if assigned_task is not None else 0
            robot_votes.append((assignment_index, suitability_score))
        
        # Sort robot votes based on suitability scores in descending order
        robot_votes.sort(key=lambda x: x[1], reverse=True)

        # Distribute the total votes among assignments proportional to suitability scores
        total_score = sum(score for _, score in robot_votes if score > 0)
        for assignment_index, score in robot_votes:
            if total_score > 0:
                cumulative_votes[assignment_index] += total_votes * (score / total_score)

    # Rank assignments based on cumulative votes (higher score is better)
    ranked_assignments = sorted(range(num_assignments), key=lambda i: cumulative_votes[i], reverse=True)

    return cumulative_votes, ranked_assignments

def rank_assignments_condorcet_method(assignments: List[List[Tuple[int, int]]], suitability_matrix: List[List[float]]) -> Tuple[List[int], List[int]]:
    """
    Uses the Condorcet method to rank assignments by comparing each assignment in a pairwise manner.
    
    Parameters:
        assignments: A list of assignments, where each assignment is a list of (robot, task) pairs.
        suitability_matrix: A 2D list where the element at [i][j] represents the suitability of robot i for task j.
    
    Returns:
        (pairwise_wins, ranked_assignments): A tuple containing:
      1. A list of pairwise win counts for each assignment.
      2. A list of indices representing the ranking of assignments based on pairwise win counts.
    """
    num_assignments = len(assignments)
    num_robots = len(suitability_matrix)

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
                suitability_i = suitability_matrix[robot][task_i] if task_i is not None else 0
                suitability_j = suitability_matrix[robot][task_j] if task_j is not None else 0

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

    # Rank assignments based on pairwise win counts
    ranked_assignments = sorted(range(num_assignments), key=lambda k: pairwise_wins[k], reverse=True)

    return pairwise_wins, ranked_assignments

def assign_tasks_with_voting(
        robots: List[CapabilityProfile],
        tasks: List[TaskDescription],
        suitability_matrix: np.ndarray,
        num_candidates: int,
        voting_method: Callable,
        score_threshold: float = None,
        time_budget_ms: float = None,
        generator: Callable = None,
        seed_assignment: List[Tuple[int, int]] = None,
        diversity_threshold: float = 0.8,
        extra_candidates: List = None,
        top_k_out: List = None,
        top_k: int = 3,
        ) -> Tuple[Tuple[List[Tuple[int, int]], List[int], List[int]], float, float, List[float]]:
    """
    Assigns tasks to robots using random assignment and ranks the assignments using the specified voting method.

    Parameters:
        robots: List of robot profiles.
        tasks: List of task descriptions.
        suitability_matrix: A 2D numpy array with suitability scores for each robot-task pair.
        num_candidates: Upper bound on candidate assignments to generate.
        voting_method: The name of the voting function.
        score_threshold: If not None, a float in [0, 1]. Stop generating
            candidates once one has mean-per-pair suitability >= this value.
            Because every suitability path clamps to [0, 1], threshold=0.7
            means "candidate averages 0.7 suitability per pair" and is
            tunable across runs regardless of num_pairs.
        time_budget_ms: If not None, stop generating candidates once wall-clock
            elapsed (from function entry) reaches this many milliseconds.
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
    # Every candidate (seed, hall-of-fame, generated) is filtered through
    # _filter_zero_pairs FIRST: zero-score (infeasible) pairs are dropped
    # before ranking, with the dropped robots/tasks moved to the candidate's
    # unassigned lists. Candidates that are empty after filtering are not
    # added. This re-scores the warm-start / hall-of-fame entries against
    # the CURRENT matrix (no stored scores).
    full_pool = []
    if seed_assignment:
        seed_pairs = [(int(r), int(t)) for r, t in seed_assignment]
        seed_robot_ids = {r for r, _ in seed_pairs}
        seed_task_ids = {t for _, t in seed_pairs}
        seed_unr = [r for r in range(num_robots) if r not in seed_robot_ids]
        seed_unt = [t for t in range(num_tasks) if t not in seed_task_ids]
        seed_cand = _filter_zero_pairs((seed_pairs, seed_unr, seed_unt), suitability_matrix)
        if seed_cand is not None:
            full_pool.append(seed_cand)

    # Hall-of-fame extras (already validated/translated by the caller). Add as
    # many as fit; they earned their place so they bypass the diversity filter,
    # but they DO seed accepted_pair_sets so generated candidates avoid them.
    # Zero-score pairs are dropped by the same filter.
    if extra_candidates:
        for cand in extra_candidates:
            if len(full_pool) >= num_candidates:
                break
            fixed = _filter_zero_pairs(cand, suitability_matrix)
            if fixed is None:
                continue
            full_pool.append(fixed)

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
            fixed = _filter_zero_pairs(candidate, suitability_matrix)
            if fixed is None:
                continue
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
    # with what we have. If ZERO-PAIR FILTERING emptied the entire pool
    # (every candidate's pairs were all infeasible), assign NOTHING this
    # round - same behavior as Murty's on an all-infeasible matrix. Voting
    # must never execute a zero-score pair.
    if not full_pool:
        length = (time.perf_counter_ns() - start) / 1000.0
        empty = ([], list(range(num_robots)), list(range(num_tasks)))
        print(f"Best assignment in voting {empty} (pool empty after zero-pair filter)")
        return empty, 0.0, length, []

    # Anytime accumulation over the pre-generated pool. When neither budget is
    # set the loop consumes the whole pool identically to the old flow.
    # NOTE: the score_threshold check averages over FILTERED pairs only
    # (zero pairs are already dropped), so with the same threshold the
    # anytime termination now fires EARLIER/more often than before the
    # zero-pair filter.
    random_assignments = []
    for one in full_pool:
        random_assignments.append(one)

        if score_threshold is not None:
            # Cheap per-candidate score = mean-per-pair suitability across
            # every FILTERED pair in the candidate. Because per-pair
            # suitability is clamped to [0, 1] at every scorer, this mean is
            # in [0, 1] too - a threshold of 0.7 means "candidate averages
            # 0.7 suitability per pair" and is tunable across runs.
            pairs = one[0]
            if pairs:
                candidate_score = sum(suitability_matrix[r][t] for r, t in pairs) / len(pairs)
            else:
                candidate_score = 0.0
            if candidate_score >= score_threshold:
                break
        if time_budget_ms is not None:
            if (time.perf_counter_ns() - start) / 1e6 >= time_budget_ms:
                break

    # Priority-weighted view for voting ranking ONLY. Everything else in this
    # function (anytime score_threshold above, best_score reporting) stays
    # on the raw matrix to preserve [0, 1] semantics and cross-run
    # comparability of reported scores. The weighted matrix may have entries
    # up to 2.0 (urgent factor), which is fine for internal ranking but would
    # inflate reported metrics.
    priority_weighted_matrix = apply_priority_weights(suitability_matrix, tasks)

    # Every candidate was zero-pair-filtered before entering the pool, so no
    # candidate contains a zero-score pair and no post-hoc zero handling is
    # needed: the executed candidate is the voting rule's actual winner,
    # assignment_ranking[0].
    total_scores, assignment_ranking = voting_method(random_assignments, priority_weighted_matrix)
    end = time.perf_counter_ns()
    length = (end - start) / 1000.0

    # Hall of fame: hand the top-`top_k` ranked candidates back to the caller.
    if top_k_out is not None:
        for rank_idx in assignment_ranking[:top_k]:
            top_k_out.append(random_assignments[rank_idx])

    best_assignment = random_assignments[assignment_ranking[0]]
    assigned_pairs = list(best_assignment[0])
    unassigned_robots = list(best_assignment[1] or [])
    unassigned_tasks = list(best_assignment[2] or [])

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
        generator: Callable = None,
        seed_assignment_ids: List[Tuple[str, str]] = None,
        diversity_threshold: float = 0.8,
        cached_candidate_ids: List = None,
        top_k_out_ids: List = None,
        hall_of_fame_k: int = 3):
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
            generator=generator, seed_assignment=seed_local,
            diversity_threshold=diversity_threshold,
            extra_candidates=extra_local, top_k_out=local_top_k, top_k=hall_of_fame_k)

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

