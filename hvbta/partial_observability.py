"""
Partial-observability experiment support.

Implements the experiment spec in `partial_observability_experiment.md`:
the simulator keeps each task's true attributes while the allocator
(suitability scoring, voting, optimization, Murty's) sees a masked view
where some fields are hidden. Masks are a pure function of
(schedule_seed, task_id, mask_count) so every method and both missing-
attribute policies in a run group see identical masks.

Missing-attribute policies (§6.2 of the spec):
  optimistic  - numeric requirements read as 0, list requirements read as
               empty (nothing required), duration reads its distribution
               minimum.
  pessimistic - numeric requirements read as the generation-distribution
               maximum, list requirements read as unmatchable (the
               requirement is treated as unmet), duration reads its
               distribution maximum.

Non-requirement fields (task_type, difficulty) use neutral defaults
(population mode / median of the strict task profiles) under BOTH policies.
"""
import copy
import json
import os
import random
import zlib
from collections import Counter

import numpy as np

from .models import CapabilityProfile, TaskDescription
from .profiles import STRICT_TASK_PROFILES
from .suitability import balanced_hard_fail, _balanced_pair
from .allocators.misc_assignment import unassign_task_from_robot

# ---------------------------------------------------------------------------
# Maskable fields (11) - see spec §2. `duration` maps to the TaskDescription
# attribute `time_to_complete`. `performance_metric` is NOT maskable (not
# read by the balanced scorer). `reward`, `nl_description`, `priority_level`
# are never masked.
# ---------------------------------------------------------------------------
MASKABLE_FIELDS = (
    "task_type",
    "difficulty",
    "navigation_constraints",
    "required_capabilities.payload",
    "required_capabilities.reach",
    "environmental_conditions",
    "sensors_needed",
    "manipulators_needed",
    "communication_requirements",
    "safety_protocols",
    "duration",
)

# Fixed mask counts per level (spec §4): low = ceil(11 * 0.15) = 2,
# high = ceil(11 * 0.38) = 5.
MASK_COUNTS = {"none": 0, "low": 2, "high": 5}

PO_CONDITIONS = (
    ("none", "none"),
    ("low", "optimistic"),
    ("low", "pessimistic"),
    ("high", "optimistic"),
    ("high", "pessimistic"),
)


def mask_for(schedule_seed: int, task_id: str, k: int) -> frozenset:
    """
    Pure function computing a task's mask (spec §5).

    Builds a dedicated random.Random from zlib.adler32 over
    repr((schedule_seed, task_id)) - stable across processes - shuffles
    the maskable field list and takes the first k. Because the same RNG
    stream is used for every k, a task's `low` mask is a subset of its
    `high` mask, and the mask never depends on the policy.
    """
    if k <= 0:
        return frozenset()
    rng = random.Random(zlib.adler32(repr((schedule_seed, task_id)).encode()))
    fields = list(MASKABLE_FIELDS)
    rng.shuffle(fields)
    return frozenset(fields[:k])


# ---------------------------------------------------------------------------
# Policy substitution values, computed once from the task-generation
# distribution (STRICT_TASK_PROFILES). These are deterministic and
# run-independent so every method in a group scores identical views.
# ---------------------------------------------------------------------------
def _task_type_mode() -> str:
    """Population mode of task_type; deterministic first-seen tie-break."""
    counts = Counter(p["task_type"] for p in STRICT_TASK_PROFILES)
    best = max(counts.values())
    for p in STRICT_TASK_PROFILES:
        if counts[p["task_type"]] == best:
            return p["task_type"]


TASK_TYPE_MODE = _task_type_mode()  # neutral default for masked task_type
DIFFICULTY_MEDIAN = float(np.median([p["difficulty"] for p in STRICT_TASK_PROFILES]))
PAYLOAD_MAX = float(max(p["required_capabilities"]["payload"] for p in STRICT_TASK_PROFILES))
REACH_MAX = float(max(p["required_capabilities"]["reach"] for p in STRICT_TASK_PROFILES))
DURATION_MIN = float(min(p["duration"] for p in STRICT_TASK_PROFILES))
DURATION_MAX = float(max(p["duration"] for p in STRICT_TASK_PROFILES))

# Sentinel for pessimistic list requirements: matches nothing any robot has,
# so the requirement is scored as unmet (and triggers balanced's manipulator
# hard-fail when manipulators_needed is masked).
UNMET = "__MASKED__"

_LIST_FIELDS = (
    "navigation_constraints",
    "environmental_conditions",
    "sensors_needed",
    "manipulators_needed",
    "communication_requirements",
    "safety_protocols",
)


def build_masked_view(task: TaskDescription, masked_fields: frozenset, policy: str):
    """
    Return a copy of `task` in which every field in `masked_fields` holds
    the policy's substitute value (spec §6.1/§6.2). The true task object is
    never mutated and true values of masked fields are unreachable from the
    copy. Fields not in the mask keep their true values (shared references
    for lists, which are never mutated in place).
    """
    view = copy.copy(task)  # shallow: replaces masked attrs wholesale
    optimistic = (policy == "optimistic")

    if "task_type" in masked_fields:
        # Neutral default (population mode). A raw sentinel can never reach
        # the scorer: task_type is a dict key in balanced's special-functions
        # mapping and an unknown value would raise KeyError.
        view.task_type = TASK_TYPE_MODE
    if "difficulty" in masked_fields:
        view.difficulty = DIFFICULTY_MEDIAN

    rc = dict(task.required_capabilities or {})
    if "required_capabilities.payload" in masked_fields:
        rc["payload"] = 0.0 if optimistic else PAYLOAD_MAX
    if "required_capabilities.reach" in masked_fields:
        rc["reach"] = 0.0 if optimistic else REACH_MAX
    view.required_capabilities = rc

    for field in _LIST_FIELDS:
        if field in masked_fields:
            setattr(view, field, [] if optimistic else [UNMET])

    if "duration" in masked_fields:
        # duration feeds the battery hard-fail: optimistic assumes the
        # shortest possible task, pessimistic the longest.
        view.time_to_complete = DURATION_MIN if optimistic else DURATION_MAX

    return view


class PartialObservabilityState:
    """
    Per-run state for the partial-observability experiment.

    Owns: the active mask table (task_id -> masked field set, shrunk on
    reveal), the reveal/mis-assignment/nav-drop counters, per-robot travel
    accounting, and the per-round true-vs-masked suitability accumulators.

    The simulator keeps the true task objects; the allocator only ever
    scores through `masked_scorer(base_scorer)`, which swaps in masked views
    at the suitability-source boundary (all allocator decisions flow through
    that call - Phase 0 finding).
    """

    def __init__(self, schedule_seed: int, mask_level: str, mask_policy: str):
        if mask_level not in MASK_COUNTS:
            raise ValueError(f"Unknown mask_level {mask_level!r}; expected one of {list(MASK_COUNTS)}")
        if mask_level == "none":
            mask_policy = "none"
        self.schedule_seed = schedule_seed
        self.mask_level = mask_level
        self.mask_policy = mask_policy
        self.mask_count = MASK_COUNTS[mask_level]

        # mask tables
        self._masked = {}        # task_id -> frozenset (active; removed on reveal)
        self._revealed = set()   # task_ids whose mask was permanently removed
        self._mask_history = {}  # task_id -> frozenset (masks ever applied; survives reveal)

        # outcome counters
        self.arrivals = 0
        self.mis_assignment_count = 0
        self.wasted_travel_steps = 0
        self.nav_drop_count = 0
        self.nav_drop_wasted_travel = 0
        self.reveal_events = 0
        self.total_travel_steps = 0
        self.all_zero_skip_ticks = 0  # ticks where allocation was skipped (all-zero submatrix)
        self.field_mis_assignments = {}  # masked field name -> mis-assignments involving it
        self.masked_wait = {}            # task_id -> ticks spent unassigned while masked
        self.recovery_latencies = []     # ticks from mis-assignment to next reassignment

        # per-robot episode tracking
        self._travel_ticks = {}    # robot_id -> moves since current assignment
        self._arrival_task = {}     # robot_id -> task_id of last processed arrival
        self._arrival_travel = {}  # robot_id -> travel ticks at arrival (frozen)
        self._last_task = {}       # robot_id -> task_id at last end_of_tick (change detection)

        # per-round true/masked suitability accumulators
        self._rounds = 0
        self._true_round_sum = 0.0
        self._masked_round_sum = 0.0

        # task_id -> tick of an unresolved mis-assignment (recovery latency)
        self._pending_recover = {}

    # ------------------------------------------------------------------
    # Masks
    # ------------------------------------------------------------------
    def mask_of(self, task_id: str) -> frozenset:
        """Active mask for a task (computed lazily; empty once revealed)."""
        if task_id in self._revealed:
            return frozenset()
        m = self._masked.get(task_id)
        if m is None:
            m = mask_for(self.schedule_seed, task_id, self.mask_count)
            self._masked[task_id] = m
            if m:
                self._mask_history[task_id] = m
        return m

    def is_masked(self, task_id: str) -> bool:
        return bool(self.mask_of(task_id))

    def _pop_mask(self, task_id: str) -> frozenset:
        """Remove and return the task's mask. Reveal is permanent (once)."""
        m = self.mask_of(task_id)  # computes lazily in case it was never scored
        if not m:
            return frozenset()
        del self._masked[task_id]
        self._revealed.add(task_id)
        self.reveal_events += 1
        return m

    # ------------------------------------------------------------------
    # Masked view / scorer wrapper
    # ------------------------------------------------------------------
    def masked_view(self, task: TaskDescription):
        mf = self.mask_of(task.task_id)
        if not mf:
            return task
        return build_masked_view(task, mf, self.mask_policy)

    def masked_scorer(self, base_scorer):
        """
        Wrap a batch scorer so it only ever sees masked views of the tasks.
        Signature matches what the allocators call:
            scorer(robots, tasks, *, map_size) -> matrix
        Also covers the single-pair stealing calls ([r], [task]).
        """
        def scorer(robots, tasks, *, map_size=None):
            views = [self.masked_view(t) for t in tasks]
            return base_scorer(robots, views, map_size=map_size)
        return scorer

    # ------------------------------------------------------------------
    # Timestep hooks (all default no-ops when PO is inactive)
    # ------------------------------------------------------------------
    def on_move(self, robot: CapabilityProfile):
        """One grid move toward the current task."""
        self._travel_ticks[robot.robot_id] = self._travel_ticks.get(robot.robot_id, 0) + 1
        self.total_travel_steps += 1

    def on_unassign(self, robot: CapabilityProfile):
        """Robot freed from its task; next arrival must be re-processed."""
        self._arrival_task[robot.robot_id] = None

    def handle_arrival(self, robot: CapabilityProfile, task: TaskDescription,
                       unassigned_robots, unassigned_tasks,
                       current_tick=None, pending_event_steps=None) -> bool:
        """
        Called from the timestep when a robot reaches its task's location.
        Reveals the task's mask (exactly once) and runs balanced's hard-fail
        checks against the TRUE attributes (spec §6.3/§6.4).

        Returns True when the pairing is a mis-assignment: the robot is
        freed at its current location, the task returns to the pool (now
        unmasked), and a replan event is raised through pending_event_steps.
        Returns False when the robot should proceed with the task as normal.
        """
        rid = robot.robot_id
        if self._arrival_task.get(rid) == task.task_id:
            # already processed this (robot, task) episode
            return False
        self._arrival_task[rid] = task.task_id
        self.arrivals += 1
        self._arrival_travel[rid] = self._travel_ticks.get(rid, 0)

        masked_fields = self._pop_mask(task.task_id)

        if balanced_hard_fail(robot, task, 0.0):
            self.mis_assignment_count += 1
            self.wasted_travel_steps += self._arrival_travel[rid]
            for f in masked_fields:
                self.field_mis_assignments[f] = self.field_mis_assignments.get(f, 0) + 1
            if current_tick is not None:
                self._pending_recover[task.task_id] = current_tick
            unassign_task_from_robot(
                robot, task,
                unassigned_robots=unassigned_robots,
                unassigned_tasks=unassigned_tasks,
            )
            if pending_event_steps is not None and current_tick is not None:
                pending_event_steps.append(current_tick)
            return True

        return False

    def record_nav_drop(self, robot: CapabilityProfile):
        """Existing navigation runtime drop (spec §6.5): count, don't change."""
        self.nav_drop_count += 1
        self.nav_drop_wasted_travel += self._arrival_travel.get(robot.robot_id, 0)

    # ------------------------------------------------------------------
    # Per-tick / per-round bookkeeping (called from main_simulation)
    # ------------------------------------------------------------------
    def note_round(self, robots, map_size):
        """
        After an allocation round (initial assignment or a replan), record
        the mean true-attribute and masked-view suitability of the currently
        assigned pairs. True scores come from the real balanced scorer on
        the true tasks; masked scores are the pair scores the allocator
        itself recorded at assignment time.
        """
        true_scores = []
        masked_scores = []
        for r in robots:
            if r.assigned and r.current_task is not None:
                true_scores.append(_balanced_pair(r, r.current_task, map_size))
                ms = r.current_task_suitability
                masked_scores.append(float(ms) if ms is not None else 0.0)
        if true_scores:
            self._rounds += 1
            self._true_round_sum += float(np.mean(true_scores))
            self._masked_round_sum += float(np.mean(masked_scores))

    def end_of_tick(self, robots, tasks, tasks_ever_assigned, tick):
        """
        End-of-timestep bookkeeping:
          - detect (robot, task) pairing changes since the last tick and
            reset that robot's travel counter / arrival flag (this is where
            recovery latency for previously mis-assigned tasks is closed
            out),
          - accumulate masked-wait ticks for tasks that sit in the pool
            while masked.
        """
        for r in robots:
            rid = r.robot_id
            cur = r.current_task.task_id if (r.assigned and r.current_task is not None) else None
            if cur is not None and cur != self._last_task.get(rid):
                self._travel_ticks[rid] = 0
                self._arrival_task[rid] = None
                if cur in self._pending_recover:
                    self.recovery_latencies.append(tick - self._pending_recover.pop(cur))
            self._last_task[rid] = cur

        for t in tasks:
            if not t.assigned and self.is_masked(t.task_id):
                self.masked_wait[t.task_id] = self.masked_wait.get(t.task_id, 0) + 1

    # ------------------------------------------------------------------
    # Final metrics + diagnostics
    # ------------------------------------------------------------------
    def finalize(self, tasks, tasks_ever_assigned) -> dict:
        """Compute the PO outcome metrics (spec §7). -1 marks N/A."""
        starvation = sum(
            1 for t in tasks
            if t.task_id not in tasks_ever_assigned and self.is_masked(t.task_id)
        )
        ever_masked = set(self._mask_history) | {tid for tid, m in self._masked.items() if m}
        if ever_masked:
            mean_wait = float(np.mean([self.masked_wait.get(tid, 0) for tid in ever_masked]))
        else:
            mean_wait = -1.0
        return {
            "mis_assignment_count": self.mis_assignment_count,
            "mis_assignment_rate": (self.mis_assignment_count / self.arrivals) if self.arrivals else -1.0,
            "wasted_travel_steps": self.wasted_travel_steps,
            "wasted_travel_fraction": (self.wasted_travel_steps / self.total_travel_steps) if self.total_travel_steps else -1.0,
            "masked_starvation_count": starvation,
            "mean_masked_wait": mean_wait,
            "reveal_events": self.reveal_events,
            "recovery_latency_mean": float(np.mean(self.recovery_latencies)) if self.recovery_latencies else -1.0,
            "true_assignment_suitability": (self._true_round_sum / self._rounds) if self._rounds else -1.0,
            "masked_view_suitability": (self._masked_round_sum / self._rounds) if self._rounds else -1.0,
            "nav_drop_count": self.nav_drop_count,
            "nav_drop_wasted_travel": self.nav_drop_wasted_travel,
            "all_zero_skip_ticks": self.all_zero_skip_ticks,
        }

    def diagnostics(self) -> dict:
        masks = {tid: sorted(m) for tid, m in self._mask_history.items()}
        for tid, m in self._masked.items():
            if m:
                masks.setdefault(tid, sorted(m))
        return {
            "schedule_seed": self.schedule_seed,
            "mask_level": self.mask_level,
            "mask_policy": self.mask_policy,
            "mask_count": self.mask_count,
            "neutral_defaults": {
                "task_type_mode": TASK_TYPE_MODE,
                "difficulty_median": DIFFICULTY_MEDIAN,
            },
            "policy_values": {
                "payload_max": PAYLOAD_MAX,
                "reach_max": REACH_MAX,
                "duration_min": DURATION_MIN,
                "duration_max": DURATION_MAX,
                "unmet_sentinel": UNMET,
            },
            "task_masks": masks,
            "per_field_mis_assignments": dict(sorted(self.field_mis_assignments.items())),
            "arrivals": self.arrivals,
            "reveal_events": self.reveal_events,
            "mis_assignment_count": self.mis_assignment_count,
            "nav_drop_count": self.nav_drop_count,
            "all_zero_skip_ticks": self.all_zero_skip_ticks,
            "recovery_latencies": list(self.recovery_latencies),
        }

    def dump_diagnostics(self, path: str, **extra):
        data = self.diagnostics()
        data.update(extra)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
