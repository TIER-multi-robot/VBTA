"""
LaCAM planner — calls the compiled C++ lacam0 binary via subprocess.

The C++ lacam0 implementation (https://github.com/kei18/lacam0) is a complete,
highly optimized MAPF solver that includes the real LCA mechanism, PIBT swap,
tree rewiring (LaCAM*), and hindrance heuristics. It solves 400 agents on a
32x32 map in 31ms — orders of magnitude faster than the Python implementation
and actually complete on dense-corridor maps where the Python version fails.

## Integration approach

1. Convert the sim's current map (grid) to a lacam-format .map file (cached —
   the same map is used across all runs, so we write it once).
2. Convert the agents' start/goal positions to a lacam-format .scen file.
3. Call the compiled binary as a subprocess: `main -m mapfile -i scenfile -N
   num_agents -o outputfile -t timeout`.
4. Parse the output file's `solution=` section to extract per-agent paths.
5. Convert (x,y) coordinates back to (row,col) tuples.

The .scen file format (from Moving AI MAPF benchmark):
    0	map_name	width	height	start_col	start_row	goal_col	goal_row	...
(whitespace/tab separated; lacam parses lines matching a regex)

The output format (from make_log in post_processing.cpp):
    agents=N
    ...
    solution=
    0:(x0,y0),(x1,y1),...
    1:(x0,y0),(x1,y1),...
    ...

Where x=col, y=row, matching lacam's internal coordinate system.

`agents` is ALREADY augmented with stationary blockers by _prepare_agents;
this adapter must NOT augment again. Stationary agents (start == goal) are
included in the .scen file — lacam handles them naturally (they just stay
put in the solution).
"""

import os
import subprocess
import tempfile
import time
import re


# Path to the compiled lacam0 binary. Resolved relative to this file.
# This file is at <repo>/hvbta/pathfinding/lacam_planner.py, so the repo
# root is three levels up. The binary is at <repo>/lacam0/build/main[.exe].
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_LACAM_BINARY = os.path.join(_REPO_ROOT, "lacam0", "build", "main.exe")
if not os.path.exists(_LACAM_BINARY):
    _LACAM_BINARY = os.path.join(_REPO_ROOT, "lacam0", "build", "main")

# Cache of written .map files, keyed by map_dict identity (so we only write
# the file once per map per process).
_map_file_cache = {}

# Last failure reason, set on every None return from plan() (pure
# diagnostics for the Phase 3 cascade investigation; not read by any
# decision path). Values: "timeout" (subprocess hit the time limit),
# "solved=0" (engine finished but reported no solution), "no_output_file",
# "no_solution_section", "empty_solution", "exception:<msg>".
LAST_FAILURE_REASON = None


def _ensure_map_file(map_dict):
    """Write the sim's grid to a temp .map file in lacam format (cached).

    Cache key is (rows, cols, frozenset(obstacles)) — NOT id(map_dict) —
    because Python reuses id() values for garbage-collected objects, which
    would cause the cache to return the wrong map file.
    """
    rows, cols = map_dict["dimension"]
    obs_key = frozenset(map_dict["obstacles"])
    cache_key = (rows, cols, obs_key)
    if cache_key in _map_file_cache:
        return _map_file_cache[cache_key]

    rows, cols = map_dict["dimension"]
    obstacles = set(map_dict["obstacles"])

    fd, map_path = tempfile.mkstemp(suffix=".map", prefix="lacam_map_")
    with os.fdopen(fd, "w") as f:
        f.write(f"type octile\n")
        f.write(f"height {rows}\n")
        f.write(f"width {cols}\n")
        f.write("map\n")
        for r in range(rows):
            line = []
            for c in range(cols):
                # lacam uses '.' or 'G' for free, '@' or 'T' for obstacle
                line.append("@" if (r, c) in obstacles else ".")
            f.write("".join(line) + "\n")

    _map_file_cache[cache_key] = map_path
    return map_path


def _write_scen_file(agents, map_file_name):
    """Write a .scen file with the agents' start/goal positions."""
    lines = ["0\t" + os.path.basename(map_file_name) + "\tW\tH\tSC\tSR\tGC\tGR\t0"]
    for a in agents:
        # lacam scen format: col, row for start and goal.
        # Our coordinates are (row, col) = (x, y) where x=row, y=col.
        # lacam uses (x=col, y=row), so we SWAP.
        sr, sc = a["start"]
        gr, gc = a["goal"]
        lines.append(f"0\t{map_file_name}\t0\t0\t{sc}\t{sr}\t{gc}\t{gr}\t0")
    fd, scen_path = tempfile.mkstemp(suffix=".scen", prefix="lacam_scen_")
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines) + "\n")
    return scen_path


def _parse_output(output_path, num_agents):
    """Parse the lacam output file's solution= section into per-agent paths.

    Returns {agent_idx: [(row, col), ...]} or None on failure.
    Agent index corresponds to the order in the .scen file.
    """
    if not os.path.exists(output_path):
        return None

    with open(output_path, "r") as f:
        content = f.read()

    # Check if solved (lacam outputs "solved=1" or "solved=0", not true/false)
    if "solved=0" in content:
        return None

    # Extract solution section
    match = re.search(r"solution=\n(.*)", content, re.DOTALL)
    if not match:
        return None

    solution_text = match.group(1)
    paths = {}
    for line in solution_text.strip().split("\n"):
        if not line.strip():
            continue
        # Format: "t:(x0,y0),(x1,y1),..."
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        t = int(parts[0])
        coords_text = parts[1].strip().rstrip(",")
        if not coords_text:
            continue
        coords = []
        for coord_str in coords_text.split("),("):
            coord_str = coord_str.strip("()")
            x, y = coord_str.split(",")
            # lacam: x=col, y=row. We want (row, col).
            coords.append((int(y), int(x)))

        for agent_idx, cell in enumerate(coords):
            if agent_idx not in paths:
                paths[agent_idx] = []
            paths[agent_idx].append(cell)

    if not paths:
        return None
    return paths


def plan(agents, map_dict, time_limit=3.0):
    """
    agents: list of {"id":rid, "start":(row,col), "goal":(row,col)} (already
            augmented with stationary blockers; do not augment again).
    map_dict: {"dimension":(rows,cols), "obstacles":[(r,c), ...]}
    time_limit: wall-clock seconds for the solver (default 10s).

    Returns {robot_id: [(row,col), ...]} or None on failure.
    """
    if not agents:
        return {}

    global LAST_FAILURE_REASON
    map_path = _ensure_map_file(map_dict)
    scen_path = _write_scen_file(agents, os.path.basename(map_path))

    fd, output_path = tempfile.mkstemp(suffix=".txt", prefix="lacam_out_")
    os.fdopen(fd).close()  # just create the file, we'll overwrite via subprocess

    try:
        cmd = [
            _LACAM_BINARY,
            "-m", map_path,
            "-i", scen_path,
            "-N", str(len(agents)),
            "-o", output_path,
            "-t", str(time_limit),
            "-v", "0",  # quiet
        ]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=time_limit + 5,  # +5s grace for process overhead
        )

        if not os.path.exists(output_path):
            LAST_FAILURE_REASON = "no_output_file"
            return None
        with open(output_path, "r") as f:
            content = f.read()
        if "solved=0" in content:
            # Distinguish budget exhaustion from a genuine no-solution:
            # comp_time at/near the limit = timeout; well under it = the
            # engine exhausted the (finite) search space, i.e. unsolvable.
            import re as _re
            m = _re.search(r"comp_time=(\d+)", content)
            comp_time = int(m.group(1)) if m else -1
            limit_ms = int(time_limit * 1000)
            LAST_FAILURE_REASON = (
                f"solved=0,comp_time={comp_time},limit={limit_ms}" if comp_time >= 0 else "solved=0")
            return None
        if "solution=" not in content:
            LAST_FAILURE_REASON = "no_solution_section"
            return None

        paths = _parse_output(output_path, len(agents))
        if paths is None:
            LAST_FAILURE_REASON = "empty_solution"
            return None

        LAST_FAILURE_REASON = None
        # Map agent indices back to agent IDs (same order as the scen file).
        return {agents[i]["id"]: paths.get(i, [agents[i]["start"]])
                for i in range(len(agents))}

    except subprocess.TimeoutExpired:
        LAST_FAILURE_REASON = "timeout"
        return None
    except Exception as e:
        LAST_FAILURE_REASON = f"exception:{type(e).__name__}"
        return None
    finally:
        # Clean up temp files (keep the cached map file).
        try:
            os.unlink(scen_path)
        except OSError:
            pass
        try:
            os.unlink(output_path)
        except OSError:
            pass
