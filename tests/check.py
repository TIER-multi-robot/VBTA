from hvbta.profiles import STRICT_ROBOT_PROFILES, STRICT_TASK_PROFILES

# Hard gates: payload >= required, reach >= required, at-least-one manipulator matches
print("=== Robot profiles (hard gates only) ===")
for rp in STRICT_ROBOT_PROFILES:
    print(f"  {rp['name']:15} payload={rp['payload_capacity']:5.1f}  reach={rp['reach']:5.1f}  manips={rp['manipulators']}")

print("\n=== Task profiles (hard gates only) ===")
for tp in STRICT_TASK_PROFILES:
    rc = tp["required_capabilities"]
    print(f"  {tp['task_type']:20} payload={rc['payload']:5.1f}  reach={rc['reach']:5.1f}  manips={tp.get('manipulators_needed', [])}")

print("\n=== Compatibility matrix (P=payload, R=reach, M=manipulator) ===")
print(f"{'Task':<22}", end="")
for rp in STRICT_ROBOT_PROFILES:
    print(f"{rp['name']:>12}", end="")
print()
print("-" * (22 + 12 * len(STRICT_ROBOT_PROFILES)))

for tp in STRICT_TASK_PROFILES:
    print(f"{tp['task_type']:<22}", end="")
    for rp in STRICT_ROBOT_PROFILES:
        req_p = tp["required_capabilities"]["payload"]
        req_r = tp["required_capabilities"]["reach"]
        needed_m = tp.get("manipulators_needed", [])
        p_ok = rp["payload_capacity"] >= req_p
        r_ok = rp["reach"] >= req_r
        m_ok = any(m in rp["manipulators"] for m in needed_m) if needed_m else True
        ok = p_ok and r_ok and m_ok
        if ok:
            print(f"{'OK':>12}", end="")
        else:
            reasons = []
            if not p_ok: reasons.append("P")
            if not r_ok: reasons.append("R")
            if not m_ok: reasons.append("M")
            print(f"{'X('+''.join(reasons)+')':>12}", end="")
    print()

print("\n=== Tasks per robot ===")
for rp in STRICT_ROBOT_PROFILES:
    tasks = []
    for tp in STRICT_TASK_PROFILES:
        req_p = tp["required_capabilities"]["payload"]
        req_r = tp["required_capabilities"]["reach"]
        needed_m = tp.get("manipulators_needed", [])
        if (rp["payload_capacity"] >= req_p and rp["reach"] >= req_r
                and any(m in rp["manipulators"] for m in needed_m) if needed_m else True):
            tasks.append(tp["task_type"])
    print(f"  {rp['name']:15}: {len(tasks)}/10 — {tasks}")

print("\n=== Robots per task ===")
for tp in STRICT_TASK_PROFILES:
    robots = []
    for rp in STRICT_ROBOT_PROFILES:
        req_p = tp["required_capabilities"]["payload"]
        req_r = tp["required_capabilities"]["reach"]
        needed_m = tp.get("manipulators_needed", [])
        if (rp["payload_capacity"] >= req_p and rp["reach"] >= req_r
                and any(m in rp["manipulators"] for m in needed_m) if needed_m else True):
            robots.append(rp["name"])
    print(f"  {tp['task_type']:20}: {len(robots)}/6 — {robots}")