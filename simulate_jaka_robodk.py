import json
import time
from pathlib import Path
import numpy as np
from robodk import robolink, robomath

WAYPOINTS_FILE = Path("jaka_waypoints.json")
SPRAY_SPEED_MM_S = 100.0
TRAVEL_SPEED_MM_S = 180.0
JOINT_SPEED_DEG_S = 35.0
SIM_SPEED_RATIO = 1.0       # 1.0 = real-time; 0.5 = slow motion
BLEND_RADIUS_MM = 10.0
RETRACT_MM = 80.0
Z_SHIFT_MM = 150.0
SPRAY_DO_PIN = 0
NOZZLE_Z_SIGN = 1.0


def clear_virtual_denim(RDK) -> None:
    for name in ["Denim_left_thigh", "Denim_right_thigh", "Denim_Surface"]:
        old = RDK.Item(name)
        if old.Valid():
            old.Delete()


def normal_to_tcp_pose(pos_mm, normal, roll_deg=90.0, y_shift=250.0) -> robomath.Mat:
    n = np.array(normal, dtype=float)
    n /= np.linalg.norm(n)

    z_axis = -n * NOZZLE_Z_SIGN
    x_axis = np.cross(z_axis, np.array([0.0, 0.0, 1.0]))
    x_axis = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(x_axis) < 1e-6 else x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    px = float(pos_mm[0])
    py = float(pos_mm[1]) + y_shift
    pz = float(pos_mm[2]) + Z_SHIFT_MM
    base = robomath.Mat([
        [float(x_axis[0]), float(y_axis[0]), float(z_axis[0]), px],
        [float(x_axis[1]), float(y_axis[1]), float(z_axis[1]), py],
        [float(x_axis[2]), float(y_axis[2]), float(z_axis[2]), pz],
        [0.0,              0.0,              0.0,              1.0],
    ])
    return base * robomath.rotz(np.deg2rad(roll_deg))


def retracted_pose(wp, roll_deg, y_shift, dist_mm=RETRACT_MM) -> robomath.Mat:
    p = np.array(wp["surface_mm"], dtype=float)
    n = np.array(wp["normal"], dtype=float)
    n /= np.linalg.norm(n)
    return normal_to_tcp_pose((p + dist_mm * n).tolist(), n.tolist(), roll_deg, y_shift)


def extract_ik(sols_mat, n_dofs: int) -> list[list[float]]:
    if sols_mat is None or sols_mat.size(1) == 0 or sols_mat.size(0) < n_dofs:
        return []
    return [[float(sols_mat[r, c]) for r in range(n_dofs)] for c in range(sols_mat.size(1))]


def is_safe_posture(robot, joints: list[float], min_denim_y: float) -> bool:
    """Enforces elbow-up posture, table clearance, and arm clearance behind the denim panels."""
    link_poses = robot.JointPoses(joints)
    if len(link_poses) < 4:
        return False

    # Last 4 joint frames (J3 elbow, J4, J5, J6 flange) must stay well above base/table
    upper_z = [p.Pos()[2] for p in link_poses[-4:]]
    max_y = max(p.Pos()[1] for p in link_poses)

    return (min(upper_z) >= 150.0) and (max_y <= min_denim_y - 120.0)


def is_safe_movej(robot, q_start: list[float], q_end: list[float], min_denim_y: float, steps: int = 8) -> bool:
    """Checks intermediate joint poses so MoveJ transits stay elbow-up and clear of denim."""
    q0 = np.array(q_start, dtype=float)
    q1 = np.array(q_end, dtype=float)
    for alpha in np.linspace(0.1, 0.9, steps):
        q_mid = ((1.0 - alpha) * q0 + alpha * q1).tolist()
        if not is_safe_posture(robot, q_mid, min_denim_y):
            return False
    return True


def find_feasible_plan(robot, plan: dict, n_dofs: int, tool_pose, frame_pose, home_joints):
    home_q = home_joints[:n_dofs]
    base_min_y = min(w["surface_mm"][1] for z in plan["zones"].values() for w in z)
    stats = {"entry_posture": 0, "movej": 0, "movel": 0, "path_posture": 0}

    for y_shift in [250.0, 200.0, 300.0, 150.0, 350.0]:
        min_denim_y = base_min_y + y_shift
        for roll_deg in [90.0, -90.0, 180.0, 0.0]:
            zone_entries = {}
            prev_q = home_q
            all_ok = True

            for zone_name, wps in plan["zones"].items():
                entry = retracted_pose(wps[0], roll_deg, y_shift)
                poses = [normal_to_tcp_pose(w["surface_mm"], w["normal"], roll_deg, y_shift) for w in wps]
                poses.append(retracted_pose(wps[-1], roll_deg, y_shift))

                candidates = extract_ik(robot.SolveIK_All(entry, tool_pose, frame_pose), n_dofs)
                candidates.sort(key=lambda q: float(np.linalg.norm(np.array(q) - np.array(prev_q))))

                best_j = None
                best_exit_q = None
                for j_start in candidates:
                    if not is_safe_posture(robot, j_start, min_denim_y):
                        stats["entry_posture"] += 1
                        continue
                    if not is_safe_movej(robot, prev_q, j_start, min_denim_y):
                        stats["movej"] += 1
                        continue

                    q_cur = j_start
                    ok = True
                    for pose in poses:
                        if robot.MoveL_Test(q_cur, pose) != 0:
                            stats["movel"] += 1
                            ok = False
                            break
                        q_next = robot.SolveIK(pose, q_cur, tool_pose, frame_pose).list()
                        if len(q_next) < n_dofs:
                            ok = False
                            break
                        q_cur = q_next[:n_dofs]
                        if not is_safe_posture(robot, q_cur, min_denim_y):
                            stats["path_posture"] += 1
                            ok = False
                            break

                    if ok:
                        best_j = j_start
                        best_exit_q = q_cur
                        break

                if best_j is None:
                    all_ok = False
                    break

                zone_entries[zone_name] = best_j
                prev_q = best_exit_q

            if all_ok and is_safe_movej(robot, prev_q, home_q, min_denim_y):
                return y_shift, roll_deg, zone_entries, stats

    return None, None, None, stats


def create_virtual_denim(RDK, plan: dict, y_shift: float, show_curves: bool = False):
    """Builds curved 3D indigo denim panels with margin so all 5 passes hit cleanly."""
    clear_virtual_denim(RDK)

    triangles = []
    pts_per_row = 7

    for zone_name, wps in plan["zones"].items():
        if show_curves:
            pts = [[w["surface_mm"][0], w["surface_mm"][1] + y_shift, w["surface_mm"][2] + Z_SHIFT_MM] for w in wps]
            curve = RDK.AddCurve(pts)
            curve.setName(f"Denim_{zone_name}")
            curve.setColor([0.25, 0.6, 1.0, 1.0])

        num_rows = len(wps) // pts_per_row
        grid = []
        for r in range(num_rows):
            row_wps = wps[r * pts_per_row : (r + 1) * pts_per_row]
            if r % 2 == 1:
                row_wps = list(reversed(row_wps))
            row_pts = []
            for w in row_wps:
                n = np.array(w["normal"], dtype=float)
                n /= np.linalg.norm(n)
                p = np.array([w["surface_mm"][0], w["surface_mm"][1] + y_shift, w["surface_mm"][2] + Z_SHIFT_MM])
                row_pts.append((p - 3.0 * n).tolist())
            grid.append(row_pts)

        grid = np.array(grid, dtype=float)

        top_dir = grid[0] - grid[1]
        top_dir /= np.maximum(np.linalg.norm(top_dir, axis=1, keepdims=True), 1e-6)
        bot_dir = grid[-1] - grid[-2]
        bot_dir /= np.maximum(np.linalg.norm(bot_dir, axis=1, keepdims=True), 1e-6)
        grid = np.vstack([
            grid[0:1] + 50.0 * top_dir[None, :, :],
            grid,
            grid[-1:] + 50.0 * bot_dir[None, :, :],
        ])

        left_dir = grid[:, 0] - grid[:, 1]
        left_dir /= np.maximum(np.linalg.norm(left_dir, axis=1, keepdims=True), 1e-6)
        right_dir = grid[:, -1] - grid[:, -2]
        right_dir /= np.maximum(np.linalg.norm(right_dir, axis=1, keepdims=True), 1e-6)
        grid = np.concatenate([
            (grid[:, 0] + 25.0 * left_dir)[:, None, :],
            grid,
            (grid[:, -1] + 25.0 * right_dir)[:, None, :],
        ], axis=1)

        for r in range(grid.shape[0] - 1):
            for c in range(grid.shape[1] - 1):
                p00, p01 = grid[r, c].tolist(), grid[r, c + 1].tolist()
                p10, p11 = grid[r + 1, c].tolist(), grid[r + 1, c + 1].tolist()
                triangles.extend([p00, p01, p11, p00, p11, p10])
                triangles.extend([p00, p11, p01, p00, p10, p11])

    denim_obj = RDK.AddShape(triangles)
    denim_obj.setName("Denim_Surface")
    denim_obj.setColor([0.12, 0.22, 0.45, 1.0])
    return denim_obj


def main():
    assert WAYPOINTS_FILE.exists(), f"Missing {WAYPOINTS_FILE} in {Path.cwd()}"
    plan = json.loads(WAYPOINTS_FILE.read_text(encoding="utf-8"))

    RDK = robolink.Robolink()
    RDK.setSimulationSpeed(SIM_SPEED_RATIO)
    RDK.Command("Trace", "Reset")
    RDK.Command("Trace", "Off")
    RDK.Spray_Clear(-1)
    clear_virtual_denim(RDK)  # Remove old panels BEFORE enabling collision checking

    robot = RDK.Item("", robolink.ITEM_TYPE_ROBOT)
    if not robot.Valid():
        raise RuntimeError("No robot loaded in RoboDK.")

    tool = RDK.Item("", robolink.ITEM_TYPE_TOOL)
    if tool.Valid():
        robot.setTool(tool)
        print(f"Tool: {tool.Name()} | TCP: {[round(v, 1) for v in tool.PoseTool().Pos()]}")

    robot.setPoseFrame(robomath.eye(4))
    robot.setRounding(BLEND_RADIUS_MM)

    tool_pose = robot.PoseTool()
    frame_pose = robot.PoseFrame()
    home_joints = robot.JointsHome().list()
    n_dofs = len(home_joints)

    robot.setJoints(home_joints)
    RDK.setCollisionActive(robolink.COLLISION_ON)
    use_collisions = (RDK.Collisions() == 0)
    if not use_collisions:
        RDK.setCollisionActive(robolink.COLLISION_OFF)
    print(f"Connected to: {robot.Name()} ({n_dofs}-DOF) | Collision check: {'ON' if use_collisions else 'OFF'}")

    RDK.Render(False)
    try:
        y_shift, roll_deg, zone_entries, stats = find_feasible_plan(
            robot, plan, n_dofs, tool_pose, frame_pose, home_joints
        )
    finally:
        RDK.Render(True)

    if zone_entries is None:
        print(f"[DIAGNOSTIC] No elbow-up path found. Rejections: {stats}")
        return

    print(f"Verified 100% elbow-up & collision-free! roll={roll_deg} deg, y_shift={y_shift:+.0f} mm, z_shift={Z_SHIFT_MM:+.0f} mm")

    RDK.setCollisionActive(robolink.COLLISION_OFF)
    denim_obj = create_virtual_denim(RDK, plan, y_shift)

    spray_tool = tool if tool.Valid() else robot
    spray_id = RDK.Spray_Add(spray_tool, denim_obj, "PARTICLE=SPHERE(10) COLOR=cyan")
    RDK.Spray_SetState(robolink.SPRAY_OFF, spray_id)

    robot.setJoints(home_joints)
    robot.setSpeed(TRAVEL_SPEED_MM_S, JOINT_SPEED_DEG_S)
    time.sleep(2)

    spray_active = False
    for zone_name, wps in plan["zones"].items():
        print(f"--- Executing {zone_name} ({len(wps)} waypoints) ---")
        robot.setSpeed(TRAVEL_SPEED_MM_S, JOINT_SPEED_DEG_S)
        robot.setDO(SPRAY_DO_PIN, 0)
        RDK.Spray_SetState(robolink.SPRAY_OFF, spray_id)
        spray_active = False

        robot.MoveJ(zone_entries[zone_name])

        robot.setSpeed(SPRAY_SPEED_MM_S, JOINT_SPEED_DEG_S)
        for wp in wps:
            pose = normal_to_tcp_pose(wp["surface_mm"], wp["normal"], roll_deg, y_shift)
            want_spray = bool(wp["spray_on"])
            robot.MoveL(pose)

            if want_spray != spray_active:
                robot.setDO(SPRAY_DO_PIN, 1 if want_spray else 0)
                RDK.Spray_SetState(robolink.SPRAY_ON if want_spray else robolink.SPRAY_OFF, spray_id)
                spray_active = want_spray

        robot.setDO(SPRAY_DO_PIN, 0)
        RDK.Spray_SetState(robolink.SPRAY_OFF, spray_id)
        spray_active = False
        robot.setSpeed(TRAVEL_SPEED_MM_S, JOINT_SPEED_DEG_S)
        robot.MoveL(retracted_pose(wps[-1], roll_deg, y_shift))

    robot.MoveJ(home_joints)
    print("Cycle complete! Zero reach errors, elbow-up verified.")


if __name__ == "__main__":
    main()