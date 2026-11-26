import argparse
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "best.onnx"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ==================== 3D CELL GEOMETRY & RECIPE ====================
GARMENT_HEIGHT_MM = 1000.0
MANNEQUIN_Y_MM = 780.0
WAIST_Z_MM = 850.0
STANDOFF_MM = 200.0
SPRAY_PASSES = 5
U_SWEEP = [0.02, 0.15, 0.32, 0.50, 0.68, 0.85, 0.98]

ANGLE_RANGES = {
    "left_thigh": (-32.0, 8.0),    # Asymmetric wrap prevents inner-seam overlap
    "right_thigh": (-8.0, 32.0),
}
CONTOUR = [
    (0, 1), (1, 2), (2, 13), (13, 12), (12, 11), (11, 10), (10, 9),
    (9, 8), (8, 7), (7, 6), (6, 5), (5, 4), (4, 3), (3, 0),
]
# ===================================================================


def safety_gate(pts: np.ndarray, confs: np.ndarray, min_conf: float = 0.55) -> tuple[bool, str]:
    core = [0, 1, 2, 3, 4, 7, 8, 9, 12, 13]
    if any(confs[k] < min_conf for k in core):
        return False, "Low conf"
    if not (pts[0, 0] < pts[1, 0] < pts[2, 0]):
        return False, "Waist swap"
    if not (pts[8, 1] > pts[1, 1]):
        return False, "Crotch invalid"
    if not (pts[3, 0] < pts[8, 0] < pts[13, 0]):
        return False, "Hip/crotch invalid"
    if not (pts[4, 0] < pts[7, 0] and pts[9, 0] < pts[12, 0]):
        return False, "Knee swap"
    return True, "PASS"


def build_curved_3d_zone(
    tl: np.ndarray,
    tr: np.ndarray,
    bl: np.ndarray,
    br: np.ndarray,
    mm_per_px: float,
    origin_x_px: float,
    waist_y_px: float,
    theta_min_deg: float,
    theta_max_deg: float,
) -> list[dict]:
    waypoints = []
    for row in range(SPRAY_PASSES):
        v = 0.12 + 0.76 * (row / max(SPRAY_PASSES - 1, 1))
        us = U_SWEEP if row % 2 == 0 else list(reversed(U_SWEEP))

        left_edge = (1.0 - v) * tl + v * bl
        right_edge = (1.0 - v) * tr + v * br
        leg_radius_mm = max(float(np.linalg.norm(right_edge - left_edge)) * mm_per_px * 0.5, 45.0)
        leg_center_x_mm = ((left_edge[0] + right_edge[0]) * 0.5 - origin_x_px) * mm_per_px

        for u in us:
            pt_2d = (1.0 - v) * ((1.0 - u) * tl + u * tr) + v * ((1.0 - u) * bl + u * br)
            spray_on = bool(0.14 <= u <= 0.86)

            theta = np.deg2rad(theta_min_deg + u * (theta_max_deg - theta_min_deg))
            nx, ny, nz = float(np.sin(theta)), float(-np.cos(theta)), 0.0

            sx = leg_center_x_mm + leg_radius_mm * np.sin(theta)
            sy = MANNEQUIN_Y_MM - leg_radius_mm * np.cos(theta)
            sz = WAIST_Z_MM - (pt_2d[1] - waist_y_px) * mm_per_px

            px = sx + STANDOFF_MM * nx
            py = sy + STANDOFF_MM * ny
            pz = sz + STANDOFF_MM * nz

            waypoints.append({
                "px_2d": [round(float(pt_2d[0]), 1), round(float(pt_2d[1]), 1)],
                "surface_mm": [round(float(sx), 2), round(float(sy), 2), round(float(sz), 2)],
                "nozzle_mm": [round(float(px), 2), round(float(py), 2), round(float(pz), 2)],
                "normal": [round(nx, 4), round(ny, 4), round(nz, 4)],
                "spray_on": spray_on,
            })
    return waypoints


def build_full_3d_plan(kpts: np.ndarray) -> dict:
    hem_y = float(np.max(kpts[[5, 6, 10, 11], 1]))
    waist_y = float(kpts[1, 1])
    mm_per_px = GARMENT_HEIGHT_MM / max(hem_y - waist_y, 100.0)
    origin_x = float(kpts[1, 0])

    l_tl = 0.25 * kpts[0] + 0.75 * kpts[3]
    l_tr = 0.15 * kpts[1] + 0.85 * kpts[8]
    r_tl = 0.15 * kpts[1] + 0.85 * kpts[8]
    r_tr = 0.25 * kpts[2] + 0.75 * kpts[13]

    return {
        "mm_per_px": round(float(mm_per_px), 3),
        "standoff_mm": STANDOFF_MM,
        "zones": {
            "left_thigh": build_curved_3d_zone(
                l_tl, l_tr, kpts[4], kpts[7],
                mm_per_px, origin_x, waist_y, *ANGLE_RANGES["left_thigh"],
            ),
            "right_thigh": build_curved_3d_zone(
                r_tl, r_tr, kpts[9], kpts[12],
                mm_per_px, origin_x, waist_y, *ANGLE_RANGES["right_thigh"],
            ),
        },
    }


def save_preview(frame: np.ndarray, kpts: np.ndarray, plan: dict, preview_path: Path) -> None:
    vis = frame.copy()
    for a, b in CONTOUR:
        cv2.line(vis, tuple(kpts[a].astype(int)), tuple(kpts[b].astype(int)), (255, 180, 0), 2)
    for i, (x, y) in enumerate(kpts):
        cv2.circle(vis, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.putText(vis, str(i), (int(x) + 5, int(y) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    for wps in plan["zones"].values():
        for i in range(len(wps) - 1):
            p1 = tuple(int(v) for v in wps[i]["px_2d"])
            p2 = tuple(int(v) for v in wps[i + 1]["px_2d"])
            col = (255, 0, 255) if (wps[i]["spray_on"] and wps[i + 1]["spray_on"]) else (0, 255, 255)
            cv2.line(vis, p1, p2, col, 2)
    cv2.imwrite(str(preview_path), vis)


def load_frames(media_path: Path, start_sec: float | None, end_sec: float | None, max_samples: int) -> list[np.ndarray]:
    if media_path.suffix.lower() in IMAGE_EXTS:
        img = cv2.imread(str(media_path))
        if img is None:
            raise RuntimeError(f"Could not read image: {media_path}")
        return [img]

    cap = cv2.VideoCapture(str(media_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {media_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_f = int(start_sec * fps) if start_sec is not None else 0
    end_f = int(end_sec * fps) if end_sec is not None else max(total_frames - 1, 0)
    if total_frames > 0:
        end_f = min(end_f, total_frames - 1)

    span = max(end_f - start_f + 1, 1)
    stride = max(span // max_samples, 1)

    frames = []
    if start_f > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    idx = start_f
    while cap.isOpened() and (total_frames == 0 or idx <= end_f):
        ok, frame = cap.read()
        if not ok:
            break
        if (idx - start_f) % stride == 0:
            frames.append(frame)
            if len(frames) >= max_samples:
                break
        idx += 1

    cap.release()
    return frames


def generate_waypoints_from_media(
    media_path: Path,
    output_json: Path,
    model_path: Path,
    start_sec: float | None = None,
    end_sec: float | None = None,
    max_samples: int = 45,
) -> bool:
    assert model_path.exists(), f"Missing ONNX model: {model_path}"
    model = YOLO(str(model_path), task="pose")

    frames = load_frames(media_path, start_sec, end_sec, max_samples)
    if not frames:
        print("[VISION] No readable frames found.")
        return False

    passed_kpts = []
    reject_reasons: dict[str, int] = {}
    sample_frame = frames[0]

    for frame in frames:
        res = model.predict(frame, imgsz=640, conf=0.45, verbose=False)[0]
        if res.keypoints is None or len(res.keypoints.xy) == 0:
            reject_reasons["No garment"] = reject_reasons.get("No garment", 0) + 1
            continue

        pts = res.keypoints.xy[0].cpu().numpy()
        confs = res.keypoints.conf[0].cpu().numpy()
        passed, msg = safety_gate(pts, confs)

        if passed:
            passed_kpts.append(pts.copy())
            sample_frame = frame
        else:
            reject_reasons[msg] = reject_reasons.get(msg, 0) + 1

    print(f"[VISION] Sampled {len(frames)} frames | Safety gate passed: {len(passed_kpts)}/{len(frames)}")
    if not passed_kpts:
        print(f"[SAFETY GATE] Blocked garment. Reasons: {reject_reasons}")
        return False

    locked_kpts = np.median(np.stack(passed_kpts, axis=0), axis=0)
    plan_3d = build_full_3d_plan(locked_kpts)

    output_json.write_text(
        json.dumps(plan_3d, indent=2, default=lambda x: x.item()),
        encoding="utf-8",
    )

    preview_path = output_json.parent / "last_annotated.jpg"
    save_preview(sample_frame, locked_kpts, plan_3d, preview_path)
    print(f"[VISION OK] Scale: {plan_3d['mm_per_px']} mm/px | Saved: {output_json.name} & {preview_path.name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLO26s-pose ONNX -> Safety Gate -> 3D JAKA Waypoints")
    parser.add_argument("--input", required=True, type=Path, help="Path to jeans image or video")
    parser.add_argument("--output", default=BASE_DIR / "jaka_waypoints.json", type=Path, help="Output JSON path")
    parser.add_argument("--model", default=DEFAULT_MODEL, type=Path, help="Path to best.onnx")
    parser.add_argument("--start", type=float, default=None, help="Optional video start time (sec)")
    parser.add_argument("--end", type=float, default=None, help="Optional video end time (sec)")
    parser.add_argument("--max-samples", type=int, default=45, help="Max video frames to sample for median lock")
    args = parser.parse_args()

    ok = generate_waypoints_from_media(
        media_path=args.input,
        output_json=args.output,
        model_path=args.model,
        start_sec=args.start,
        end_sec=args.end,
        max_samples=args.max_samples,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()