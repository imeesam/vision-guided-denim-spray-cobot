import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VISION_SCRIPT = BASE_DIR / "vision_to_waypoints.py"
ROBOT_SCRIPT = BASE_DIR / "simulate_jaka_robodk.py"
WAYPOINTS_JSON = BASE_DIR / "jaka_waypoints.json"
SUPPORTED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".jpg", ".jpeg", ".png"}


def wait_until_copy_complete(file_path: Path) -> None:
    """Wait until Windows finishes copying a dropped video file."""
    prev_size = -1
    while True:
        if not file_path.exists():
            return
        size = file_path.stat().st_size
        if size > 0 and size == prev_size:
            return
        prev_size = size
        time.sleep(0.5)


def move_safely(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        dst = dst_dir / f"{src.stem}_{int(time.time())}{src.suffix}"
    shutil.move(str(src), str(dst))
    return dst


def run_once(media_path: Path) -> bool:
    print(f"\n{'=' * 60}")
    print(f"[PIPELINE] New garment input: {media_path.name}")
    t0 = time.perf_counter()

    # Stage 1: Vision + Safety Gate + 3D Trajectory Generation
    vision_cmd = [
        sys.executable, "-u", str(VISION_SCRIPT),
        "--input", str(media_path),
        "--output", str(WAYPOINTS_JSON),
    ]
    res_vision = subprocess.run(vision_cmd, cwd=BASE_DIR)

    if res_vision.returncode != 0 or not WAYPOINTS_JSON.exists():
        print("[PIPELINE] ABORTED — Vision/Safety Gate rejected input. Robot stayed idle.")
        return False

    t_vision = time.perf_counter() - t0
    print(f"[PIPELINE] Vision + 3D planning finished in {t_vision:.2f}s -> Sending to JAKA Zu5...")

    # Stage 2: 6-DOF IK Verification + RoboDK Execution
    robot_cmd = [sys.executable, "-u", str(ROBOT_SCRIPT)]
    res_robot = subprocess.run(robot_cmd, cwd=BASE_DIR)

    if res_robot.returncode != 0:
        print("[PIPELINE] ERROR — Robot execution script returned an error.")
        return False

    print(f"[PIPELINE] Complete! Total cycle elapsed: {time.perf_counter() - t0:.2f}s")
    return True


def watch_folder(incoming_dir: Path) -> None:
    incoming_dir.mkdir(exist_ok=True)
    processed_dir = BASE_DIR / "processed"
    rejected_dir = BASE_DIR / "rejected"
    print(f"[WATCH MODE] Drop any jeans video/image into: {incoming_dir}")
    print("[WATCH MODE] Keep RoboDK open. Press Ctrl+C to stop.")

    try:
        while True:
            files = sorted(
                p for p in incoming_dir.iterdir()
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
            )
            for media in files:
                wait_until_copy_complete(media)
                ok = run_once(media)
                target_dir = processed_dir if ok else rejected_dir
                moved_to = move_safely(media, target_dir)
                print(f"[WATCH MODE] Archived to: {moved_to.parent.name}/{moved_to.name}")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[WATCH MODE] Stopped cleanly.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Automated Vision-to-JAKA Denim Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Run once on a single video or image")
    group.add_argument("--watch", nargs="?", const="incoming", type=Path,
                       help="Auto-watch a folder for new videos/images (default: incoming/)")
    args = parser.parse_args()

    if args.input:
        if not args.input.exists():
            sys.exit(f"Input file not found: {args.input}")
        sys.exit(0 if run_once(args.input.resolve()) else 1)
    else:
        watch_dir = args.watch if args.watch.is_absolute() else (BASE_DIR / args.watch)
        watch_folder(watch_dir)


if __name__ == "__main__":
    main()