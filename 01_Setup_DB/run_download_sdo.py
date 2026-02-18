"""SDO batch download runner.

Downloads SDO data for multiple target times across telescopes.
Runs AIA and HMI concurrently for each target time.
Tracks progress with JSON file for resume support.
"""
import datetime
import json
import subprocess
import time
from pathlib import Path


# ========== Configuration ==========
PYTHON = "/Users/eunsupark/Softwares/miniconda3/envs/ap-backend/bin/python"
SCRIPT = "scripts/download_sdo.py"

# Date range and interval
START = datetime.datetime(2024, 1, 1)
END = datetime.datetime(2025, 1, 1)
INTERVAL_HOURS = 3

# Telescopes to download (run concurrently per target time)
TARGETS = [
    {"telescope": "aia", "channels": ["193", "211"], "email": "eunsupark@kasi.re.kr"},
    {"telescope": "hmi", "channels": ["m_45s"], "email": "eunsupark@kasi.re.kr"},
]

PARALLEL_DOWNLOADS = 4   # parallel file downloads per subprocess
TIME_RANGE = 12           # JSOC search range in minutes (±)

# Progress file (tracks completed tasks for resume)
PROGRESS_FILE = ".sdo_batch_progress.json"
# ====================================


def load_progress() -> set[str]:
    """Load completed task keys from progress file."""
    path = Path(PROGRESS_FILE)
    if path.exists():
        with open(path) as f:
            return set(json.load(f).get("completed", []))
    return set()


def save_progress(completed: set[str]):
    """Save completed task keys to progress file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "completed": sorted(completed),
            "updated": datetime.datetime.now().isoformat(),
        }, f, indent=2)


def task_key(target_time: datetime.datetime, telescope: str) -> str:
    """Generate unique key for a (time, telescope) pair."""
    return f"{target_time.isoformat()}_{telescope}"


def run_download(target: dict, target_time: datetime.datetime) -> subprocess.Popen:
    """Launch download_sdo.py as subprocess.

    Args:
        target: Target config dict with telescope, channels, email.
        target_time: Target datetime.

    Returns:
        Popen process handle.
    """
    cmd = [
        PYTHON, SCRIPT,
        "--telescope", target["telescope"],
        "--channels", *target["channels"],
        "--email", target["email"],
        "--target-time", target_time.strftime("%Y-%m-%d %H:%M:%S"),
        "--parallel", str(PARALLEL_DOWNLOADS),
        "--time-range", str(TIME_RANGE),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def parse_stdout(stdout: str) -> dict:
    """Extract download/valid counts from script output.

    Args:
        stdout: Script stdout text.

    Returns:
        Dict with 'downloaded' and 'valid' counts.
    """
    info = {"downloaded": 0, "valid": 0, "skipped_db": 0}
    for line in (stdout or "").split("\n"):
        if "Already in DB" in line:
            info["skipped_db"] += 1
        if "Downloaded:" in line:
            try:
                info["downloaded"] = int(line.split("Downloaded:")[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if "Valid:" in line:
            try:
                info["valid"] = int(line.split("Valid:")[1].strip().split(",")[0])
            except (ValueError, IndexError):
                pass
    return info


def format_duration(seconds: float) -> str:
    """Format seconds to human-readable duration string."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def main():
    # Generate all target times
    target_times = []
    current = START
    while current < END:
        target_times.append(current)
        current += datetime.timedelta(hours=INTERVAL_HOURS)

    # Load progress
    completed = load_progress()

    # Build remaining tasks grouped by target_time
    remaining_times = []
    remaining_tasks = {}  # time -> list of targets
    for t in target_times:
        pending = [tgt for tgt in TARGETS if task_key(t, tgt["telescope"]) not in completed]
        if pending:
            remaining_times.append(t)
            remaining_tasks[t] = pending

    total_tasks = len(target_times) * len(TARGETS)
    done_tasks = total_tasks - sum(len(v) for v in remaining_tasks.values())
    telescopes_str = " + ".join(t["telescope"].upper() for t in TARGETS)

    print("=" * 60)
    print("SDO Batch Download")
    print("=" * 60)
    print(f"  Period    : {START:%Y-%m-%d} ~ {END:%Y-%m-%d}")
    print(f"  Interval  : {INTERVAL_HOURS}h ({len(target_times)} time points)")
    print(f"  Telescopes: {telescopes_str}")
    print(f"  Progress  : {done_tasks}/{total_tasks} ({done_tasks * 100 // total_tasks}%)")
    print(f"  Remaining : {len(remaining_times)} time points")
    print("=" * 60)
    print()

    if not remaining_times:
        print("All tasks completed!")
        return

    start_wall = time.time()
    processed_count = 0

    try:
        for time_idx, target_time in enumerate(remaining_times):
            targets_for_time = remaining_tasks[target_time]
            current_num = done_tasks + processed_count + 1

            # Header
            print(f"[{current_num}/{total_tasks}] {target_time:%Y-%m-%d %H:%M:%S}"
                  f"  ({time_idx + 1}/{len(remaining_times)} remaining)")

            # Launch all telescopes concurrently
            procs = {}
            for target in targets_for_time:
                tel = target["telescope"]
                procs[tel] = (target, run_download(target, target_time))

            # Wait for results
            for tel, (target, proc) in procs.items():
                stdout, stderr = proc.communicate()
                ch_str = ",".join(target["channels"])
                key = task_key(target_time, tel)

                if proc.returncode == 0:
                    info = parse_stdout(stdout)
                    if info["skipped_db"] > 0:
                        print(f"  {tel.upper():>3}/{ch_str:<12} SKIP (already in DB)")
                    else:
                        status = f"dl:{info['downloaded']} valid:{info['valid']}"
                        print(f"  {tel.upper():>3}/{ch_str:<12} OK  ({status})")
                    completed.add(key)
                    processed_count += 1
                else:
                    err_lines = [l for l in (stderr or "").strip().split("\n") if l.strip()]
                    err_msg = err_lines[-1][:80] if err_lines else "unknown"
                    print(f"  {tel.upper():>3}/{ch_str:<12} FAIL ({err_msg})")

            save_progress(completed)

            # ETA
            elapsed = time.time() - start_wall
            if processed_count > 0:
                remaining_count = sum(len(v) for v in remaining_tasks.values()) - processed_count
                avg_per_task = elapsed / processed_count
                eta_seconds = avg_per_task * remaining_count
                speed = processed_count / elapsed * 3600
                print(f"  -- {speed:.0f} tasks/h | ETA: {format_duration(eta_seconds)}")

            print()

    except KeyboardInterrupt:
        print("\n\nInterrupted! Progress saved.")
        save_progress(completed)
        return

    # Summary
    elapsed_total = time.time() - start_wall
    final_done = len(completed)
    print("=" * 60)
    print(f"Session: {processed_count} tasks in {format_duration(elapsed_total)}")
    print(f"Total progress: {final_done}/{total_tasks} ({final_done * 100 // total_tasks}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
