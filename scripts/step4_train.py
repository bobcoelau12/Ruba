""" Step 4 training driver: runs ctg_approx/avi.py (unchanged training algorithm) over several Colab sessions.

* Resume: every call continues from <save_dir>/<nnet_name>/current (avi.py's own checkpoint) until --total_itrs.
* Clean stop: --session_time is this session's wall-clock budget. avi.py checks its budget only before an update
  cycle, so the driver passes it the remaining time minus 1.5x the longest measured cycle; the last cycle therefore
  ends before the session limit instead of being cut off mid-cycle.
* Snapshots: at every 1/--num_snapshots of --total_itrs the current model is copied to
  <save_dir>/<nnet_name>/snapshots/itr_<N>/ (written to a temporary folder, then renamed; an existing snapshot is
  never overwritten). To land exactly on the boundaries, avi.py is run in chunks with --max_itrs set to the next
  boundary, and boundaries are rounded up to whole update cycles (ceil(states_per_update / batch_size) iterations).
* Session log: one row per session in --log_csv (start/end iteration and time, update_num, deepest depth of the
  latest greedy check and over all checks so far, iterations per hour, status). The row is rewritten after every
  chunk; a row still marked "running" when the next session starts belonged to a session that died, and is closed
  as "interrupted" with the checkpoint iteration it left behind.

Arguments this script does not know are passed to avi.py unchanged (e.g. --loss_thresh, --lr).
"""
import argparse
import csv
import datetime
import math
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.step4_training_depth import parse_log, summarize  # noqa: E402

CSV_FIELDS = ["session", "start_utc", "end_utc", "start_itr", "end_itr", "update_num", "deepest_depth",
              "deepest_depth_max", "gbfs_steps", "itrs_per_hour", "status"]
SHOW = re.compile(r"device:|Training model|Itr: \d*000,|Greedy test|Back Steps|Last loss|Updating target|"
                  r"Time budget|Done|Error|Traceback")


def now_utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def cycle_itrs(states_per_update: int, batch_size: int) -> int:
    return int(math.ceil(states_per_update / batch_size))


def snapshot_boundaries(total_itrs: int, num_snapshots: int, cycle: int) -> List[int]:
    """ Iterations at which snapshots are taken: k/num_snapshots of the total, rounded up to whole cycles. """
    total = int(math.ceil(total_itrs / cycle)) * cycle
    bounds = [int(math.ceil(total * k / num_snapshots / cycle)) * cycle for k in range(1, num_snapshots + 1)]
    return sorted(set(bounds))


def read_pickle(path: str, default):
    return pickle.load(open(path, "rb")) if os.path.isfile(path) else default


def take_snapshots(curr_dir: str, snap_dir: str, itr: int, bounds: List[int]) -> List[str]:
    """ For every boundary <= itr without a snapshot in [boundary, next boundary), copy current/ to itr_<itr>. """
    os.makedirs(snap_dir, exist_ok=True)
    existing = []
    for name in os.listdir(snap_dir):
        m = re.fullmatch(r"itr_(\d+)", name)
        if m:
            existing.append(int(m.group(1)))
    made = []
    for k, b in enumerate(bounds):
        upper = bounds[k + 1] if k + 1 < len(bounds) else float("inf")
        if itr < b or any(b <= e < upper for e in existing):
            continue
        if itr >= upper:  # boundary passed without a snapshot (should not happen with chunked runs)
            print("WARNING: no snapshot was taken at boundary %i; current itr is already %i" % (b, itr))
            continue
        final = os.path.join(snap_dir, "itr_%i" % itr)
        if os.path.exists(final):
            continue
        tmp = os.path.join(snap_dir, ".tmp_itr_%i" % itr)
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(tmp)
        for f in os.listdir(curr_dir):
            if os.path.isfile(os.path.join(curr_dir, f)) and not f.endswith(".tmp"):
                shutil.copy2(os.path.join(curr_dir, f), tmp)
        os.rename(tmp, final)
        existing.append(itr)
        made.append(final)
        print("Snapshot: %s" % final, flush=True)
    return made


def read_rows(path: str) -> List[Dict]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    os.replace(tmp, path)


def fill_row(row: Dict, itr: int, update_num: int, log_file: str, elapsed: float) -> None:
    s = summarize(parse_log(log_file)) if os.path.isfile(log_file) else summarize([])
    row.update({"end_itr": itr, "update_num": update_num,
                "deepest_depth": "" if s["deepest_depth"] is None else s["deepest_depth"],
                "deepest_depth_max": "" if s["deepest_depth_max"] is None else s["deepest_depth_max"],
                "gbfs_steps": "" if s["gbfs_steps"] is None else s["gbfs_steps"],
                "itrs_per_hour": "%.0f" % ((itr - int(row["start_itr"])) / elapsed * 3600) if elapsed > 0 else ""})


def run_avi(avi_args: List[str]) -> None:
    cmd = [sys.executable, "-u", os.path.join(ROOT, "ctg_approx", "avi.py")] + avi_args
    env = dict(os.environ, PYTHONPATH=ROOT)
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            bufsize=1)
    tail: List[str] = []
    for line in proc.stdout:
        tail = (tail + [line])[-40:]
        if SHOW.search(line):
            print(line, end="", flush=True)
    if proc.wait() != 0:
        print("".join(tail))
        raise RuntimeError("avi.py exited with code %i" % proc.returncode)


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--nnet_name', type=str, default="cube4")
    parser.add_argument('--env', type=str, default="cube4")
    parser.add_argument('--total_itrs', type=int, required=True, help="Total training iterations over all sessions")
    parser.add_argument('--session_time', type=float, required=True, help="Wall-clock budget of this session (s)")
    parser.add_argument('--first_cycle_estimate', type=float, default=300.0,
                        help="Assumed duration (s) of one update cycle until one has been measured")
    parser.add_argument('--num_snapshots', type=int, default=10)
    parser.add_argument('--log_csv', type=str, required=True)
    parser.add_argument('--back_max', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--states_per_update', type=int, default=200000)
    args, passthrough = parser.parse_known_args(argv)

    start_time = time.time()
    model_dir = os.path.join(args.save_dir, args.nnet_name)
    curr_dir = os.path.join(model_dir, "current")
    snap_dir = os.path.join(model_dir, "snapshots")
    log_file = os.path.join(model_dir, "output.txt")

    cycle = cycle_itrs(args.states_per_update, args.batch_size)
    bounds = snapshot_boundaries(args.total_itrs, args.num_snapshots, cycle)
    total = bounds[-1]
    itr = int(read_pickle(os.path.join(curr_dir, "train_itr.pkl"), 0))
    update_num = int(read_pickle(os.path.join(curr_dir, "update_num.pkl"), 0))
    print("Total itrs %i (cycle %i itrs), snapshots at %s; checkpoint itr %i" % (total, cycle, bounds, itr),
          flush=True)

    rows = read_rows(args.log_csv)
    for r in rows:
        if r["status"] == "running":
            r["status"] = "interrupted"
            r["end_itr"] = itr
    row = {"session": len(rows) + 1, "start_utc": now_utc(), "start_itr": itr, "status": "running"}
    rows.append(row)
    fill_row(row, itr, update_num, log_file, 0.0)
    write_rows(args.log_csv, rows)

    avi_fixed = ["--env", args.env, "--nnet_name", args.nnet_name, "--save_dir", args.save_dir,
                 "--back_max", str(args.back_max), "--batch_size", str(args.batch_size),
                 "--states_per_update", str(args.states_per_update)] + passthrough
    cycle_times: List[float] = []  # measured seconds per update cycle (includes avi.py start-up)
    status = "finished"
    while True:
        take_snapshots(curr_dir, snap_dir, itr, bounds)
        if itr >= total:
            break
        # safety factor: a chunk's mean cycle time hides slower cycles (the greedy check grows with update_num)
        longest_cycle = 1.5 * max(cycle_times) if cycle_times else args.first_cycle_estimate
        remaining = args.session_time - (time.time() - start_time)
        if remaining - longest_cycle <= 0:
            status = "time_limit"
            break
        target = min(b for b in bounds if b > itr)
        chunk_start, itr_before = time.time(), itr
        run_avi(avi_fixed + ["--max_itrs", str(target), "--max_time", "%.0f" % (remaining - longest_cycle)])
        itr = int(read_pickle(os.path.join(curr_dir, "train_itr.pkl"), 0))
        update_num = int(read_pickle(os.path.join(curr_dir, "update_num.pkl"), 0))
        cycles_done = (itr - itr_before) / cycle
        if cycles_done > 0:
            cycle_times.append((time.time() - chunk_start) / cycles_done)
        fill_row(row, itr, update_num, log_file, time.time() - start_time)
        write_rows(args.log_csv, rows)
        if itr == itr_before:  # avi.py stopped on its time budget before any cycle
            status = "time_limit"
            break

    row.update({"end_utc": now_utc(), "status": status})
    fill_row(row, itr, update_num, log_file, time.time() - start_time)
    write_rows(args.log_csv, rows)
    print("Session %s %s: itr %s -> %s, deepest depth %s (max so far %s), %s itrs/hour" % (
        row["session"], status, row["start_itr"], row["end_itr"], row["deepest_depth"], row["deepest_depth_max"],
        row["itrs_per_hour"]), flush=True)


if __name__ == "__main__":
    main()
