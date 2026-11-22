""" CPU smoke test of the Step 4 pipeline with tiny settings (minutes, not results):

  train cube4 with back_max 30 over two "sessions" (driver, chunked runs) -> snapshots -> session CSV ->
  training-depth plot -> eval of the latest snapshot (A* + beam, J4 + zero baseline, depths 1-3) ->
  eval across all snapshots (solve rate vs training iteration).

    python scripts/step4_smoke.py [--work_dir DIR]
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd):
    print("$ " + " ".join(cmd), flush=True)
    env = dict(os.environ, PYTHONPATH=ROOT, CUDA_VISIBLE_DEVICES="", PYTHONWARNINGS="ignore")
    out = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout[-3000:], out.stderr[-3000:])
        raise RuntimeError("command failed: %s" % " ".join(cmd))
    return out.stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default=None)
    args = parser.parse_args()
    work = args.work_dir or tempfile.mkdtemp(prefix="step4_smoke_")
    save_dir = os.path.join(work, "saved_models")
    snaps = os.path.join(save_dir, "cube4", "snapshots")
    log_csv = os.path.join(work, "sessions.csv")

    train = [sys.executable, "scripts/step4_train.py", "--save_dir", save_dir, "--total_itrs", "6",
             "--num_snapshots", "3", "--log_csv", log_csv, "--back_max", "30", "--batch_size", "50",
             "--states_per_update", "100", "--num_test", "62", "--update_nnet_batch_size", "200"]
    # session 1 is too short for the whole budget: the driver stops before its limit, session 2 resumes
    print(run(train + ["--session_time", "1", "--first_cycle_estimate", "0.1"]))
    print(run(train + ["--session_time", "600"]))
    rows = list(csv.DictReader(open(log_csv)))
    for r in rows:
        print("session %s: %s, itr %s -> %s, deepest depth %s (max %s), GBFS steps %s" % (
            r["session"], r["status"], r["start_itr"], r["end_itr"], r["deepest_depth"], r["deepest_depth_max"],
            r["gbfs_steps"]))
    assert rows[-1]["status"] == "finished" and rows[-1]["end_itr"] == "6"
    assert sorted(os.listdir(snaps)) == ["itr_2", "itr_4", "itr_6"], os.listdir(snaps)

    print(run([sys.executable, "scripts/step4_training_depth.py", "--log",
               os.path.join(save_dir, "cube4", "output.txt"), "--out", os.path.join(work, "training_depth")]))

    eval_args = ["--depths", "1", "2", "3", "--num_per_depth", "3", "--max_nodes", "20000", "--beam_width", "64",
                 "--beam_max_steps", "6"]
    print(run([sys.executable, "scripts/step4_eval.py", "--model_dir", os.path.join(snaps, "itr_6"), "--out",
               os.path.join(work, "eval_itr_6.json")] + eval_args))
    second = run([sys.executable, "scripts/step4_eval.py", "--model_dir", os.path.join(snaps, "itr_6"), "--out",
                  os.path.join(work, "eval_itr_6.json")] + eval_args)
    assert "Resuming: 36 finished sections" in second, second
    res = json.load(open(os.path.join(work, "eval_itr_6.json")))
    assert sum(r["found_but_replay_failed"] for r in res["sections"].values()) == 0
    print("eval re-run skipped all 36 sections; every found path replayed to solved")

    print(run([sys.executable, "scripts/step4_eval.py", "--snapshots_dir", snaps, "--out_dir",
               os.path.join(work, "snapshot_eval"), "--targets", "native4"] + eval_args))
    print("work dir: %s" % work)


if __name__ == "__main__":
    main()
