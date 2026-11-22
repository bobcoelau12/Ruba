import csv
import json
import os
import pickle
import subprocess
import sys

import pytest

from scripts import step4_eval as ev
from scripts import step4_train as tr
from scripts.step4_training_depth import deepest_depth, parse_log, summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def check_block(itr, update_num, steps, solved, complete=True, new_format=True):
    lines = []
    if new_format:
        lines.append("Greedy test at itr %i (update_num %i, back_max 30)" % (itr, update_num))
    lines.append("Solving 100 states with GBFS with %i steps" % steps)
    for d, p in solved.items():
        lines.append("Back Steps: %i, %%Solved: %.2f, avgSolveSteps: 1.00, CTG Mean(Std/Min/Max): 1.00("
                     "0.10/0.90/1.10)" % (d, p))
    if complete:
        lines.append("Test time: 1.23")
    return lines


def training_block(update_num, first_itr, itrs):
    return ["Training model for update number %i for %i iterations" % (update_num, itrs),
            "Itr: %i, lr: 1.00E-03, loss: 1.00E-01, targ_ctg: 1.00, nnet_ctg: 1.00, Time: 0.10" % first_itr,
            "Itr: %i, lr: 1.00E-03, loss: 1.00E-01, targ_ctg: 1.00, nnet_ctg: 1.00, Time: 0.10" % (first_itr + 100)]


# ---------------------------------------------------------------- deepest depth from the log

def test_deepest_depth_definition():
    assert deepest_depth({0: 100, 1: 90, 2: 50.0, 3: 49.99}) == 2
    assert deepest_depth({0: 100, 1: 40, 2: 60}) == 2  # largest depth >= 50%, not the first one below
    assert deepest_depth({0: 10}) == 0 and deepest_depth({}) == 0


def test_parse_log_sessions_interruptions_and_old_format(tmp_path):
    lines = ["device: cpu"]
    # old format (Step 3 logs): itr = first printed Itr + iterations of the training block
    lines += training_block(0, 0, 200) + ["Last loss was 0.1"] + check_block(0, 0, 1, {0: 100, 1: 80}, new_format=False)
    lines += training_block(0, 200, 200) + check_block(400, 0, 1, {0: 100, 1: 90, 2: 10})
    # cut off by a disconnect: no "Test time" -> ignored
    lines += training_block(1, 400, 200) + check_block(600, 1, 2, {0: 100, 1: 99}, complete=False)
    # next session resumes from the checkpoint at 600 and repeats nothing; a repeated itr keeps the last check
    lines += ["device: cuda:0"] + training_block(1, 600, 200) + check_block(800, 1, 2, {0: 100, 1: 95, 2: 55})
    lines += training_block(1, 600, 200) + check_block(800, 1, 2, {0: 100, 1: 96, 2: 70, 3: 52})
    log = tmp_path / "output.txt"
    log.write_text("\n".join(lines) + "\n")

    checks = parse_log(str(log))
    assert [c["itr"] for c in checks] == [200, 400, 800]
    assert checks[0]["update_num"] == 0 and checks[0]["solved"] == {0: 100.0, 1: 80.0}
    assert checks[2]["solved"][3] == 52.0 and checks[2]["gbfs_steps"] == 2
    s = summarize(checks)
    assert s == {"itr": 800, "deepest_depth": 3, "deepest_depth_max": 3, "gbfs_steps": 2}


def test_training_depth_script_writes_csv_and_plots(tmp_path):
    log = tmp_path / "output.txt"
    log.write_text("\n".join(check_block(200, 0, 1, {0: 100, 1: 60}) + check_block(400, 1, 2, {0: 100, 1: 90, 2: 51}))
                   + "\n")
    out = subprocess.run([sys.executable, "scripts/step4_training_depth.py", "--log", str(log), "--out",
                          str(tmp_path / "plots")], cwd=ROOT, capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=ROOT))
    assert out.returncode == 0, out.stderr
    for f in ("training_depth.csv", "training_depth.png", "training_solved.png"):
        assert (tmp_path / "plots" / f).is_file()
    rows = list(csv.DictReader(open(tmp_path / "plots" / "training_depth.csv")))
    assert [r["deepest_depth"] for r in rows] == ["1", "2"]


# ---------------------------------------------------------------- snapshots

def test_snapshot_boundaries_are_whole_cycles():
    assert tr.cycle_itrs(200000, 1000) == 200
    b = tr.snapshot_boundaries(100000, 10, 200)
    assert b == list(range(10000, 100001, 10000))
    b = tr.snapshot_boundaries(1000, 10, 200)  # 10% of 1000 is less than a cycle: boundaries merge
    assert b == [200, 400, 600, 800, 1000]
    b = tr.snapshot_boundaries(1100, 10, 200)  # the total is rounded up to whole cycles
    assert b[-1] == 1200 and all(x % 200 == 0 for x in b)


def test_take_snapshots_never_overwrites(tmp_path):
    curr, snaps = tmp_path / "current", tmp_path / "snapshots"
    curr.mkdir()
    (curr / "model_state_dict.pt").write_text("v1")
    (curr / "model_state_dict.pt.tmp").write_text("partial")
    bounds = [2, 4, 6]
    assert tr.take_snapshots(str(curr), str(snaps), 0, bounds) == []
    made = tr.take_snapshots(str(curr), str(snaps), 2, bounds)
    assert made == [str(snaps / "itr_2")]
    assert not (snaps / "itr_2" / "model_state_dict.pt.tmp").exists()
    (curr / "model_state_dict.pt").write_text("v2")
    assert tr.take_snapshots(str(curr), str(snaps), 2, bounds) == []  # boundary 2 already has a snapshot
    assert (snaps / "itr_2" / "model_state_dict.pt").read_text() == "v1"
    os.makedirs(snaps / ".tmp_itr_4")  # left behind by an interrupted copy
    assert tr.take_snapshots(str(curr), str(snaps), 4, bounds) == [str(snaps / "itr_4")]
    assert (snaps / "itr_4" / "model_state_dict.pt").read_text() == "v2"
    assert not (snaps / ".tmp_itr_4").exists()


# ---------------------------------------------------------------- gate

def section(k, n):
    return {"solved": k, "n": n, "solve_rate": k / n}


def test_wilson_and_clearly_above():
    lo, hi = ev.wilson(0, 50)
    assert lo == 0.0 and 0.06 < hi < 0.08
    lo, hi = ev.wilson(50, 50)
    assert hi == pytest.approx(1.0) and 0.92 < lo < 0.94
    assert ev.clearly_above(15, 50, 0, 50)  # 30 points, intervals apart
    assert not ev.clearly_above(9, 50, 0, 50)  # 18 points: below the 20-point threshold
    assert not ev.clearly_above(30, 50, 20, 50)  # 20 points but the intervals overlap
    assert not ev.clearly_above(3, 5, 0, 5)  # 60 points but with n=5 the intervals overlap


def test_gate_check():
    base = {"native4/astar/zero/d%i" % d: section(0, 50) for d in (5, 6, 8)}
    base.update({"native4/beam/zero/d%i" % d: section(5, 50) for d in (5, 6, 8)})
    j4 = {"native4/astar/J4/d5": section(0, 50), "native4/astar/J4/d6": section(40, 50),
          "native4/astar/J4/d8": section(30, 50), "native4/beam/J4/d6": section(45, 50),
          "native4/beam/J4/d8": section(10, 50)}
    assert ev.gate_check(j4, base, verbose=False) is True  # depth 5 is not part of the gate; A* passes 6 and 8
    j4["native4/astar/J4/d8"] = section(14, 50)  # 28 - 10 = 18 points over the best baseline (beam, 10%)
    assert ev.gate_check(j4, base, verbose=False) is False
    assert ev.gate_check({"native4/astar/J4/d3": section(5, 5)}, base, verbose=False) is None


# ---------------------------------------------------------------- tiny CPU run: train -> snapshot -> eval

def test_tiny_train_snapshot_eval(tmp_path):
    env = dict(os.environ, PYTHONPATH=ROOT, CUDA_VISIBLE_DEVICES="", PYTHONWARNINGS="ignore")
    save_dir, log_csv = tmp_path / "saved_models", tmp_path / "sessions.csv"
    train = [sys.executable, "scripts/step4_train.py", "--save_dir", str(save_dir), "--total_itrs", "4",
             "--session_time", "600", "--num_snapshots", "2", "--log_csv", str(log_csv), "--back_max", "30",
             "--batch_size", "50", "--states_per_update", "100", "--num_test", "31", "--update_nnet_batch_size", "200"]
    out = subprocess.run(train, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
    model_dir = save_dir / "cube4"
    assert sorted(os.listdir(model_dir / "snapshots")) == ["itr_2", "itr_4"]
    assert pickle.load(open(model_dir / "snapshots" / "itr_2" / "train_itr.pkl", "rb")) == 2
    assert pickle.load(open(model_dir / "current" / "train_itr.pkl", "rb")) == 4
    assert pickle.load(open(model_dir / "args.pkl", "rb")).back_max == 30
    checks = parse_log(str(model_dir / "output.txt"))
    assert [c["itr"] for c in checks] == [2, 4] and max(checks[0]["solved"]) == 30

    rows = list(csv.DictReader(open(log_csv)))
    assert len(rows) == 1 and rows[0]["status"] == "finished"
    assert (rows[0]["start_itr"], rows[0]["end_itr"]) == ("0", "4") and rows[0]["deepest_depth"] != ""

    # a session that died leaves a "running" row; the next session closes it and, at the budget, does nothing
    rows[0]["status"] = "running"
    tr.write_rows(str(log_csv), rows)
    out = subprocess.run(train, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    rows = list(csv.DictReader(open(log_csv)))
    assert [r["status"] for r in rows] == ["interrupted", "finished"]
    assert (rows[1]["start_itr"], rows[1]["end_itr"]) == ("4", "4")
    assert sorted(os.listdir(model_dir / "snapshots")) == ["itr_2", "itr_4"]

    # evaluate one snapshot with both solvers, then all snapshots
    eval_args = ["--targets", "native4", "embedded2", "--depths", "1", "2", "--num_per_depth", "2",
                 "--max_nodes", "20000", "--beam_width", "64", "--beam_max_steps", "4"]
    res_file = tmp_path / "eval.json"
    out = subprocess.run([sys.executable, "scripts/step4_eval.py", "--model_dir", str(model_dir / "snapshots" /
                          "itr_4"), "--out", str(res_file)] + eval_args, cwd=ROOT, env=env, capture_output=True,
                         text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    res = json.load(open(res_file))
    assert len(res["sections"]) == 2 * 2 * 2 * 2 and res["config"]["model_itr"] == 4
    for key, r in res["sections"].items():
        assert r["found_but_replay_failed"] == 0
        for cube in r["cubes"]:
            assert cube["solved"] == cube["found"] and cube["nodes"] > 0 and cube["time"] > 0
    assert "Gate: no native 4x4 depth >= 6" in out.stdout

    out_dir = tmp_path / "snap_eval"
    out = subprocess.run([sys.executable, "scripts/step4_eval.py", "--snapshots_dir", str(model_dir / "snapshots"),
                          "--out_dir", str(out_dir)] + eval_args, cwd=ROOT, env=env, capture_output=True, text=True,
                         timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    assert sorted(os.listdir(out_dir)) == ["baseline.json", "itr_2.json", "itr_4.json", "solve_rate_vs_itr.png",
                                           "summary.json"]
    summary = json.load(open(out_dir / "summary.json"))
    assert summary["itrs"] == [2, 4] and len(summary["J4"]["native4/beam/d2"]) == 2
