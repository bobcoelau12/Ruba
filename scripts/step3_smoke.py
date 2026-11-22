""" CPU smoke test of the whole Step 3 pipeline, with tiny settings (minutes, not results):

  train cube4 (AVI, a few iterations) -> checkpoint -> resume from it -> load J4 ->
  native 4x4 solve -> embedded 3x3 solve -> embedded 2x2 solve -> replay every path -> eval re-run skips work.

    python scripts/step3_smoke.py [--work_dir DIR]
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils import env_utils, nnet_utils  # noqa: E402
from utils.cube_mapping import embed_2to4, embed_3to4  # noqa: E402
from environments.cube4 import Cube4State  # noqa: E402

TINY_AVI = ["--env", "cube4", "--back_max", "10", "--nnet_name", "cube4", "--batch_size", "50",
            "--states_per_update", "100", "--num_test", "20", "--update_nnet_batch_size", "200", "--debug"]


def cpu_env():
    env = dict(os.environ, PYTHONPATH=ROOT, CUDA_VISIBLE_DEVICES="", PYTHONWARNINGS="ignore")
    return env


def run(cmd):
    print("$ " + " ".join(cmd), flush=True)
    out = subprocess.run(cmd, cwd=ROOT, env=cpu_env(), capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stdout[-3000:], out.stderr[-3000:])
        raise RuntimeError("command failed: %s" % " ".join(cmd))
    return out.stdout


def train(save_dir: str, max_itrs: int) -> float:
    run([sys.executable, "ctg_approx/avi.py", *TINY_AVI, "--save_dir", save_dir, "--max_itrs", str(max_itrs)])
    return float(pickle.load(open(os.path.join(save_dir, "cube4", "current", "train_itr.pkl"), "rb")))


def heuristic_on_embedded(model_dir: str) -> np.ndarray:
    device, _, on_gpu = nnet_utils.get_device()
    env4 = env_utils.get_environment("cube4")
    heur = nnet_utils.load_heuristic_fn(model_dir, device, on_gpu, env4.get_nnet_model(), env4, clip_zero=True)
    np.random.seed(0)
    s3, _ = env_utils.get_environment("cube3").generate_states(5, (0, 5))
    s2, _ = env_utils.get_environment("cube2").generate_states(5, (0, 5))
    states4 = [Cube4State(x) for x in embed_3to4(np.stack([s.colors for s in s3]))] + \
              [Cube4State(x) for x in embed_2to4(np.stack([s.colors for s in s2]))]
    return heur(states4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    work = args.work_dir or tempfile.mkdtemp(prefix="step3_smoke_")
    save_dir = os.path.join(work, "saved_models")
    model_dir = os.path.join(save_dir, "cube4", "current")
    out = os.path.join(work, "eval.json")

    itr1 = train(save_dir, 2)
    itr2 = train(save_dir, 4)
    assert itr1 == 2 and itr2 == 4, (itr1, itr2)
    print("train -> checkpoint -> resume: itr %g then %g" % (itr1, itr2))

    h = heuristic_on_embedded(model_dir)
    assert np.all(np.isfinite(h)), h
    print("J4 on embedded 3x3 / 2x2 states: finite, mean %.3f" % h.mean())

    eval_cmd = [sys.executable, "scripts/step3_eval.py", "--model_dir", model_dir, "--out", out,
                "--depths", "1", "2", "--baseline_depths", "1", "--num_per_depth", "3", "--max_nodes", "20000"]
    print(run(eval_cmd))
    second = run(eval_cmd)
    assert "Resuming: 9 finished sections" in second, second
    print("eval re-run skipped all 9 finished sections")

    results = json.load(open(out))
    replay_failures = sum(r["found_but_replay_failed"] for r in results["sections"].values())
    assert replay_failures == 0
    for key, r in sorted(results["sections"].items()):
        for cube in r["cubes"]:
            assert cube["solved"] == cube["found"]
    print("all %i sections: every found path replayed to solved in its native env" % len(results["sections"]))
    print("work dir: %s" % work)


if __name__ == "__main__":
    main()
