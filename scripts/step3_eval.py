""" Step 3 evaluation: native 4x4 vs embedded 3x3 vs embedded 2x2, all searched with the 4x4 network J4.

For each solver and depth, fixed-seed scrambles are solved with weighted A*, every path is replayed in the native
environment of the scrambled cube, and solve rate, path length, nodes generated and time per cube are written to a
JSON file and printed as a table. A zero-heuristic baseline runs on the shallow depths.

Results are saved after every (solver, heuristic, depth) section. Re-running with the same config and checkpoint
skips finished sections; if the config or the checkpoint's training iteration changed, everything is recomputed.

Example:
    python scripts/step3_eval.py --model_dir saved_models/cube4/current --out results/step3/eval.json
"""
import argparse
import json
import os
import pickle
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.cube4 import Cube4State  # noqa: E402
from search_methods import transfer_search as ts  # noqa: E402
from utils import env_utils, nnet_utils  # noqa: E402

SOLVERS = ["native4", "embedded3", "embedded2"]
SOLVER_ENV = {"native4": "cube4", "embedded3": "cube3", "embedded2": "cube2"}
AXIS = {'U': 0, 'D': 0, 'L': 1, 'R': 1, 'B': 2, 'F': 2}


def scramble(env, depth: int, num: int, seed: int) -> List[List[int]]:
    """ Fixed-seed scrambles of exactly `depth` moves. Consecutive moves never share an axis, so a move is never
    followed by its inverse and (on the 2x2) never by the opposite face turned the same way, which would be a
    whole-cube rotation. """
    rng = np.random.RandomState(seed)
    axes = [AXIS[m[0].upper()] for m in env.moves]
    seqs: List[List[int]] = []
    for _ in range(num):
        seq: List[int] = []
        while len(seq) < depth:
            m = int(rng.randint(env.get_num_moves()))
            if seq and axes[m] == axes[seq[-1]]:
                continue
            seq.append(m)
        seqs.append(seq)
    return seqs


def apply_seq(env, seq: List[int]):
    state = env.generate_goal_states(1)[0]
    for m in seq:
        state = env.next_state([state], m)[0][0]
    return state


def run_section(solver: str, depth: int, heuristic_fn, args) -> Dict:
    env = env_utils.get_environment(SOLVER_ENV[solver])
    seed = args.seed + 1000 * SOLVERS.index(solver) + depth
    seqs = scramble(env, depth, args.num_per_depth, seed)
    states = [apply_seq(env, s) for s in seqs]

    if solver == "native4":
        res = ts.solve_native4(states, heuristic_fn, args.weight, args.batch_size, args.max_nodes)
    else:
        res = ts.solve_embedded(states, SOLVER_ENV[solver], heuristic_fn, args.weight, args.batch_size,
                                args.max_nodes)

    solved = [r for r in res if r["solved"]]
    summary = {
        "n": len(res),
        "solved": len(solved),
        "found_but_replay_failed": sum(r["found"] and not r["solved"] for r in res),
        "solve_rate": len(solved) / len(res),
        "mean_len": float(np.mean([len(r["moves"]) for r in solved])) if solved else None,
        "mean_nodes": float(np.mean([r["nodes"] for r in res])),
        "mean_time": float(np.mean([r["time"] for r in res])),
        "cubes": [dict(r, scramble=s) for r, s in zip(res, seqs)],
    }
    return summary


def load_heuristic(model_dir: str, nnet_batch_size: int):
    device, devices, on_gpu = nnet_utils.get_device()
    print("device: %s, devices: %s, on_gpu: %s" % (device, devices, on_gpu))
    env = env_utils.get_environment("cube4")
    return nnet_utils.load_heuristic_fn(model_dir, device, on_gpu, env.get_nnet_model(), env, clip_zero=True,
                                        batch_size=nnet_batch_size)


def print_table(results: Dict) -> None:
    header = "%-10s %-5s %5s %8s %6s %8s %10s %9s" % ("solver", "heur", "depth", "solved", "rate", "mean_len",
                                                       "mean_nodes", "time/cube")
    print(header)
    print("-" * len(header))
    for key in sorted(results["sections"], key=lambda k: (k.split("/")[1], SOLVERS.index(k.split("/")[0]),
                                                          int(k.split("/")[2][1:]))):
        solver, heur, depth = key.split("/")
        r = results["sections"][key]
        mean_len = "-" if r["mean_len"] is None else "%.2f" % r["mean_len"]
        print("%-10s %-5s %5s %4i/%-3i %6.2f %8s %10.0f %8.2fs" % (solver, heur, depth[1:], r["solved"], r["n"],
                                                                   r["solve_rate"], mean_len, r["mean_nodes"],
                                                                   r["mean_time"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help="Checkpoint dir, e.g. <save_dir>/cube4/current")
    parser.add_argument('--out', type=str, required=True, help="JSON results file")
    parser.add_argument('--depths', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    parser.add_argument('--baseline_depths', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--num_per_depth', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weight', type=float, default=0.8, help="Weight on path cost (cost = weight * g + h)")
    parser.add_argument('--batch_size', type=int, default=100, help="Nodes popped per cube per A* step")
    parser.add_argument('--max_nodes', type=int, default=200000, help="Per-cube budget of generated nodes")
    parser.add_argument('--nnet_batch_size', type=int, default=10000)
    parser.add_argument('--solvers', type=str, nargs='+', default=SOLVERS, choices=SOLVERS)
    args = parser.parse_args()

    itr_file = os.path.join(args.model_dir, "train_itr.pkl")
    model_itr = float(pickle.load(open(itr_file, "rb"))) if os.path.isfile(itr_file) else None
    config = {k: v for k, v in vars(args).items() if k not in ("out", "nnet_batch_size")}
    config["model_itr"] = model_itr

    results = {"config": config, "sections": {}}
    if os.path.isfile(args.out):
        previous = json.load(open(args.out))
        if previous.get("config") == config:
            results = previous
            print("Resuming: %i finished sections in %s" % (len(results["sections"]), args.out))
        else:
            print("Config or checkpoint changed since %s was written: recomputing all sections" % args.out)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    j4 = None
    plan = [(s, "J4", d) for s in args.solvers for d in args.depths] + \
           [(s, "zero", d) for s in args.solvers for d in args.baseline_depths]
    for solver, heur, depth in plan:
        key = "%s/%s/d%i" % (solver, heur, depth)
        if key in results["sections"]:
            continue
        if heur == "J4" and j4 is None:
            j4 = load_heuristic(args.model_dir, args.nnet_batch_size)
        heuristic_fn = j4 if heur == "J4" else ts.zero_heuristic
        start = time.time()
        results["sections"][key] = run_section(solver, depth, heuristic_fn, args)
        r = results["sections"][key]
        print("%s: solved %i/%i (%.1fs)" % (key, r["solved"], r["n"], time.time() - start))
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=1)
        os.replace(tmp, args.out)

    print("")
    print_table(results)


if __name__ == "__main__":
    main()
