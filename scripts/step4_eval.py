""" Step 4 evaluation: weighted A* and beam search with J4 vs a zero-heuristic baseline, on native 4x4 and (optional)
embedded 3x3 / 2x2 scrambles.

* Scrambles: fixed seed, exactly d moves, no two consecutive moves on the same axis (Step 3's rule). The same
  scrambles are used for every solver and heuristic of a target (seed = --seed + 1000 * target index + depth).
* Every cube is searched on its own, so nodes and time are per cube. Each cube has the same node budget
  (--max_nodes) for both solvers; a cube that exceeds it counts as a failure. Every path is replayed in the native
  environment of the cube; only a replay that ends solved counts.
* Baseline: zero heuristic, same budget, all depths, both solvers. (For beam search a zero heuristic keeps the first
  beam_width unseen successors in move order.)
* Gate (native 4x4 only, depths >= 6): J4 is "clearly above" the baseline at a depth if its solve rate is at least
  GATE_POINTS percentage points higher than the best baseline solver at that depth AND the two Wilson 95% intervals
  do not overlap. The gate passes if one J4 solver is clearly above at every evaluated depth >= 6.

Results are written after every section and resumed if the config (including the checkpoint iteration) matches.

One checkpoint or snapshot:
    python scripts/step4_eval.py --model_dir <save_dir>/cube4/snapshots/itr_N --out results/step4/eval.json
All snapshots (one JSON per snapshot + one baseline JSON, then solve rate vs training iteration):
    python scripts/step4_eval.py --snapshots_dir <save_dir>/cube4/snapshots --out_dir results/step4/snapshots
"""
import argparse
import json
import math
import os
import pickle
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.step3_eval import apply_seq, load_heuristic, scramble  # noqa: E402
from search_methods import beam_search as bs  # noqa: E402
from search_methods import transfer_search as ts  # noqa: E402
from utils import env_utils  # noqa: E402

TARGETS = ["native4", "embedded3", "embedded2"]
TARGET_ENV = {"native4": "cube4", "embedded3": "cube3", "embedded2": "cube2"}
SOLVERS = ["astar", "beam"]
DEFAULT_DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
GATE_MIN_DEPTH = 6
GATE_POINTS = 20.0


# ---------------------------------------------------------------- statistics

def wilson(k: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """ Wilson score 95% interval for a proportion k/n. """
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def clearly_above(k1: int, n1: int, k0: int, n0: int, points: float = GATE_POINTS) -> bool:
    """ Rate k1/n1 is >= `points` percentage points above k0/n0 and the Wilson 95% intervals do not overlap. """
    if n1 == 0 or n0 == 0:
        return False
    lo1, _ = wilson(k1, n1)
    _, hi0 = wilson(k0, n0)
    return 100.0 * (k1 / n1 - k0 / n0) >= points and lo1 > hi0


def gate_check(j4_sections: Dict, base_sections: Dict, points: float = GATE_POINTS,
               min_depth: int = GATE_MIN_DEPTH, verbose: bool = True) -> Optional[bool]:
    """ Native 4x4 gate. Returns None if no native4 depth >= min_depth was evaluated with J4 and the baseline. """
    depths = sorted({int(k.split("/")[3][1:]) for k in j4_sections if k.startswith("native4/")})
    depths = [d for d in depths if d >= min_depth]
    rows = []
    for d in depths:
        base = [base_sections.get("native4/%s/zero/d%i" % (s, d)) for s in SOLVERS]
        base = [b for b in base if b is not None]
        j4 = {s: j4_sections.get("native4/%s/J4/d%i" % (s, d)) for s in SOLVERS}
        if not base or not any(j4.values()):
            continue
        b = max(base, key=lambda r: r["solved"] / r["n"])
        rows.append((d, b, j4))
    if not rows:
        if verbose:
            print("Gate: no native 4x4 depth >= %i evaluated with both J4 and the baseline" % min_depth)
        return None

    def fmt(r):
        lo, hi = wilson(r["solved"], r["n"])
        return "%5.1f%% [%4.1f,%5.1f]" % (100 * r["solved"] / r["n"], 100 * lo, 100 * hi)

    passes = {s: True for s in SOLVERS}
    if verbose:
        print("Gate (native 4x4, depths >= %i): J4 clearly above the best zero-heuristic solver = at least %g points "
              "higher AND non-overlapping Wilson 95%% intervals" % (min_depth, points))
        print("%5s  %-22s %-22s %-22s %s" % ("depth", "baseline (best)", "J4 A*", "J4 beam", "clearly above"))
    for d, b, j4 in rows:
        marks = []
        for s in SOLVERS:
            ok = j4[s] is not None and clearly_above(j4[s]["solved"], j4[s]["n"], b["solved"], b["n"], points)
            passes[s] = passes[s] and ok
            marks.append("%s:%s" % (s, "yes" if ok else "no"))
        if verbose:
            print("%5i  %-22s %-22s %-22s %s" % (d, fmt(b), fmt(j4["astar"]) if j4["astar"] else "-",
                                                 fmt(j4["beam"]) if j4["beam"] else "-", " ".join(marks)))
    result = any(passes.values())
    if verbose:
        print("GATE %s: %s" % ("PASS" if result else "FAIL",
                                ", ".join("%s %s" % (s, "passes" if p else "fails") for s, p in passes.items())))
    return result


# ---------------------------------------------------------------- running sections

def solve_one(target: str, solver: str, state, heuristic_fn, args) -> Dict:
    """ Search one cube alone, so nodes and time are this cube's. """
    if solver == "astar":
        if target == "native4":
            return ts.solve_native4([state], heuristic_fn, args.weight, args.batch_size, args.max_nodes)[0]
        return ts.solve_embedded([state], TARGET_ENV[target], heuristic_fn, args.weight, args.batch_size,
                                 args.max_nodes)[0]
    if target == "native4":
        return bs.solve_native4([state], heuristic_fn, args.beam_width, args.beam_max_steps, args.max_nodes)[0]
    return bs.solve_embedded([state], TARGET_ENV[target], heuristic_fn, args.beam_width, args.beam_max_steps,
                             args.max_nodes)[0]


def run_section(target: str, solver: str, depth: int, heuristic_fn, args) -> Dict:
    env = env_utils.get_environment(TARGET_ENV[target])
    seqs = scramble(env, depth, args.num_per_depth, args.seed + 1000 * TARGETS.index(target) + depth)
    res = []
    for seq in seqs:
        state = apply_seq(env, seq)
        start = time.perf_counter()
        r = solve_one(target, solver, state, heuristic_fn, args)
        r["time"] = time.perf_counter() - start  # wall clock of this cube's own search
        r["scramble"] = seq
        res.append(r)

    solved = [r for r in res if r["solved"]]
    return {"n": len(res), "solved": len(solved),
            "found_but_replay_failed": sum(r["found"] and not r["solved"] for r in res),
            "solve_rate": len(solved) / len(res),
            "mean_len": float(np.mean([len(r["moves"]) for r in solved])) if solved else None,
            "mean_nodes": float(np.mean([r["nodes"] for r in res])),
            "mean_time": float(np.mean([r["time"] for r in res])),
            "mean_nodes_solved": float(np.mean([r["nodes"] for r in solved])) if solved else None,
            "cubes": res}


def section_config(args, model_itr: Optional[float]) -> Dict:
    keys = ["targets", "solvers", "depths", "num_per_depth", "seed", "weight", "batch_size", "beam_width",
            "beam_max_steps", "max_nodes"]
    config = {k: getattr(args, k) for k in keys}
    config["model_itr"] = model_itr
    return config


def run_file(out: str, plan: List[Tuple[str, str, str, int]], config: Dict, model_dir: Optional[str], args) -> Dict:
    """ Run (target, solver, heur, depth) sections into `out`, resuming if its config matches. """
    results = {"config": config, "sections": {}}
    if os.path.isfile(out):
        previous = json.load(open(out))
        if previous.get("config") == config:
            results = previous
            print("Resuming: %i finished sections in %s" % (len(results["sections"]), out))
        else:
            print("Config or checkpoint changed since %s was written: recomputing all sections" % out)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    j4 = None
    for target, solver, heur, depth in plan:
        key = "%s/%s/%s/d%i" % (target, solver, heur, depth)
        if key in results["sections"]:
            continue
        if heur == "J4" and j4 is None:
            j4 = load_heuristic(model_dir, args.nnet_batch_size)
        start = time.time()
        r = run_section(target, solver, depth, j4 if heur == "J4" else ts.zero_heuristic, args)
        results["sections"][key] = r
        print("%s: solved %i/%i (%.1fs)" % (key, r["solved"], r["n"], time.time() - start), flush=True)
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(results, f, indent=1)
        os.replace(tmp, out)
    return results


def make_plan(args, heurs: List[str], depths: List[int]) -> List[Tuple[str, str, str, int]]:
    return [(t, s, h, d) for h in heurs for t in args.targets for s in args.solvers for d in depths]


def model_itr_of(model_dir: str) -> Optional[float]:
    f = os.path.join(model_dir, "train_itr.pkl")
    return float(pickle.load(open(f, "rb"))) if os.path.isfile(f) else None


# ---------------------------------------------------------------- output

def print_table(sections: Dict) -> None:
    header = "%-10s %-6s %-5s %5s %8s %6s %13s %8s %10s %9s" % (
        "target", "solver", "heur", "depth", "solved", "rate", "95% CI", "mean_len", "nodes/cube", "time/cube")
    print(header)
    print("-" * len(header))

    def order(k):
        t, s, h, d = k.split("/")
        return h != "J4", TARGETS.index(t), SOLVERS.index(s), int(d[1:])

    for key in sorted(sections, key=order):
        t, s, h, d = key.split("/")
        r = sections[key]
        lo, hi = wilson(r["solved"], r["n"])
        mean_len = "-" if r["mean_len"] is None else "%.2f" % r["mean_len"]
        print("%-10s %-6s %-5s %5s %4i/%-3i %6.2f %13s %8s %10.0f %8.2fs" % (
            t, s, h, d[1:], r["solved"], r["n"], r["solve_rate"], "[%.2f,%.2f]" % (lo, hi), mean_len,
            r["mean_nodes"], r["mean_time"]))


def snapshot_dirs(snapshots_dir: str) -> List[Tuple[int, str]]:
    out = []
    for name in os.listdir(snapshots_dir):
        m = re.fullmatch(r"itr_(\d+)", name)
        if m and os.path.isfile(os.path.join(snapshots_dir, name, "model_state_dict.pt")):
            out.append((int(m.group(1)), os.path.join(snapshots_dir, name)))
    return sorted(out)


def plot_vs_itr(summary: Dict, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, surface = "#0b0b0b", "#fcfcfb"
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # ordinal blue, light -> dark
    panels = sorted({(k.split("/")[0], k.split("/")[1]) for k in summary["J4"]},
                    key=lambda p: (TARGETS.index(p[0]), SOLVERS.index(p[1])))
    depths = sorted({int(k.split("/")[2][1:]) for k in summary["J4"]})
    shown = [d for d in depths if d >= GATE_MIN_DEPTH] or depths
    shown = shown[-len(ramp):]
    colors = {d: ramp[int(round(i * (len(ramp) - 1) / max(1, len(shown) - 1)))] for i, d in enumerate(shown)}

    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.8), sharey=True, facecolor=surface,
                             squeeze=False)
    for ax, (t, s) in zip(axes[0], panels):
        ax.set_facecolor(surface)
        for d in shown:
            pts = summary["J4"].get("%s/%s/d%i" % (t, s, d))
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], color=colors[d], linewidth=2, marker="o",
                    markersize=4, label="depth %i" % d)
            base = summary["zero"].get("%s/%s/d%i" % (t, s, d))
            if base is not None:
                ax.axhline(100 * base, color=colors[d], linewidth=1, linestyle=":")
        ax.set_title("%s, %s" % (t, "A*" if s == "astar" else "beam"), color=ink, loc="left")
        ax.set_xlabel("training iteration", color=ink)
        ax.set_ylim(-3, 103)
        ax.grid(True, color="#e4e3df", linewidth=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0][0].set_ylabel("solve rate (%)  -  dotted: zero heuristic", color=ink)
    axes[0][-1].legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_snapshots(args) -> None:
    snaps = snapshot_dirs(args.snapshots_dir)
    if not snaps:
        raise SystemExit("No snapshots (itr_<N>/model_state_dict.pt) in %s" % args.snapshots_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    base = {"sections": {}}
    if not args.no_baseline:
        base = run_file(os.path.join(args.out_dir, "baseline.json"), make_plan(args, ["zero"], args.depths),
                        section_config(args, None), None, args)
    summary = {"itrs": [i for i, _ in snaps], "J4": {}, "zero": {}}
    last = None
    for itr, d in snaps:
        print("== snapshot itr %i ==" % itr)
        last = run_file(os.path.join(args.out_dir, "itr_%i.json" % itr), make_plan(args, ["J4"], args.depths),
                        section_config(args, model_itr_of(d)), d, args)
        for key, r in last["sections"].items():
            t, s, _, dep = key.split("/")
            summary["J4"].setdefault("%s/%s/%s" % (t, s, dep), []).append([itr, r["solve_rate"], r["solved"], r["n"]])
    for key, r in base["sections"].items():
        t, s, _, dep = key.split("/")
        summary["zero"]["%s/%s/%s" % (t, s, dep)] = r["solve_rate"]
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    plot_vs_itr(summary, os.path.join(args.out_dir, "solve_rate_vs_itr.png"))
    print("wrote %s and %s" % (os.path.join(args.out_dir, "summary.json"),
                               os.path.join(args.out_dir, "solve_rate_vs_itr.png")))
    print("\nLatest snapshot (itr %i):" % snaps[-1][0])
    print_table(dict(last["sections"], **base["sections"]))
    print("")
    gate_check(last["sections"], base["sections"])


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--model_dir', type=str, help="One checkpoint dir: .../cube4/current or .../snapshots/itr_N")
    src.add_argument('--snapshots_dir', type=str, help="Evaluate every itr_N snapshot in this dir")
    parser.add_argument('--out', type=str, help="JSON results file (with --model_dir)")
    parser.add_argument('--out_dir', type=str, help="Output dir (with --snapshots_dir)")
    parser.add_argument('--targets', type=str, nargs='+', default=TARGETS, choices=TARGETS)
    parser.add_argument('--solvers', type=str, nargs='+', default=SOLVERS, choices=SOLVERS)
    parser.add_argument('--depths', type=int, nargs='+', default=DEFAULT_DEPTHS)
    parser.add_argument('--num_per_depth', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_nodes', type=int, default=200000, help="Per-cube node budget, both solvers")
    parser.add_argument('--weight', type=float, default=0.8, help="A*: cost = weight * g + h")
    parser.add_argument('--batch_size', type=int, default=100, help="A*: nodes popped per step")
    parser.add_argument('--beam_width', type=int, default=256)
    parser.add_argument('--beam_max_steps', type=int, default=50)
    parser.add_argument('--nnet_batch_size', type=int, default=10000)
    parser.add_argument('--no_baseline', action='store_true', default=False)
    args = parser.parse_args(argv)

    if args.snapshots_dir:
        if not args.out_dir:
            parser.error("--snapshots_dir needs --out_dir")
        run_snapshots(args)
        return
    if not args.out:
        parser.error("--model_dir needs --out")

    heurs = ["J4"] if args.no_baseline else ["J4", "zero"]
    results = run_file(args.out, make_plan(args, heurs, args.depths), section_config(args, model_itr_of(args.model_dir)),
                       args.model_dir, args)
    print("")
    print_table(results["sections"])
    print("")
    sections = results["sections"]
    gate_check({k: v for k, v in sections.items() if "/J4/" in k}, {k: v for k, v in sections.items() if "/zero/" in k})


if __name__ == "__main__":
    main()
