""" Experiment E1: zero-shot transfer of the 4x4 cost-to-go network J4 to 3x3 and 2x2 through the mapping (M1, M3).

For every 4x4 checkpoint, embedded 3x3 and 2x2 scrambles are solved by depth with the Step 3 solver (batched
weighted A* over the 4x4 embedding, mapped moves only, search_methods/transfer_search.py). Two baselines run under
the same per-cube node budget:

* bfs:    the same A* with h = 0. With unit move costs this is uniform-cost search, i.e. breadth-first search.
* random: a uniform random policy over the mapped moves. Random walks of --random_max_len moves start from the
          scrambled cube, --random_walkers at a time, and restart until the node budget is spent. Every move of every
          walker counts as one generated node; the first walk that reaches a solved cube is the solution.

Optionally (--solvers astar beam) beam search with J4 runs too, under the same budget.

Heuristic scale. One 2x2 move (an outer + inner macro) is two 4x4 moves, so J4 is about twice the 2x2 distance,
while a 3x3 move is one 4x4 move. Weighted A* (cost = weight * g + scale * J4) therefore uses --h_scale_2x2 0.5 and
--h_scale_3x3 1.0 by default. Greedy and beam search only rank states, so the scale does not change them and they
use J4 unscaled.

Everything else follows Step 4 (scripts/step4_eval.py): fixed-seed scrambles of exactly d moves with no two
consecutive moves on the same axis, with the same seeds as Step 4 (so the 3x3 / 2x2 scrambles are identical to the
Step 4 evals); every cube is searched on its own, so nodes and time are per cube; every returned path is replayed in
the native Cube3 / Cube2 and only a replay that ends solved counts; a cube that spends the budget is a failure.

Outputs (in --out_dir, every file written after each section and resumed if its config matches):
    baselines.json        bfs + random, all targets and depths (checkpoint independent, run once)
    itr_<N>.json          J4 sections of checkpoint itr_<N>
    summary.json          one row per (checkpoint or baseline, target, solver, depth): solve rate, Wilson 95% CI,
                          nodes (mean, median, mean over solved), solution length (mean over solved, mean - depth)
    e1_by_depth.png       solve rate, nodes and length vs depth; 3x3 | 2x2; one line per checkpoint + baselines
    e1_vs_checkpoint.png  solve rate and nodes vs 4x4 training iteration, per depth; 3x3 | 2x2
    e1_3x3_vs_2x2.png     latest checkpoint: 3x3 against 2x2 on solve rate, nodes and length vs depth

Example:
    python scripts/e1_transfer.py --snapshots_dir <save_dir>/cube4/snapshots --checkpoints 16000 54600 109200 163800 \
        --out_dir results/e1
"""
import argparse
import json
import math
import os
import re
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.cube4 import Cube4, Cube4State  # noqa: E402
from scripts.step3_eval import apply_seq, load_heuristic, scramble  # noqa: E402
from scripts.step4_eval import TARGETS as STEP4_TARGETS, model_itr_of, snapshot_dirs, wilson  # noqa: E402
from search_methods import beam_search as bs  # noqa: E402
from search_methods import transfer_search as ts  # noqa: E402
from utils import env_utils  # noqa: E402

TARGETS = ["embedded3", "embedded2", "native4"]  # native4 is an optional reference, not part of E1 by default
TARGET_ENV = {"native4": "cube4", "embedded3": "cube3", "embedded2": "cube2"}
J4_SOLVERS = ["astar", "beam"]
BASELINES = ["bfs", "random"]
DEFAULT_DEPTHS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15]


# ---------------------------------------------------------------- solvers

def h_scale(target: str, args) -> float:
    return {"embedded3": args.h_scale_3x3, "embedded2": args.h_scale_2x2, "native4": 1.0}[target]


def scaled(heuristic_fn: Callable, scale: float) -> Callable:
    if scale == 1.0:
        return heuristic_fn
    return lambda states: np.asarray(heuristic_fn(states), dtype=float) * scale


def solved_np(states_np: np.ndarray) -> np.ndarray:
    """ Cube4.is_solved on a (num, 96) colour array: every face shows one colour. """
    faces = (states_np // 16).reshape(-1, 6, 16)
    return np.all(faces == faces[:, :, :1], axis=(1, 2))


def random_policy(state4: Cube4State, env: Cube4, max_nodes: int, max_len: int, walkers: int,
                  rng: np.random.RandomState) -> Dict:
    """ Uniform random walks over env's moves from state4 (restart every max_len moves) until a walk reaches a solved
    cube or max_nodes moves have been made in total. Moves are env move indices. """
    if solved_np(state4.colors[None])[0]:
        return {"found": True, "moves": [], "nodes": 0}
    num_moves = env.get_num_moves()
    nodes = 0
    while nodes < max_nodes:
        width = min(walkers, max_nodes - nodes)
        states = np.repeat(state4.colors[None], width, axis=0)
        acts = rng.randint(num_moves, size=(width, max_len))
        for t in range(max_len):
            width = min(width, max_nodes - nodes)
            if width <= 0:
                break
            states = states[:width]
            for a in range(num_moves):
                idx = np.flatnonzero(acts[:width, t] == a)
                if idx.size:
                    states[idx] = env._move_np(states[idx], a)[0]
            hit = np.flatnonzero(solved_np(states))
            if hit.size:
                i = int(hit[0])
                return {"found": True, "moves": [int(m) for m in acts[i, :t + 1]], "nodes": nodes + i + 1}
            nodes += width
    return {"found": False, "moves": [], "nodes": int(nodes)}


def solve_random(target: str, state, args, rng: np.random.RandomState) -> Dict:
    if target == "native4":
        env, small_env, state4 = Cube4(), Cube4(), state
    else:
        embed, move_map = ts.SMALL_ENVS[TARGET_ENV[target]]
        env, small_env = ts.MappedCube4(move_map), env_utils.get_environment(TARGET_ENV[target])
        state4 = Cube4State(embed(state.colors[None])[0])
    res = random_policy(state4, env, args.max_nodes, args.random_max_len, args.random_walkers, rng)
    res["moves"] = ts.translate_path(res["moves"])
    res["solved"] = bool(res["found"] and ts.replay_solves(small_env, state, res["moves"]))
    return res


def solve_one(target: str, solver: str, state, heuristic_fn: Optional[Callable], args,
              rng: np.random.RandomState) -> Dict:
    """ Search one cube alone, so nodes and time are this cube's. """
    if solver == "random":
        return solve_random(target, state, args, rng)
    if solver in ("astar", "bfs"):
        h = ts.zero_heuristic if solver == "bfs" else scaled(heuristic_fn, h_scale(target, args))
        if target == "native4":
            return ts.solve_native4([state], h, args.weight, args.batch_size, args.max_nodes)[0]
        return ts.solve_embedded([state], TARGET_ENV[target], h, args.weight, args.batch_size, args.max_nodes)[0]
    if solver == "beam":
        if target == "native4":
            return bs.solve_native4([state], heuristic_fn, args.beam_width, args.beam_max_steps, args.max_nodes)[0]
        return bs.solve_embedded([state], TARGET_ENV[target], heuristic_fn, args.beam_width, args.beam_max_steps,
                                 args.max_nodes)[0]
    raise ValueError(solver)


def scramble_seed(target: str, depth: int, seed: int) -> int:
    """ Step 4's seed rule, so the scrambles match the Step 4 evals. """
    return seed + 1000 * STEP4_TARGETS.index(target) + depth


def summarize(res: List[Dict], depth: int, max_nodes: int) -> Dict:
    solved = [r for r in res if r["solved"]]
    nodes = [r["nodes"] for r in res]
    mean_len = float(np.mean([len(r["moves"]) for r in solved])) if solved else None
    return {"n": len(res), "solved": len(solved),
            "found_but_replay_failed": sum(r["found"] and not r["solved"] for r in res),
            "solve_rate": len(solved) / len(res),
            "mean_len": mean_len, "mean_excess_len": None if mean_len is None else mean_len - depth,
            "mean_nodes": float(np.mean(nodes)), "median_nodes": float(np.median(nodes)),
            "mean_nodes_solved": float(np.mean([r["nodes"] for r in solved])) if solved else None,
            "budget_hits": sum((not r["found"]) and r["nodes"] >= max_nodes for r in res),
            "mean_time": float(np.mean([r["time"] for r in res])),
            "cubes": res}


def run_section(target: str, solver: str, depth: int, heuristic_fn: Optional[Callable], args) -> Dict:
    env = env_utils.get_environment(TARGET_ENV[target])
    seqs = scramble(env, depth, args.num_per_depth, scramble_seed(target, depth, args.seed))
    rng = np.random.RandomState(args.seed + 7919 * depth + 104729 * TARGETS.index(target))
    res = []
    for seq in seqs:
        state = apply_seq(env, seq)
        start = time.perf_counter()
        r = solve_one(target, solver, state, heuristic_fn, args, rng)
        r["time"] = time.perf_counter() - start
        r["scramble"] = seq
        res.append(r)
    return summarize(res, depth, args.max_nodes)


def skipped_section(n: int, reason: str) -> Dict:
    return {"n": n, "solved": 0, "found_but_replay_failed": 0, "solve_rate": 0.0, "mean_len": None,
            "mean_excess_len": None, "mean_nodes": None, "median_nodes": None, "mean_nodes_solved": None,
            "budget_hits": None, "mean_time": None, "skipped": reason, "cubes": []}


# ---------------------------------------------------------------- files

def base_config(args) -> Dict:
    keys = ["targets", "depths", "num_per_depth", "seed", "max_nodes", "weight", "batch_size"]
    return {k: getattr(args, k) for k in keys}


def j4_config(args, model_itr: Optional[float]) -> Dict:
    config = base_config(args)
    config.update({"solvers": args.solvers, "h_scale_3x3": args.h_scale_3x3, "h_scale_2x2": args.h_scale_2x2,
                   "model_itr": model_itr})
    if "beam" in args.solvers:
        config.update({"beam_width": args.beam_width, "beam_max_steps": args.beam_max_steps})
    return config


def baseline_config(args) -> Dict:
    config = base_config(args)
    config.update({"baselines": args.baselines, "random_max_len": args.random_max_len,
                   "random_walkers": args.random_walkers, "baseline_stop_after_zero": args.baseline_stop_after_zero})
    return config


def load_results(out: str, config: Dict) -> Dict:
    results = {"config": config, "sections": {}}
    if os.path.isfile(out):
        previous = json.load(open(out))
        if previous.get("config") == config:
            results = previous
            print("Resuming: %i finished sections in %s" % (len(results["sections"]), out))
        else:
            print("Config or checkpoint changed since %s was written: recomputing all sections" % out)
    return results


def save_results(out: str, results: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=1)
    os.replace(tmp, out)


def reusable_step4_section(step4_json: Optional[str], target: str, depth: int, model_itr: Optional[float],
                           args) -> Optional[Dict]:
    """ A Step 4 A* J4 section that is exactly what E1 would compute (same scrambles, budget, weight, batch, J4 scale 1
    and checkpoint), or None. Saves re-running the costly embedded 3x3 A* at the deep depths. """
    if step4_json is None or not os.path.isfile(step4_json) or h_scale(target, args) != 1.0:
        return None
    prev = json.load(open(step4_json))
    c = prev.get("config", {})
    same = all(c.get(k) == getattr(args, k) for k in ("num_per_depth", "seed", "max_nodes", "weight", "batch_size"))
    if not same or model_itr is None or c.get("model_itr") != model_itr:
        return None
    sec = prev.get("sections", {}).get("%s/astar/J4/d%i" % (target, depth))
    if sec is None:
        return None
    return dict(summarize(sec["cubes"], depth, args.max_nodes), reused_from=os.path.basename(step4_json))


def run_baselines(args) -> Dict:
    out = os.path.join(args.out_dir, "baselines.json")
    results = load_results(out, baseline_config(args))
    for target in args.targets:
        for solver in args.baselines:
            dead = False  # a shallower depth already solved 0 cubes
            for depth in args.depths:
                key = "%s/%s/d%i" % (target, solver, depth)
                if key in results["sections"]:
                    dead = dead or results["sections"][key]["solved"] == 0
                    continue
                if dead and args.baseline_stop_after_zero:
                    r = skipped_section(args.num_per_depth, "a shallower depth solved 0 cubes under the same budget")
                else:
                    start = time.time()
                    r = run_section(target, solver, depth, None, args)
                    print("baseline %s: solved %i/%i, mean nodes %.0f (%.1fs)" % (
                        key, r["solved"], r["n"], r["mean_nodes"], time.time() - start), flush=True)
                dead = dead or r["solved"] == 0
                results["sections"][key] = r
                save_results(out, results)
    return results


def run_checkpoint(itr: int, model_dir: str, args) -> Dict:
    out = os.path.join(args.out_dir, "itr_%i.json" % itr)
    model_itr = model_itr_of(model_dir)
    results = load_results(out, j4_config(args, model_itr))
    step4_json = os.path.join(args.reuse_step4_dir, "eval_itr_%i.json" % itr) if args.reuse_step4_dir else None
    heuristic_fn = None
    for target in args.targets:
        for solver in args.solvers:
            for depth in args.depths:
                key = "%s/%s/d%i" % (target, solver, depth)
                if key in results["sections"]:
                    continue
                r = reusable_step4_section(step4_json, target, depth, model_itr, args) if solver == "astar" else None
                if r is not None:
                    print("itr_%i %s: reused from %s (solved %i/%i)" % (itr, key, r["reused_from"], r["solved"], r["n"]))
                else:
                    if heuristic_fn is None:
                        heuristic_fn = load_heuristic(model_dir, args.nnet_batch_size)
                    start = time.time()
                    r = run_section(target, solver, depth, heuristic_fn, args)
                    print("itr_%i %s: solved %i/%i, mean nodes %.0f (%.1fs)" % (
                        itr, key, r["solved"], r["n"], r["mean_nodes"], time.time() - start), flush=True)
                results["sections"][key] = r
                save_results(out, results)
    return results


# ---------------------------------------------------------------- summary and plots

def summary_rows(checkpoints: Dict[int, Dict], baselines: Dict) -> List[Dict]:
    rows = []

    def add(source: str, itr: Optional[int], key: str, r: Dict):
        target, solver, d = key.split("/")
        lo, hi = wilson(r["solved"], r["n"])
        row = {"source": source, "itr": itr, "target": target, "solver": solver, "depth": int(d[1:]),
               "ci_low": lo, "ci_high": hi}
        row.update({k: v for k, v in r.items() if k != "cubes"})
        rows.append(row)

    for itr in sorted(checkpoints):
        for key, r in checkpoints[itr]["sections"].items():
            add("itr_%i" % itr, itr, key, r)
    for key, r in baselines.get("sections", {}).items():
        add("baseline", None, key, r)
    return rows


def print_table(rows: List[Dict]) -> None:
    header = "%-11s %-10s %-6s %5s %8s %15s %8s %8s %11s %11s" % (
        "source", "target", "solver", "depth", "solved", "95% CI", "mean_len", "excess", "mean_nodes", "med_nodes")
    print(header)
    print("-" * len(header))
    for r in sorted(rows, key=lambda r: (r["target"], r["solver"], r["itr"] is None, r["itr"] or 0, r["depth"])):
        def f(v, fmt):
            return "-" if v is None else fmt % v
        print("%-11s %-10s %-6s %5i %4i/%-3i %15s %8s %8s %11s %11s" % (
            r["source"], r["target"], r["solver"], r["depth"], r["solved"], r["n"],
            "[%.2f,%.2f]" % (r["ci_low"], r["ci_high"]), f(r["mean_len"], "%.2f"), f(r["mean_excess_len"], "%+.2f"),
            f(r["mean_nodes"], "%.0f"), f(r["median_nodes"], "%.0f")) + ("  (skipped)" if r.get("skipped") else ""))


INK, INK2, SURFACE, GRID = "#0b0b0b", "#5c5c5c", "#fcfcfb", "#e4e3df"
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]  # ordinal blue, light -> dark
TARGET_COLOR = {"embedded3": "#257de0", "embedded2": "#f47230", "native4": "#00b894"}
TARGET_NAME = {"embedded3": "3x3 (embedded)", "embedded2": "2x2 (embedded)", "native4": "4x4 (native)"}
BASE_STYLE = {"bfs": dict(color="#6b6b6b", linestyle="--", marker="s"),
              "random": dict(color="#9c9c9c", linestyle=":", marker="^")}
SOLVER_NAME = {"astar": "J4 A*", "beam": "J4 beam", "bfs": "BFS (h = 0)", "random": "random policy"}


def _style_axis(ax, title=None, xticks=None):
    ax.set_facecolor(SURFACE)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.grid(True, color=GRID, linewidth=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2)
    if title:
        ax.set_title(title, color=INK, loc="left", fontsize=10)


def _ramp(n: int) -> List[str]:
    if n <= 1:
        return [RAMP[-2]]
    return [RAMP[int(round(i * (len(RAMP) - 1) / (n - 1)))] for i in range(n)]


def _series(rows, **match):
    pts = sorted((r["depth"], r) for r in rows if all(r[k] == v for k, v in match.items()) and not r.get("skipped"))
    return [p[0] for p in pts], [p[1] for p in pts]


METRICS = [("solve_rate", "solve rate (%)", lambda r: 100 * r["solve_rate"], False),
           ("mean_nodes", "mean nodes per cube (log)", lambda r: r["mean_nodes"], True),
           ("mean_len", "mean solution length, solved", lambda r: r["mean_len"], False)]


def plot_by_depth(rows: List[Dict], targets: List[str], solver: str, max_nodes: int, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    itrs = sorted({r["itr"] for r in rows if r["itr"] is not None})
    colors = dict(zip(itrs, _ramp(len(itrs))))
    fig, axes = plt.subplots(len(METRICS), len(targets), figsize=(5.2 * len(targets), 3.3 * len(METRICS)),
                             facecolor=SURFACE, squeeze=False)
    for col, target in enumerate(targets):
        for row, (_, label, get, log) in enumerate(METRICS):
            ax = axes[row][col]
            _style_axis(ax, "%s: %s" % (TARGET_NAME[target], label) if row == 0 else label,
                        sorted({r["depth"] for r in rows}))
            for itr in itrs:
                xs, rs = _series(rows, target=target, solver=solver, itr=itr)
                ys = [get(r) for r in rs]
                keep = [(x, y) for x, y in zip(xs, ys) if y is not None]
                if keep:
                    ax.plot(*zip(*keep), color=colors[itr], linewidth=2, marker="o", markersize=4,
                            label="%s itr %i" % (SOLVER_NAME[solver], itr))
            for base, style in BASE_STYLE.items():
                xs, rs = _series(rows, target=target, solver=base, source="baseline")
                keep = [(x, get(r)) for x, r in zip(xs, rs) if get(r) is not None]
                if keep:
                    ax.plot(*zip(*keep), linewidth=1.5, markersize=5, label=SOLVER_NAME[base], **style)
            if row == 0:
                ax.set_ylim(-3, 103)
            if log:
                ax.set_yscale("log")
                ax.axhline(max_nodes, color=INK2, linewidth=0.8)
                ax.annotate("node budget", (ax.get_xlim()[0], max_nodes), color=INK2, fontsize=8,
                            xytext=(2, 3), textcoords="offset points")
            if row == 2:
                lim = max(r["depth"] for r in rows)
                ax.plot([1, lim], [1, lim], color=GRID, linewidth=1, zorder=0)
                ax.annotate("length = depth", (lim, lim), color=INK2, fontsize=8, ha="right", va="bottom")
            if row == len(METRICS) - 1:
                ax.set_xlabel("scramble depth (small-cube moves)", color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_vs_checkpoint(rows: List[Dict], targets: List[str], solver: str, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    j4 = [r for r in rows if r["itr"] is not None and r["solver"] == solver]
    if len({r["itr"] for r in j4}) < 2:
        return
    depths = sorted({r["depth"] for r in j4})
    shown = [d for d in depths if d >= 6] or depths
    shown = shown[-len(RAMP):]
    colors = dict(zip(shown, _ramp(len(shown))))
    fig, axes = plt.subplots(2, len(targets), figsize=(5.2 * len(targets), 6.8), facecolor=SURFACE, squeeze=False)
    for col, target in enumerate(targets):
        for row, (_, label, get, log) in enumerate(METRICS[:2]):
            ax = axes[row][col]
            _style_axis(ax, "%s: %s" % (TARGET_NAME[target], label) if row == 0 else label)
            for d in shown:
                pts = sorted((r["itr"], get(r)) for r in j4 if r["target"] == target and r["depth"] == d)
                if pts:
                    ax.plot(*zip(*pts), color=colors[d], linewidth=2, marker="o", markersize=4, label="depth %i" % d)
            if row == 0:
                ax.set_ylim(-3, 103)
            if log:
                ax.set_yscale("log")
            if row == 1:
                ax.set_xlabel("4x4 training iteration of J4", color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_small_vs_small(rows: List[Dict], targets: List[str], solver: str, itr: int, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.6 * len(METRICS), 3.6), facecolor=SURFACE, squeeze=False)
    for ax, (_, label, get, log) in zip(axes[0], METRICS):
        _style_axis(ax, label, sorted({r["depth"] for r in rows}))
        for target in targets:
            xs, rs = _series(rows, target=target, solver=solver, itr=itr)
            keep = [(x, get(r)) for x, r in zip(xs, rs) if get(r) is not None]
            if keep:
                ax.plot(*zip(*keep), color=TARGET_COLOR[target], linewidth=2, marker="o", markersize=5,
                        label="%s, %s" % (TARGET_NAME[target], SOLVER_NAME[solver]))
            xs, rs = _series(rows, target=target, solver="bfs", source="baseline")
            keep = [(x, get(r)) for x, r in zip(xs, rs) if get(r) is not None]
            if keep:
                ax.plot(*zip(*keep), color=TARGET_COLOR[target], linewidth=1.2, linestyle="--", marker="s",
                        markersize=4, label="%s, BFS (h = 0)" % TARGET_NAME[target])
        if label.startswith("solve"):
            ax.set_ylim(-3, 103)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("scramble depth (small-cube moves)", color=INK)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("3x3 vs 2x2 at itr %i" % itr, color=INK, x=0.01, ha="left")
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_outputs(checkpoints: Dict[int, Dict], baselines: Dict, args) -> List[Dict]:
    rows = summary_rows(checkpoints, baselines)
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"config": {"j4": j4_config(args, None), "baselines": baseline_config(args),
                              "checkpoints": sorted(checkpoints)}, "rows": rows}, f, indent=1)
    solver = "astar" if "astar" in args.solvers else args.solvers[0]
    targets = [t for t in TARGETS if t in args.targets]
    plot_by_depth(rows, targets, solver, args.max_nodes, os.path.join(args.out_dir, "e1_by_depth.png"))
    plot_vs_checkpoint(rows, targets, solver, os.path.join(args.out_dir, "e1_vs_checkpoint.png"))
    small = [t for t in targets if t != "native4"]
    if checkpoints and len(small) == 2:
        plot_small_vs_small(rows, small, solver, max(checkpoints), os.path.join(args.out_dir, "e1_3x3_vs_2x2.png"))
    print("wrote summary.json and plots in %s" % args.out_dir)
    return rows


# ---------------------------------------------------------------- main

def pick_checkpoints(snapshots_dir: str, wanted: Optional[List[str]]) -> List[Tuple[int, str]]:
    snaps = snapshot_dirs(snapshots_dir)
    if not snaps:
        raise SystemExit("No snapshots (itr_<N>/model_state_dict.pt) in %s" % snapshots_dir)
    if not wanted:
        return snaps
    by_itr = dict(snaps)
    out, missing = [], []
    for w in wanted:
        if w == "current":  # the latest training checkpoint, <save_dir>/cube4/current, named by its iteration
            cur = os.path.join(os.path.dirname(os.path.abspath(snapshots_dir)), "current")
            itr = model_itr_of(cur)
            if itr is None or not os.path.isfile(os.path.join(cur, "model_state_dict.pt")):
                raise SystemExit("No checkpoint with train_itr.pkl in %s" % cur)
            if int(itr) not in by_itr:
                out.append((int(itr), cur))
            continue
        m = re.fullmatch(r"(?:itr_)?(\d+)", w)
        if m and int(m.group(1)) in by_itr:
            out.append((int(m.group(1)), by_itr[int(m.group(1))]))
        else:
            missing.append(w)
    if missing:
        raise SystemExit("Checkpoints %s not in %s (available: %s)" % (missing, snapshots_dir,
                                                                       ", ".join("itr_%i" % i for i, _ in snaps)))
    return sorted(set(out))


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument('--snapshots_dir', type=str, help="Dir of itr_<N> checkpoints (Step 4 snapshots)")
    parser.add_argument('--checkpoints', type=str, nargs='*', default=None,
                        help="Checkpoint iterations (e.g. 16000 54600 or itr_16000) and/or 'current' (the latest "
                             "training checkpoint next to the snapshots dir). Default: every snapshot")
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--targets', type=str, nargs='+', default=["embedded3", "embedded2"], choices=TARGETS)
    parser.add_argument('--solvers', type=str, nargs='+', default=["astar"], choices=J4_SOLVERS,
                        help="Solvers with J4 (astar = the Step 3 solver)")
    parser.add_argument('--baselines', type=str, nargs='*', default=BASELINES, choices=BASELINES)
    parser.add_argument('--depths', type=int, nargs='+', default=DEFAULT_DEPTHS)
    parser.add_argument('--num_per_depth', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_nodes', type=int, default=200000, help="Per-cube node budget, every method")
    parser.add_argument('--weight', type=float, default=0.8, help="A*: cost = weight * g + scale * J4")
    parser.add_argument('--batch_size', type=int, default=100, help="A*: nodes popped per step")
    parser.add_argument('--h_scale_3x3', type=float, default=1.0, help="A*: J4 multiplier on embedded 3x3")
    parser.add_argument('--h_scale_2x2', type=float, default=0.5, help="A*: J4 multiplier on embedded 2x2")
    parser.add_argument('--beam_width', type=int, default=256)
    parser.add_argument('--beam_max_steps', type=int, default=50)
    parser.add_argument('--random_max_len', type=int, default=30, help="Random policy: moves per walk")
    parser.add_argument('--random_walkers', type=int, default=1000, help="Random policy: walks run in parallel")
    parser.add_argument('--baseline_stop_after_zero', type=int, default=0,
                        help="1: once a baseline solves 0 cubes at a depth, record deeper depths as 0 without "
                             "running them (they would spend the whole budget). 0: run every depth")
    parser.add_argument('--reuse_step4_dir', type=str, default=None,
                        help="Dir with Step 4 eval_itr_<N>.json files; A* sections with identical settings and J4 "
                             "scale 1 (3x3) are copied instead of re-run")
    parser.add_argument('--nnet_batch_size', type=int, default=10000)
    parser.add_argument('--plots_only', action='store_true', default=False,
                        help="Only rebuild summary.json and plots from the JSON files in --out_dir")
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    checkpoints: Dict[int, Dict] = {}
    if args.plots_only:
        base_file = os.path.join(args.out_dir, "baselines.json")
        baselines = json.load(open(base_file)) if os.path.isfile(base_file) else {"sections": {}}
        for name in os.listdir(args.out_dir):
            m = re.fullmatch(r"itr_(\d+)\.json", name)
            if m:
                checkpoints[int(m.group(1))] = json.load(open(os.path.join(args.out_dir, name)))
        for res in [baselines] + list(checkpoints.values()):  # plot with the settings the results were run with
            for k in ("max_nodes", "targets", "solvers"):
                if k in res.get("config", {}):
                    setattr(args, k, res["config"][k])
    else:
        if not args.snapshots_dir:
            parser.error("--snapshots_dir is required unless --plots_only")
        chosen = pick_checkpoints(args.snapshots_dir, args.checkpoints)
        print("E1 checkpoints: %s" % ", ".join("itr_%i" % i for i, _ in chosen))
        baselines = run_baselines(args) if args.baselines else {"sections": {}}
        for itr, model_dir in chosen:
            print("== checkpoint itr_%i ==" % itr, flush=True)
            checkpoints[itr] = run_checkpoint(itr, model_dir, args)
            write_outputs(checkpoints, baselines, args)  # plots refresh after every checkpoint

    rows = write_outputs(checkpoints, baselines, args)
    print("")
    print_table(rows)
    failed = sum(r["found_but_replay_failed"] for r in rows)
    print("\nreplay failures (found a 4x4 goal but the native replay was not solved): %i" % failed)


if __name__ == "__main__":
    main()
