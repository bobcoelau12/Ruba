""" "Deepest depth reached during training", from the greedy (GBFS) check that avi.py runs after every update cycle.

Definition: for one greedy check, the deepest depth is the largest scramble depth whose %Solved is >= 50%. Depths come
from the "Back Steps: d, %Solved: p, ..." lines in output.txt. The check runs S = min(update_num + 1, back_max) GBFS
steps (printed as "with S steps"), and gbfs_test only records a state as solved when a step starts on it, so the
final move is never checked: only states solved in <= S - 1 moves count. The reported depth is therefore capped at
S - 1 = min(update_num, back_max - 1), i.e. by how often the target network has been updated.

output.txt is appended to by every session, so it can contain several sessions. A check that was cut off (no
"Test time" line, e.g. the runtime died) is ignored; if the same iteration appears twice, the last complete check
wins.

    python scripts/step4_training_depth.py --log <save_dir>/cube4/output.txt --out <dir>
writes training_depth.csv, training_depth.png (deepest depth vs iteration) and training_solved.png (%Solved per depth
vs iteration).
"""
import argparse
import csv
import os
import re
from typing import Dict, List, Optional

SOLVED_THRESHOLD = 50.0

RE_TEST = re.compile(r"^Greedy test at itr (\d+) \(update_num (\d+), back_max (\d+)\)")
RE_TRAIN = re.compile(r"^Training model for update number (\d+) for (\d+) iterations")
RE_ITR = re.compile(r"^Itr: (\d+),")
RE_SOLVING = re.compile(r"^Solving (\d+) states with GBFS with (\d+) steps")
RE_BACK = re.compile(r"^Back Steps: (\d+), %Solved: ([0-9.]+)")
RE_END = re.compile(r"^Test time:")


def parse_log(path: str) -> List[Dict]:
    """ One entry per complete greedy check, sorted by iteration:
    {"itr", "update_num" (None for old logs), "gbfs_steps", "solved": {depth: percent}} """
    checks: Dict[int, Dict] = {}
    pending_itr: Optional[int] = None  # iteration of the next check
    pending_update: Optional[int] = None
    train_itrs: Optional[int] = None  # old logs: iterations of the current training block
    current: Optional[Dict] = None

    with open(path, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            m = RE_TEST.match(line)
            if m:
                pending_itr, pending_update = int(m.group(1)), int(m.group(2))
                current = None
                continue
            m = RE_TRAIN.match(line)
            if m:
                pending_update, train_itrs = int(m.group(1)), int(m.group(2))
                pending_itr = None
                current = None
                continue
            m = RE_ITR.match(line)
            if m and train_itrs is not None and pending_itr is None:
                # old logs (no "Greedy test" line): the check follows training from this first printed itr
                pending_itr = int(m.group(1)) + train_itrs
                continue
            m = RE_SOLVING.match(line)
            if m and pending_itr is not None:
                current = {"itr": pending_itr, "update_num": pending_update, "gbfs_steps": int(m.group(2)),
                           "solved": {}}
                continue
            m = RE_BACK.match(line)
            if m and current is not None:
                current["solved"][int(m.group(1))] = float(m.group(2))
                continue
            if RE_END.match(line) and current is not None:
                checks[current["itr"]] = current
                current, pending_itr, train_itrs = None, None, None

    return [checks[k] for k in sorted(checks)]


def deepest_depth(solved: Dict[int, float], threshold: float = SOLVED_THRESHOLD) -> int:
    """ Largest depth with %Solved >= threshold (0 if none; depth 0 is the goal state). """
    depths = [d for d, p in solved.items() if p >= threshold]
    return max(depths) if depths else 0


def summarize(checks: List[Dict]) -> Dict:
    """ Latest check and the deepest depth over all checks so far. """
    if not checks:
        return {"itr": None, "deepest_depth": None, "deepest_depth_max": None, "gbfs_steps": None}
    depths = [deepest_depth(c["solved"]) for c in checks]
    return {"itr": checks[-1]["itr"], "deepest_depth": depths[-1], "deepest_depth_max": max(depths),
            "gbfs_steps": checks[-1]["gbfs_steps"]}


def write_csv(checks: List[Dict], path: str) -> None:
    depths = sorted({d for c in checks for d in c["solved"]})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["itr", "update_num", "gbfs_steps", "deepest_depth"] + ["solved_d%i" % d for d in depths])
        for c in checks:
            w.writerow([c["itr"], c["update_num"], c["gbfs_steps"], deepest_depth(c["solved"])] +
                       [c["solved"].get(d, "") for d in depths])


def plot(checks: List[Dict], out_dir: str) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ink, muted, surface, blue = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6"
    itrs = [c["itr"] for c in checks]
    files = []

    fig, ax = plt.subplots(figsize=(8, 4), facecolor=surface)
    ax.set_facecolor(surface)
    ax.plot(itrs, [deepest_depth(c["solved"]) for c in checks], color=blue, linewidth=2,
            label="deepest depth with %%Solved >= %g%%" % SOLVED_THRESHOLD)
    ax.plot(itrs, [c["gbfs_steps"] - 1 for c in checks], color=muted, linewidth=1, linestyle="--",
            label="cap: deepest depth the check can solve (GBFS steps - 1)")
    ax.set_xlabel("training iteration", color=ink)
    ax.set_ylabel("scramble depth", color=ink)
    ax.set_title("Deepest depth reached during training (greedy check)", color=ink, loc="left")
    ax.grid(True, color="#e4e3df", linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    files.append(os.path.join(out_dir, "training_depth.png"))
    fig.savefig(files[-1], dpi=130)
    plt.close(fig)

    depths = sorted({d for c in checks for d in c["solved"]})
    grid = np.full((len(depths), len(checks)), np.nan)
    for j, c in enumerate(checks):
        for i, d in enumerate(depths):
            if d in c["solved"]:
                grid[i, j] = c["solved"][d]
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=surface)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#86b6ef", "#2a78d6",
                                                                             "#0d366b"])
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, vmin=0, vmax=100, interpolation="nearest",
                   extent=(-0.5, len(checks) - 0.5, -0.5, len(depths) - 0.5))
    ticks = np.unique(np.linspace(0, len(checks) - 1, min(8, len(checks)), dtype=int))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(itrs[t]) for t in ticks])
    ax.set_yticks(range(0, len(depths), max(1, len(depths) // 10)))
    ax.set_yticklabels([str(depths[i]) for i in range(0, len(depths), max(1, len(depths) // 10))])
    ax.set_xlabel("training iteration", color=ink)
    ax.set_ylabel("scramble depth", color=ink)
    ax.set_title("%Solved by the greedy check, per scramble depth", color=ink, loc="left")
    fig.colorbar(im, ax=ax, label="%Solved")
    fig.tight_layout()
    files.append(os.path.join(out_dir, "training_solved.png"))
    fig.savefig(files[-1], dpi=130)
    plt.close(fig)

    return files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True, help="avi.py output.txt (all sessions)")
    parser.add_argument('--out', type=str, required=True, help="Output directory for the CSV and plots")
    args = parser.parse_args()

    checks = parse_log(args.log)
    if not checks:
        print("No complete greedy checks in %s" % args.log)
        return
    os.makedirs(args.out, exist_ok=True)
    write_csv(checks, os.path.join(args.out, "training_depth.csv"))
    files = plot(checks, args.out)
    s = summarize(checks)
    print("%i greedy checks; latest at itr %i: deepest depth %i (max so far %i, GBFS steps %i)" % (
        len(checks), s["itr"], s["deepest_depth"], s["deepest_depth_max"], s["gbfs_steps"]))
    print("wrote %s" % ", ".join([os.path.join(args.out, "training_depth.csv")] + files))


if __name__ == "__main__":
    main()
