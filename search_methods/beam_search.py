""" Batched beam search with a cost-to-go heuristic (e.g. J4).

Per cube, the beam starts as the scrambled state. Each step expands every beam state (all successors, generated in
move order), and stops the cube as soon as a successor is a goal. Otherwise successors already seen by this cube
(any earlier beam or successor, compared with the state's hash, i.e. the colour hash for Cube4State) are dropped,
duplicates within the step are kept once, and the beam_width successors with the smallest heuristic value form the
next beam (ties keep move order). All cubes of a call are expanded and scored together, so the network sees one
batch per step.

A cube stops when: a goal is found (found), its beam is empty (no unseen successors), it reached max_steps, or it
generated >= max_nodes successors. The budget is checked between steps, so the last step can exceed it by at most
beam_width x num_moves nodes. Nodes counts generated successors (before duplicate removal), as astar.py does.

Paths are returned like the A* solver's: move indices of `env`. solve_native4 / solve_embedded replay every path in
the native environment of the cube; only a replay that ends solved counts as solved.
"""
import time
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from environments.cube4 import Cube4, Cube4State
from environments.environment_abstract import Environment, State
from search_methods import transfer_search as ts
from utils import env_utils


class _Cube:
    __slots__ = ["beam", "paths", "seen", "nodes", "steps", "found", "moves", "done", "time", "stop"]

    def __init__(self, state: State):
        self.beam: List[State] = [state]
        self.paths: List[List[int]] = [[]]
        self.seen: Set[State] = {state}
        self.nodes: int = 0
        self.steps: int = 0
        self.found: bool = False
        self.moves: List[int] = []
        self.done: bool = False
        self.time: Optional[float] = None
        self.stop: str = ""


def beam_search(states: List[State], env: Environment, heuristic_fn: Callable, beam_width: int = 256,
                max_steps: int = 50, max_nodes: Optional[int] = None) -> List[Dict]:
    """ @return per state: found, moves (env move indices), nodes, steps, time (s until this cube stopped), stop """
    start_time = time.time()
    cubes = [_Cube(s) for s in states]

    roots_solved = env.is_solved(states) if len(states) > 0 else []
    for cube, solved in zip(cubes, roots_solved):
        if solved:
            cube.found, cube.done, cube.stop, cube.time = True, True, "found", time.time() - start_time

    while True:
        active = [c for c in cubes if not c.done]
        if not active:
            break

        # expand all beams of all active cubes at once
        flat_states: List[State] = [s for c in active for s in c.beam]
        children_by_state, _ = env.expand(flat_states)
        num_moves = env.get_num_moves()
        flat_children: List[State] = [ch for kids in children_by_state for ch in kids]
        solved_flat = env.is_solved(flat_children)

        candidates: List[List[State]] = []
        cand_paths: List[List[List[int]]] = []
        pos_state = 0
        for c in active:
            c.steps += 1
            c.nodes += len(c.beam) * num_moves
            cand: List[State] = []
            paths: List[List[int]] = []
            for b in range(len(c.beam)):
                kids = children_by_state[pos_state + b]
                base = (pos_state + b) * num_moves
                for move, child in enumerate(kids):
                    if c.found:
                        break
                    if solved_flat[base + move]:
                        c.found, c.moves = True, c.paths[b] + [move]
                        break
                    if child in c.seen:
                        continue
                    c.seen.add(child)
                    cand.append(child)
                    paths.append(c.paths[b] + [move])
            pos_state += len(c.beam)
            candidates.append(cand)
            cand_paths.append(paths)

        # score all surviving candidates in one heuristic call
        to_score: List[State] = [s for c, cand in zip(active, candidates) if not c.found for s in cand]
        h_all = np.asarray(heuristic_fn(to_score), dtype=float) if to_score else np.zeros(0)
        pos = 0
        for c, cand, paths in zip(active, candidates, cand_paths):
            if c.found:
                c.done, c.stop = True, "found"
            else:
                h = h_all[pos:pos + len(cand)]
                pos += len(cand)
                keep = np.argsort(h, kind="stable")[:beam_width]
                c.beam = [cand[i] for i in keep]
                c.paths = [paths[i] for i in keep]
                if not c.beam:
                    c.done, c.stop = True, "empty"
                elif c.steps >= max_steps:
                    c.done, c.stop = True, "max_steps"
                elif max_nodes is not None and c.nodes >= max_nodes:
                    c.done, c.stop = True, "max_nodes"
            if c.done:
                c.time = time.time() - start_time
                c.beam, c.paths, c.seen = [], [], set()

    return [{"found": c.found, "moves": [int(m) for m in c.moves], "nodes": int(c.nodes), "steps": int(c.steps),
             "time": float(c.time), "stop": c.stop} for c in cubes]


def solve_native4(states: List[Cube4State], heuristic_fn: Callable, beam_width: int = 256, max_steps: int = 50,
                  max_nodes: Optional[int] = 100000) -> List[Dict]:
    """ Beam search on Cube4 with all 24 moves; paths are replayed in Cube4. """
    env = Cube4()
    results = beam_search(states, env, heuristic_fn, beam_width, max_steps, max_nodes)
    for state, res in zip(states, results):
        res["solved"] = bool(res["found"] and ts.replay_solves(env, state, res["moves"]))
    return results


def solve_embedded(small_states: List[State], small_env_name: str, heuristic_fn: Callable, beam_width: int = 256,
                   max_steps: int = 50, max_nodes: Optional[int] = 100000) -> List[Dict]:
    """ Beam search over the 4x4 embeddings of 3x3 / 2x2 states with only the mapped moves (transfer_search's
    MappedCube4), J4 as heuristic and the Cube4 goal; paths are small-cube moves, replayed in Cube3 / Cube2. """
    if small_env_name not in ts.SMALL_ENVS:
        raise ValueError("small_env_name must be one of %s" % list(ts.SMALL_ENVS))
    embed, move_map = ts.SMALL_ENVS[small_env_name]
    small_env: Environment = env_utils.get_environment(small_env_name)
    big_env = ts.MappedCube4(move_map)
    states4 = [Cube4State(x) for x in embed(np.stack([s.colors for s in small_states]))]

    results = beam_search(states4, big_env, heuristic_fn, beam_width, max_steps, max_nodes)
    for small_state, res in zip(small_states, results):
        res["moves"] = ts.translate_path(res["moves"])
        res["solved"] = bool(res["found"] and ts.replay_solves(small_env, small_state, res["moves"]))
    return results
