""" Solve small cubes (3x3, 2x2) by searching over their 4x4 embeddings with the 4x4 cost-to-go network (J4).

The search is the batched weighted A* from search_methods/astar.py, unchanged. It runs on a restricted 4x4
environment (MappedCube4) whose only successors are the mapped small-cube moves:
  3x3: 4x4 outer moves 0-11 (MOVE_3TO4);
  2x2: the 12 macros outer + inner of the same face and direction (MOVE_2TO4), each counted as cost 1.
States are Cube4State, so the closed set is the colour-based one, the heuristic is J4 on 4x4 states and the goal is
Cube4.is_solved. Because successor k of a MappedCube4 is small-cube move k, a path's move indices are small-cube
moves directly.

Every returned path is replayed in the native small environment (Cube3 / Cube2) from the original small state, and
native 4x4 paths are replayed in Cube4. Only a replay that ends solved counts as solved.
"""
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from environments.cube4 import Cube4, Cube4State
from environments.environment_abstract import Environment, State
from search_methods.astar import AStar, get_path
from utils import env_utils
from utils.cube_mapping import MOVE_2TO4, MOVE_3TO4, embed_2to4, embed_3to4

SMALL_ENVS: Dict[str, Tuple[Callable, List[List[int]]]] = {
    "cube3": (embed_3to4, MOVE_3TO4),
    "cube2": (embed_2to4, MOVE_2TO4),
}


class MappedCube4(Cube4):
    """ Cube4 restricted to mapped small-cube moves. Action k applies the 4x4 move sequence move_map[k] at cost 1.
    Everything else (states, colour-based hashing, is_solved, nnet input, nnet model) is Cube4's. """

    def __init__(self, move_map: List[List[int]]):
        super().__init__()
        self.move_map: List[List[int]] = [list(m) for m in move_map]

    def get_num_moves(self) -> int:
        return len(self.move_map)

    def _move_np(self, states_np: np.ndarray, action: int):
        for move in self.move_map[action]:
            states_np, _ = Cube4._move_np(self, states_np, move)
        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]

        return states_np, transition_costs

    def prev_state(self, states: List[Cube4State], action: int) -> List[Cube4State]:
        states_np = np.stack([x.colors for x in states], axis=0)
        for move in self.move_map[action][::-1]:
            inverse: int = Cube4.moves.index(Cube4.moves_rev[move])
            states_np, _ = Cube4._move_np(self, states_np, inverse)

        return [Cube4State(x) for x in list(states_np)]


def translate_path(moves: List[int]) -> List[int]:
    """ Search path (MappedCube4 actions) -> small-cube move indices. Identity by construction; kept as one function
    so the replay check covers the translation step. """
    return [int(m) for m in moves]


def replay_solves(env: Environment, state: State, moves: List[int]) -> bool:
    """ Apply moves in env starting from state; True only if every move is valid and the result is solved. """
    num_moves: int = env.get_num_moves()
    for move in moves:
        if not (0 <= move < num_moves):
            return False
        state = env.next_state([state], move)[0][0]

    return bool(env.is_solved([state])[0])


def _run_astar(env: Environment, states: List[State], heuristic_fn: Callable, weight: float, batch_size: int,
               max_nodes: Optional[int]) -> List[Dict]:
    """ Batched weighted A* (astar.AStar) with a per-instance node budget. An instance that has generated max_nodes
    nodes without popping a goal is retired (its open set is emptied) and reported as not found. """
    start_time = time.time()
    astar = AStar(states, env, heuristic_fn, [weight] * len(states))
    num = len(states)
    done_time: List[Optional[float]] = [None] * num
    retired: List[bool] = [False] * num

    while True:
        found = astar.has_found_goal()
        for i in range(num):
            if done_time[i] is None and (found[i] or retired[i]):
                done_time[i] = time.time() - start_time
            if (not found[i]) and (not retired[i]) and (max_nodes is not None) and \
                    astar.get_num_nodes_generated(i) >= max_nodes:
                astar.instances[i].open_set = []
                retired[i] = True
                done_time[i] = time.time() - start_time
        active = [i for i in range(num) if not found[i] and not retired[i] and len(astar.instances[i].open_set) > 0]
        if len(active) == 0:
            break
        astar.step(heuristic_fn, batch_size)

    results: List[Dict] = []
    found = astar.has_found_goal()
    for i in range(num):
        moves: List[int] = []
        if found[i]:
            _, moves, _ = get_path(astar.get_goal_node_smallest_path_cost(i))
        results.append({"found": bool(found[i]), "moves": [int(m) for m in moves],
                        "nodes": int(astar.get_num_nodes_generated(i)),
                        "time": float(done_time[i] if done_time[i] is not None else time.time() - start_time)})

    return results


def solve_embedded(small_states: List[State], small_env_name: str, heuristic_fn: Callable, weight: float = 1.0,
                   batch_size: int = 100, max_nodes: Optional[int] = 100000) -> List[Dict]:
    """ Solve 3x3 / 2x2 states by A* over their 4x4 embeddings, using only mapped moves and the 4x4 heuristic.

    @param small_states: Cube3State or Cube2State list
    @param small_env_name: "cube3" or "cube2"
    @param heuristic_fn: maps a list of Cube4State to cost-to-go estimates (J4), e.g. nnet_utils.load_heuristic_fn
    @param weight: weight on path cost (astar.py convention: cost = weight * g + h)
    @param batch_size: nodes popped per instance per A* step
    @param max_nodes: per-cube budget of generated nodes (None = unlimited)
    @return: per state: solved (replay in the native small env succeeded), found (search reached a 4x4 goal),
             moves (small-cube move indices), nodes, time (seconds until this cube finished, batched run)
    """
    if small_env_name not in SMALL_ENVS:
        raise ValueError("small_env_name must be one of %s" % list(SMALL_ENVS))
    embed, move_map = SMALL_ENVS[small_env_name]
    small_env: Environment = env_utils.get_environment(small_env_name)
    big_env = MappedCube4(move_map)

    states_np = np.stack([s.colors for s in small_states])
    states4: List[Cube4State] = [Cube4State(x) for x in embed(states_np)]

    results = _run_astar(big_env, states4, heuristic_fn, weight, batch_size, max_nodes)
    for small_state, res in zip(small_states, results):
        res["moves"] = translate_path(res["moves"])
        res["solved"] = bool(res["found"] and replay_solves(small_env, small_state, res["moves"]))

    return results


def solve_native4(states: List[Cube4State], heuristic_fn: Callable, weight: float = 1.0, batch_size: int = 100,
                  max_nodes: Optional[int] = 100000) -> List[Dict]:
    """ Native 4x4 solve: astar.py's A* on Cube4 with all 24 moves. Paths are replayed in Cube4. """
    env = Cube4()
    results = _run_astar(env, states, heuristic_fn, weight, batch_size, max_nodes)
    for state, res in zip(states, results):
        res["solved"] = bool(res["found"] and replay_solves(env, state, res["moves"]))

    return results


def zero_heuristic(states: List[State]) -> np.ndarray:
    return np.zeros(len(states))
