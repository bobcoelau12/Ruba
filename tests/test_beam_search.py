from collections import deque

import numpy as np
import pytest

from environments.cube4 import Cube4, Cube4State
from scripts.step3_eval import apply_seq, scramble
from search_methods import beam_search as bs
from search_methods import transfer_search as ts
from utils import env_utils
from utils.cube_mapping import MOVE_2TO4, embed_2to4


def scrambled(name, depth, num=6, seed=0):
    env = env_utils.get_environment(name)
    return env, [apply_seq(env, s) for s in scramble(env, depth, num, seed)]


def exact_heuristic_2x2_embedded(max_depth=3):
    """ True distance (in mapped 2x2 moves) for embedded 2x2 states within max_depth of the goal, else 99. """
    env = ts.MappedCube4(MOVE_2TO4)
    goal = env.generate_goal_states(1)[0]
    dist = {goal: 0}
    frontier = deque([goal])
    while frontier:
        s = frontier.popleft()
        if dist[s] == max_depth:
            continue
        for child in env.expand([s])[0][0]:
            if child not in dist:
                dist[child] = dist[s] + 1
                frontier.append(child)
    return lambda states: np.array([dist.get(s, 99) for s in states], dtype=float), dist


# ---------------------------------------------------------------- zero heuristic, replay

@pytest.mark.parametrize("name,depth,width", [("cube3", 1, 10), ("cube3", 2, 200), ("cube3", 3, 2000),
                                              ("cube2", 2, 200), ("cube2", 3, 2000)])
def test_zero_heuristic_wide_beam_solves_embedded(name, depth, width):
    env, states = scrambled(name, depth, seed=depth)
    results = bs.solve_embedded(states, name, ts.zero_heuristic, beam_width=width, max_steps=depth, max_nodes=None)
    for state, r in zip(states, results):
        assert r["found"] and r["solved"] and r["stop"] == "found"
        assert len(r["moves"]) <= depth and all(0 <= m < 12 for m in r["moves"])
        assert ts.replay_solves(env, state, r["moves"])
        assert r["nodes"] > 0 and r["time"] >= 0


@pytest.mark.parametrize("depth", [1, 2])
def test_zero_heuristic_wide_beam_solves_native4(depth):
    env, states = scrambled("cube4", depth, 4, seed=depth)
    results = bs.solve_native4(states, ts.zero_heuristic, beam_width=1000, max_steps=depth, max_nodes=None)
    for state, r in zip(states, results):
        assert r["solved"] and len(r["moves"]) <= depth
        assert ts.replay_solves(env, state, r["moves"])


def test_wrong_translation_fails_replay(monkeypatch):
    _, states = scrambled("cube3", 2, seed=4)
    good = bs.solve_embedded(states, "cube3", ts.zero_heuristic, beam_width=200, max_steps=2, max_nodes=None)
    assert all(r["solved"] for r in good)
    monkeypatch.setattr(ts, "translate_path", lambda moves: [(int(m) + 1) % 12 for m in moves])
    bad = bs.solve_embedded(states, "cube3", ts.zero_heuristic, beam_width=200, max_steps=2, max_nodes=None)
    assert all(r["found"] and not r["solved"] for r in bad)


def test_2x2_beam_paths_drive_the_4x4_to_solved():
    env4 = Cube4()
    _, states = scrambled("cube2", 3, seed=7)
    for state, r in zip(states, bs.solve_embedded(states, "cube2", ts.zero_heuristic, 2000, 3, None)):
        big = embed_2to4(state.colors)[None]
        for m in r["moves"]:
            for a in MOVE_2TO4[m]:
                big, _ = env4._move_np(big, a)
        assert r["solved"] and env4.is_solved([Cube4State(big[0])])[0]


# ---------------------------------------------------------------- the heuristic steers the beam

def test_exact_heuristic_width_one_finds_optimal_paths():
    heur, _ = exact_heuristic_2x2_embedded(3)
    for depth in (1, 2, 3):
        _, states = scrambled("cube2", depth, seed=20 + depth)
        for r in bs.solve_embedded(states, "cube2", heur, beam_width=1, max_steps=depth, max_nodes=None):
            assert r["solved"] and len(r["moves"]) == depth and r["steps"] == depth


def test_inverted_heuristic_width_one_fails():
    heur, _ = exact_heuristic_2x2_embedded(3)
    _, states = scrambled("cube2", 3, seed=23)
    results = bs.solve_embedded(states, "cube2", lambda s: -heur(s), beam_width=1, max_steps=3, max_nodes=None)
    assert not any(r["found"] for r in results)
    assert all(r["stop"] == "max_steps" for r in results)


def test_seen_states_are_never_rescored():
    env4 = Cube4()
    _, states = scrambled("cube4", 3, 1, seed=3)
    scored = []

    def recording(s):
        scored.extend(s)
        return np.zeros(len(s))

    r = bs.beam_search(states, env4, recording, beam_width=30, max_steps=4, max_nodes=None)[0]
    assert len(scored) == len(set(scored))  # colour hash: every state scored at most once
    assert states[0] not in set(scored)
    assert r["nodes"] > len(scored)  # duplicates (e.g. the inverse move back to a beam state) were dropped


def test_batched_equals_one_at_a_time():
    _, states = scrambled("cube3", 4, 5, seed=5)

    def h(s):
        return np.array([float(x.colors[:8].sum() % 7) for x in s])

    together = bs.solve_embedded(states, "cube3", h, beam_width=20, max_steps=6, max_nodes=None)
    alone = [bs.solve_embedded([s], "cube3", h, beam_width=20, max_steps=6, max_nodes=None)[0] for s in states]
    for a, b in zip(together, alone):
        assert (a["found"], a["moves"], a["nodes"], a["steps"]) == (b["found"], b["moves"], b["nodes"], b["steps"])


# ---------------------------------------------------------------- stopping

def test_node_budget_counts_as_failure():
    _, states = scrambled("cube3", 5, 3, seed=8)
    results = bs.solve_embedded(states, "cube3", ts.zero_heuristic, beam_width=10, max_steps=50, max_nodes=50)
    for r in results:
        assert not r["found"] and not r["solved"] and r["moves"] == [] and r["stop"] == "max_nodes"
        assert 50 <= r["nodes"] <= 50 + 10 * 12


def test_solved_root_and_empty_input():
    env4 = Cube4()
    goal = env4.generate_goal_states(1)[0]
    r = bs.solve_native4([goal], ts.zero_heuristic, 10, 5, None)[0]
    assert r["found"] and r["solved"] and r["moves"] == [] and r["nodes"] == 0
    assert bs.beam_search([], env4, ts.zero_heuristic) == []
