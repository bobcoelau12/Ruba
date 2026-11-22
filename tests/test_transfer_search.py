import os
import pickle
import subprocess
import sys

import numpy as np
import pytest

from environments.cube4 import Cube4, Cube4State
from scripts.step3_eval import apply_seq, scramble
from search_methods import transfer_search as ts
from utils import env_utils, nnet_utils
from utils.cube_mapping import MOVE_2TO4, MOVE_3TO4, embed_2to4, embed_3to4

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMALL = {"cube3": (embed_3to4, MOVE_3TO4), "cube2": (embed_2to4, MOVE_2TO4)}


def small_states(name, depth, num=8, seed=0):
    env = env_utils.get_environment(name)
    seqs = scramble(env, depth, num, seed)
    return env, seqs, [apply_seq(env, s) for s in seqs]


# ---------------------------------------------------------------- restricted environment

@pytest.mark.parametrize("name", ["cube3", "cube2"])
def test_mapped_cube4_successors_are_mapped_moves(name):
    embed, move_map = SMALL[name]
    env4 = Cube4()
    mapped = ts.MappedCube4(move_map)
    assert mapped.get_num_moves() == 12
    _, _, states = small_states(name, 4, 3)
    s4 = [Cube4State(x) for x in embed(np.stack([s.colors for s in states]))]
    children, costs = mapped.expand(s4)
    for state, kids, tc in zip(s4, children, costs):
        assert len(kids) == 12 and np.array_equal(tc, np.ones(12))
        for k, child in enumerate(kids):
            expected = state.colors[None]
            for m in move_map[k]:
                expected, _ = env4._move_np(expected, m)
            assert np.array_equal(child.colors, expected[0])
            assert np.array_equal(mapped.prev_state([child], k)[0].colors, state.colors)


# ---------------------------------------------------------------- zero heuristic solves, replay

@pytest.mark.parametrize("name", ["cube3", "cube2"])
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_zero_heuristic_solves_embedded(name, depth):
    env, seqs, states = small_states(name, depth, seed=depth)
    results = ts.solve_embedded(states, name, ts.zero_heuristic, weight=1.0, batch_size=100, max_nodes=None)
    for state, res in zip(states, results):
        assert res["found"] and res["solved"]
        assert len(res["moves"]) <= depth
        assert all(0 <= m < 12 for m in res["moves"])
        assert ts.replay_solves(env, state, res["moves"])  # independent replay in the native env
        assert res["nodes"] > 0 and res["time"] >= 0


@pytest.mark.parametrize("depth", [1, 2])
def test_zero_heuristic_solves_native4(depth):
    env, _, states = small_states("cube4", depth, 4, seed=depth)
    results = ts.solve_native4(states, ts.zero_heuristic, weight=1.0, batch_size=100, max_nodes=None)
    for state, res in zip(states, results):
        assert res["solved"] and len(res["moves"]) <= depth
        assert ts.replay_solves(env, state, res["moves"])


# ---------------------------------------------------------------- path translation

def test_known_3x3_scramble_inverse_is_found():
    env = env_utils.get_environment("cube3")
    inverse = [env.moves.index(env.moves_rev[a]) for a in range(12)]
    cases = [[a] for a in range(12)] + [[env.moves.index("U1"), env.moves.index("R1")],
                                        [env.moves.index("F-1"), env.moves.index("L1"), env.moves.index("D1")]]
    states = [apply_seq(env, seq) for seq in cases]
    results = ts.solve_embedded(states, "cube3", ts.zero_heuristic, weight=1.0, batch_size=100, max_nodes=None)
    for seq, res in zip(cases, results):
        assert res["solved"]
        # these scrambles have a unique optimal solution: the inverse sequence, as 3x3 move indices
        assert res["moves"] == [inverse[m] for m in seq[::-1]], ([env.moves[m] for m in seq],
                                                                 [env.moves[m] for m in res["moves"]])


def test_2x2_paths_are_small_moves_and_drive_the_4x4_to_solved():
    env = env_utils.get_environment("cube2")
    env4 = Cube4()
    _, seqs, states = small_states("cube2", 2, 6, seed=9)
    results = ts.solve_embedded(states, "cube2", ts.zero_heuristic, weight=1.0, batch_size=100, max_nodes=None)
    for state, res in zip(states, results):
        assert res["solved"] and len(res["moves"]) <= 2 and all(0 <= m < 12 for m in res["moves"])
        big = embed_2to4(state.colors)[None]
        for m in res["moves"]:
            for a in MOVE_2TO4[m]:
                big, _ = env4._move_np(big, a)
        assert env4.is_solved([Cube4State(big[0])])[0]


def test_wrong_translation_fails_replay(monkeypatch):
    _, _, states = small_states("cube3", 2, 6, seed=11)
    good = ts.solve_embedded(states, "cube3", ts.zero_heuristic, weight=1.0, batch_size=100, max_nodes=None)
    assert all(r["solved"] for r in good)

    monkeypatch.setattr(ts, "translate_path", lambda moves: [(int(m) + 1) % 12 for m in moves])
    bad = ts.solve_embedded(states, "cube3", ts.zero_heuristic, weight=1.0, batch_size=100, max_nodes=None)
    for r in bad:
        assert r["found"] and not r["solved"]  # the search reached a goal, but the replay check rejects it


def test_node_budget_counts_as_failure():
    _, _, states = small_states("cube3", 5, 3, seed=12)
    results = ts.solve_embedded(states, "cube3", ts.zero_heuristic, weight=1.0, batch_size=10, max_nodes=50)
    for r in results:
        assert not r["found"] and not r["solved"] and r["moves"] == []


def test_replay_solves_rejects_invalid_moves():
    env = env_utils.get_environment("cube3")
    goal = env.generate_goal_states(1)[0]
    assert ts.replay_solves(env, goal, [])
    assert not ts.replay_solves(env, goal, [12])
    assert not ts.replay_solves(env, goal, [-1])
    assert not ts.replay_solves(env, goal, [0])


# ---------------------------------------------------------------- train -> checkpoint -> load -> heuristic

def test_train_smoke_checkpoint_load_heuristic(tmp_path):
    save_dir = str(tmp_path / "saved_models")
    cmd = [sys.executable, "ctg_approx/avi.py", "--env", "cube4", "--back_max", "10", "--nnet_name", "cube4",
           "--save_dir", save_dir, "--max_itrs", "2", "--batch_size", "50", "--states_per_update", "100",
           "--num_test", "20", "--update_nnet_batch_size", "200", "--debug"]
    env_vars = dict(os.environ, PYTHONPATH=ROOT, CUDA_VISIBLE_DEVICES="", PYTHONWARNINGS="ignore")
    out = subprocess.run(cmd, cwd=ROOT, env=env_vars, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]

    model_dir = os.path.join(save_dir, "cube4", "current")
    assert os.path.isfile(os.path.join(model_dir, "model_state_dict.pt"))
    assert pickle.load(open(os.path.join(model_dir, "train_itr.pkl"), "rb")) == 2
    assert not any(f.endswith(".tmp") for f in os.listdir(model_dir))

    device = nnet_utils.get_device()[0]
    env4 = Cube4()
    heuristic_fn = nnet_utils.load_heuristic_fn(model_dir, device, False, env4.get_nnet_model(), env4,
                                                clip_zero=True)
    _, _, s3 = small_states("cube3", 3, 5)
    _, _, s2 = small_states("cube2", 3, 5)
    states4 = [Cube4State(x) for x in embed_3to4(np.stack([s.colors for s in s3]))] + \
              [Cube4State(x) for x in embed_2to4(np.stack([s.colors for s in s2]))]
    h = heuristic_fn(states4)
    assert h.shape == (10,) and np.all(np.isfinite(h)) and np.all(h >= 0)

    # and the loaded J4 plugs into the transfer solver
    _, _, s3_d1 = small_states("cube3", 1, 3)
    results = ts.solve_embedded(s3_d1, "cube3", heuristic_fn, weight=1.0, batch_size=50, max_nodes=None)
    assert all(r["solved"] for r in results)
