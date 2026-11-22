import numpy as np
import pytest

from environments.cube2 import Cube2, Cube2State
from environments.cube3 import Cube3
from environments.cube4 import compute_rotation_idxs
from utils import env_utils

NUM_MOVES = 12


@pytest.fixture(scope="module")
def env() -> Cube2:
    return Cube2()


@pytest.fixture(scope="module")
def scrambled(env):
    np.random.seed(0)
    states, _ = env.generate_states(20, (10, 40))
    return states


def colors(states):
    return np.stack([s.colors for s in states])


def apply_moves(env, state_np, moves):
    states_np = state_np[None]
    for m in moves:
        states_np, _ = env._move_np(states_np, env.moves.index(m))
    return states_np[0]


def test_registered_and_shapes(env):
    assert isinstance(env_utils.get_environment("cube2"), Cube2)
    assert env.get_num_moves() == NUM_MOVES
    assert env.moves == Cube3.moves

    goal = env.generate_goal_states(3, np_format=True)
    assert goal.shape == (3, 24) and goal.dtype == np.uint8
    assert np.array_equal(goal[0], np.arange(24))
    assert np.array_equal(env.generate_goal_states(1)[0].colors, np.arange(24))

    nnet_in = env.state_to_nnet_input(env.generate_goal_states(2))
    assert len(nnet_in) == 1 and nnet_in[0].shape == (2, 24) and nnet_in[0].dtype == np.uint8
    assert np.array_equal(nnet_in[0][0], np.arange(24) // 4)

    nnet = env.get_nnet_model()
    assert nnet.state_dim == 24 and nnet.one_hot_depth == 6


def test_permutations_from_geometry(env):
    new, old = compute_rotation_idxs(2, env.moves)
    for m in env.moves:
        assert np.array_equal(env.rotate_idxs_new[m], new[m]) and np.array_equal(env.rotate_idxs_old[m], old[m])
        assert len(env.rotate_idxs_old[m]) == 4 + 8  # face + adjacent rows


def test_hash_eq_costs_expand(env, scrambled):
    a, b = Cube2State(scrambled[0].colors.copy()), Cube2State(scrambled[0].colors.copy())
    assert a == b and hash(a) == hash(b)

    _, tc = env.next_state(scrambled, 3)
    assert tc == [1.0] * len(scrambled)
    children, costs = env.expand(scrambled)
    for i, s in enumerate(scrambled):
        assert len(children[i]) == NUM_MOVES and np.array_equal(costs[i], np.ones(NUM_MOVES))
        for a_ in range(NUM_MOVES):
            assert children[i][a_] == env.next_state([s], a_)[0][0]


def test_generate_states(env):
    states, nums = env.generate_states(50, (0, 0))
    assert nums == [0] * 50 and all(env.is_solved(states))
    states, nums = env.generate_states(200, (3, 30))
    assert len(states) == 200 and min(nums) >= 3 and max(nums) <= 30
    assert all(sorted(s.colors.tolist()) == list(range(24)) for s in states)


@pytest.mark.parametrize("action", range(NUM_MOVES))
def test_move_order_four(env, scrambled, action):
    states = scrambled + env.generate_goal_states(1)
    moved = states
    for _ in range(4):
        moved, _ = env.next_state(moved, action)
    assert np.array_equal(colors(moved), colors(states))
    once, _ = env.next_state(states, action)
    assert not np.array_equal(colors(once), colors(states))


@pytest.mark.parametrize("action", range(NUM_MOVES))
def test_move_then_inverse(env, scrambled, action):
    moved, _ = env.next_state(scrambled, action)
    assert np.array_equal(colors(env.prev_state(moved, action)), colors(scrambled))
    name = env.moves[action]
    inv = name[0] + ("1" if name[1:] == "-1" else "-1")
    back, _ = env.next_state(moved, env.moves.index(inv))
    assert np.array_equal(colors(back), colors(scrambled))


def whole_cube_rotations(env):
    # on a 2x2 a whole-cube turn is a face turn plus the opposite face turned the same way in space
    turn_y, turn_x = ["U1", "D-1"], ["R1", "L-1"]
    seen = {tuple(env.goal_colors)}
    frontier = [env.goal_colors]
    while frontier:
        nxt = []
        for s in frontier:
            for gen in (turn_y, turn_x):
                t = apply_moves(env, s, gen)
                if tuple(t) not in seen:
                    seen.add(tuple(t))
                    nxt.append(t)
        frontier = nxt
    return [np.array(s, dtype=np.uint8) for s in seen]


def test_is_solved_all_orientations(env):
    rotations = whole_cube_rotations(env)
    assert len(rotations) == 24
    solved = env.is_solved([Cube2State(r) for r in rotations])
    assert solved.dtype == bool and solved.shape == (24,) and solved.all()


def test_not_solved_after_one_move(env):
    goal = env.generate_goal_states(1)
    for a in range(NUM_MOVES):
        assert not env.is_solved(env.next_state(goal, a)[0])[0], env.moves[a]
    for r in whole_cube_rotations(env):
        assert not env.is_solved([Cube2State(apply_moves(env, r, ["F1"]))])[0]


def test_is_solved_batch(env, scrambled):
    states = env.generate_goal_states(2) + scrambled
    expected = [True, True] + [False] * len(scrambled)
    # a random walk can land on a whole-cube rotation of solved; that is solved too
    for i, s in enumerate(scrambled):
        if all(len(set(s.colors[f * 4:(f + 1) * 4] // 4)) == 1 for f in range(6)):
            expected[2 + i] = True
    assert env.is_solved(states).tolist() == expected
