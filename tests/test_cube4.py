import hashlib

import numpy as np
import pytest

from environments.cube2 import Cube2State
from environments.cube3 import Cube3, Cube3State
from environments.cube4 import Cube4, Cube4State, compute_rotation_idxs, sticker_coords
from utils import env_utils

NUM_MOVES = 24


@pytest.fixture(scope="module")
def env() -> Cube4:
    return Cube4()


@pytest.fixture(scope="module")
def scrambled(env):
    np.random.seed(0)
    states, _ = env.generate_states(20, (10, 40))
    return states


def colors(states):
    return np.stack([s.colors for s in states])


def apply_moves(env, state_np: np.ndarray, moves) -> np.ndarray:
    states_np = state_np[None]
    for m in moves:
        states_np, _ = env._move_np(states_np, env.moves.index(m) if isinstance(m, str) else m)
    return states_np[0]


# ---------------------------------------------------------------- basic structure

def test_registered_and_shapes(env):
    assert isinstance(env_utils.get_environment("cube4"), Cube4)
    assert env.get_num_moves() == NUM_MOVES
    assert env.moves[:12] == Cube3.moves
    assert env.moves[12:] == [m.lower() for m in Cube3.moves]
    assert env.moves[12:14] == ["u-1", "u1"]

    goal = env.generate_goal_states(3, np_format=True)
    assert goal.shape == (3, 96) and goal.dtype == np.uint8
    assert np.array_equal(goal[0], np.arange(96))
    assert np.array_equal(env.generate_goal_states(1)[0].colors, np.arange(96))

    nnet_in = env.state_to_nnet_input(env.generate_goal_states(2))
    assert len(nnet_in) == 1 and nnet_in[0].shape == (2, 96) and nnet_in[0].dtype == np.uint8
    assert np.array_equal(nnet_in[0][0], np.arange(96) // 16)

    nnet = env.get_nnet_model()
    assert nnet.state_dim == 96 and nnet.one_hot_depth == 6


def test_hash_and_eq(env, scrambled):
    a = Cube4State(scrambled[0].colors.copy())
    b = Cube4State(scrambled[0].colors.copy())
    assert a == b and hash(a) == hash(b)
    assert len({a, b}) == 1


def test_costs_and_expand(env, scrambled):
    _, tc = env.next_state(scrambled, 5)
    assert tc == [1.0] * len(scrambled)

    children, costs = env.expand(scrambled)
    assert len(children) == len(scrambled)
    for i, s in enumerate(scrambled):
        assert len(children[i]) == NUM_MOVES
        assert np.array_equal(costs[i], np.ones(NUM_MOVES))
        for a in range(NUM_MOVES):
            assert children[i][a] == env.next_state([s], a)[0][0]


def test_generate_states(env):
    states, nums = env.generate_states(50, (0, 0))
    assert nums == [0] * 50 and all(env.is_solved(states))
    states, nums = env.generate_states(200, (3, 30))
    assert len(states) == 200 and min(nums) >= 3 and max(nums) <= 30
    assert all(sorted(s.colors.tolist()) == list(range(96)) for s in states)


# ---------------------------------------------------------------- move algebra

@pytest.mark.parametrize("action", range(NUM_MOVES))
def test_move_order_four(env, scrambled, action):
    states = scrambled + env.generate_goal_states(1)
    moved = states
    for _ in range(4):
        moved, _ = env.next_state(moved, action)
    assert np.array_equal(colors(moved), colors(states))

    # a quarter turn (and a half turn) is not the identity
    once, _ = env.next_state(states, action)
    twice, _ = env.next_state(once, action)
    assert not np.array_equal(colors(once), colors(states))
    assert not np.array_equal(colors(twice), colors(states))


@pytest.mark.parametrize("action", range(NUM_MOVES))
def test_move_then_inverse(env, scrambled, action):
    moved, _ = env.next_state(scrambled, action)
    assert np.array_equal(colors(env.prev_state(moved, action)), colors(scrambled))

    # the named inverse move (sign flipped) undoes it too
    name = env.moves[action]
    inv = name[0] + ("1" if name[1:] == "-1" else "-1")
    back, _ = env.next_state(moved, env.moves.index(inv))
    assert np.array_equal(colors(back), colors(scrambled))


def test_moves_are_distinct_layers(env):
    # each move moves 4 * 4 * (4 + 1 if outer) stickers: 16 face + 16 side for outer, 16 side for inner
    for m in env.moves:
        n = len(env.rotate_idxs_old[m])
        assert n == (32 if m[0].isupper() else 16), m
        assert sorted(env.rotate_idxs_old[m]) == sorted(env.rotate_idxs_new[m])
    perms = {tuple(apply_moves(env, env.goal_colors, [m])) for m in env.moves}
    assert len(perms) == NUM_MOVES


# ---------------------------------------------------------------- 3x3 correspondence

def test_geometry_reproduces_cube3_exactly():
    """The geometric construction used by Cube4, run at cube_len=3, gives exactly Cube3's permutations."""
    cube3 = Cube3()
    new, old = compute_rotation_idxs(3, Cube3.moves)
    for m in Cube3.moves:
        p_geo = np.arange(54)
        p_geo[new[m]] = old[m]
        p_c3 = np.arange(54)
        p_c3[cube3.rotate_idxs_new[m]] = cube3.rotate_idxs_old[m]
        assert np.array_equal(p_geo, p_c3), m


def test_outer_moves_reduce_to_cube3(env):
    """Under outer moves only, a 4x4 behaves like a 3x3: corner stickers follow 3x3 corners, both wing stickers of
    an edge follow the 3x3 edge sticker, and centres stay put. So colours agree under the row/col reduction
    0 -> 0, 1 -> 1, 2 -> 1, 3 -> 2. This checks that 3x3 move i == 4x4 move i."""
    cube3 = Cube3()
    reduce_idx = np.array([0, 1, 1, 2])
    to3 = np.array([f * 9 + reduce_idx[r] * 3 + reduce_idx[c] for f in range(6) for r in range(4) for c in range(4)])

    rng = np.random.RandomState(1)
    for _ in range(20):
        seq = rng.randint(0, 12, size=rng.randint(1, 40))
        s4 = env.generate_goal_states(1, np_format=True)
        s3 = cube3.generate_goal_states(1, np_format=True)
        for a in seq:
            s4, _ = env._move_np(s4, a)
            s3, _ = cube3._move_np(s3, a)
        assert np.array_equal(s4[0] // 16, s3[0][to3] // 9)


# ---------------------------------------------------------------- independent reference: Jumanji

# Fixed sticker-order mapping, written out once. JUMANJI_TO_CUBE4[k] is the Cube4 sticker index at Jumanji's
# flattened sticker k = face * 16 + row * 4 + col (Jumanji faces UP, FRONT, RIGHT, BACK, LEFT, DOWN, each read in
# reading order when looking at that face, as documented in jumanji/environments/logic/rubiks_cube/utils.py).
# Derived from those viewing conventions and the face layout of DeepCubeA's Cube3 (U, D, L, R, B, F):
# Jumanji UP, FRONT, RIGHT, BACK, LEFT, DOWN -> Cube4 U, F, R, B, L, D, and on every face
# Cube4 (row, col) = (jumanji col, 3 - jumanji row).
JUMANJI_TO_CUBE4 = [
    3, 7, 11, 15, 2, 6, 10, 14, 1, 5, 9, 13, 0, 4, 8, 12,  # UP    -> U (0-15)
    83, 87, 91, 95, 82, 86, 90, 94, 81, 85, 89, 93, 80, 84, 88, 92,  # FRONT -> F (80-95)
    51, 55, 59, 63, 50, 54, 58, 62, 49, 53, 57, 61, 48, 52, 56, 60,  # RIGHT -> R (48-63)
    67, 71, 75, 79, 66, 70, 74, 78, 65, 69, 73, 77, 64, 68, 72, 76,  # BACK  -> B (64-79)
    35, 39, 43, 47, 34, 38, 42, 46, 33, 37, 41, 45, 32, 36, 40, 44,  # LEFT  -> L (32-47)
    19, 23, 27, 31, 18, 22, 26, 30, 17, 21, 25, 29, 16, 20, 24, 28,  # DOWN  -> D (16-31)
]

# Cube4 move index -> Jumanji action (face, depth, amount). Jumanji faces: 0 UP, 1 FRONT, 2 RIGHT, 3 BACK, 4 LEFT,
# 5 DOWN. Amount 0 = clockwise, 1 = anticlockwise (looking at the face). Cube4 "X1" = clockwise, "X-1" = anti.
CUBE4_TO_JUMANJI_ACTION = [
    (0, 0, 1), (0, 0, 0),  # U-1, U1
    (5, 0, 1), (5, 0, 0),  # D-1, D1
    (4, 0, 1), (4, 0, 0),  # L-1, L1
    (2, 0, 1), (2, 0, 0),  # R-1, R1
    (3, 0, 1), (3, 0, 0),  # B-1, B1
    (1, 0, 1), (1, 0, 0),  # F-1, F1
    (0, 1, 1), (0, 1, 0),  # u-1, u1
    (5, 1, 1), (5, 1, 0),  # d-1, d1
    (4, 1, 1), (4, 1, 0),  # l-1, l1
    (2, 1, 1), (2, 1, 0),  # r-1, r1
    (3, 1, 1), (3, 1, 0),  # b-1, b1
    (1, 1, 1), (1, 1, 0),  # f-1, f1
]


def test_jumanji_mapping_is_permutation():
    assert sorted(JUMANJI_TO_CUBE4) == list(range(96))


def test_moves_match_jumanji(env, scrambled):
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    from jumanji.environments import RubiksCube
    from jumanji.environments.logic.rubiks_cube.generator import ScramblingGenerator
    from jumanji.environments.logic.rubiks_cube.types import State as JState

    jenv = RubiksCube(generator=ScramblingGenerator(cube_size=4, num_scrambles_on_reset=10))
    assert jenv.generator.cube_size == 4
    step = jax.jit(jenv.step)
    j2o = np.array(JUMANJI_TO_CUBE4)

    def to_jumanji(c4: np.ndarray) -> JState:
        cube = c4.astype(np.int32)[j2o].reshape(6, 4, 4)
        return JState(cube=jnp.asarray(cube), step_count=jnp.int32(0), key=jax.random.PRNGKey(0))

    def from_jumanji(state: JState) -> np.ndarray:
        out = np.zeros(96, dtype=np.uint8)
        out[j2o] = np.asarray(state.cube).reshape(-1)
        return out

    states = env.generate_goal_states(1) + scrambled
    for s in states:
        assert np.array_equal(from_jumanji(to_jumanji(s.colors)), s.colors)
        for a in range(NUM_MOVES):
            ours = env.next_state([s], a)[0][0].colors
            theirs, _ = step(to_jumanji(s.colors), jnp.array(CUBE4_TO_JUMANJI_ACTION[a], dtype=jnp.int32))
            assert np.array_equal(ours, from_jumanji(theirs)), env.moves[a]


# ---------------------------------------------------------------- is_solved

def whole_cube_rotations(env):
    """All 24 orientations of the solved cube, generated by whole-cube turns about the U and R axes."""
    turn_y = ["U1", "u1", "d-1", "D-1"]
    turn_x = ["R1", "r1", "l-1", "L-1"]
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


def test_is_solved_goal_and_rotations(env):
    assert env.is_solved(env.generate_goal_states(1)).tolist() == [True]
    rotations = whole_cube_rotations(env)
    assert len(rotations) == 24
    assert sum(np.array_equal(r, env.goal_colors) for r in rotations) == 1
    solved = env.is_solved([Cube4State(r) for r in rotations])
    assert solved.dtype == bool and solved.shape == (24,) and solved.all()


def test_is_not_solved_after_one_move(env):
    goal = env.generate_goal_states(1)
    for a in range(NUM_MOVES):
        assert not env.is_solved(env.next_state(goal, a)[0])[0], env.moves[a]
    # and a rotated cube that is then turned is not solved either
    for r in whole_cube_rotations(env):
        assert not env.is_solved([Cube4State(apply_moves(env, r, ["F1"]))])[0]


def test_is_solved_batch(env, scrambled):
    states = env.generate_goal_states(2) + scrambled
    assert env.is_solved(states).tolist() == [True, True] + [False] * len(scrambled)


# ---------------------------------------------------------------- colour-based equality and hashing

def piece_stickers():
    """4x4 sticker indices grouped by cubie: corners (8, 3), wings (24, 2), centres (24,)."""
    points, normals = sticker_coords(4)
    cubies = {}
    for i, c in enumerate(points - normals):
        cubies.setdefault(tuple(c), []).append(i)
    corners = np.array([v for v in cubies.values() if len(v) == 3])
    wings = np.array([v for v in cubies.values() if len(v) == 2])
    centres = np.array([v[0] for v in cubies.values() if len(v) == 1])
    assert corners.shape == (8, 3) and wings.shape == (24, 2) and centres.shape == (24,)
    return corners, wings, centres


def shuffle_same_colour_centres(state: np.ndarray, rng) -> np.ndarray:
    """Permute centre labels among centre positions that currently show the same colour."""
    _, _, centres = piece_stickers()
    out = state.copy()
    for colour in range(6):
        pos = centres[state[centres] // 16 == colour]
        out[pos] = state[rng.permutation(pos)]
    return out


def test_colours_identify_corners_and_wings(env):
    """On reachable states, the colours a corner or wing slot shows determine exactly which labels sit there: every
    (slot, colours) pair seen maps to one label tuple, and every slot/piece/orientation combination is seen (a wing
    can sit in a slot in only one orientation, so its mirror twin can never show the same colours there)."""
    corners, wings, _ = piece_stickers()
    np.random.seed(1)
    states = colors(env.generate_states(5000, (20, 60))[0]).astype(np.int64)

    for slots, num_combos in ((corners, 8 * 8 * 3), (wings, 24 * 24)):
        lab = states[:, slots]  # (B, num_slots, k)
        col = lab // 16
        k = slots.shape[1]
        key = np.arange(len(slots))[None, :] * 6 ** k + (col * 6 ** np.arange(k)[::-1]).sum(axis=2)
        value = (lab * 96 ** np.arange(k)[::-1]).sum(axis=2)
        pairs = np.unique(np.stack([key.ravel(), value.ravel()], axis=1), axis=0)
        keys, counts = np.unique(pairs[:, 0], return_counts=True)
        assert counts.max() == 1  # colours -> labels is a function
        assert len(keys) == num_combos  # and every combination occurred


def test_colour_equal_only_for_centre_permutations(env):
    """Pairs from random scrambles: same-colour centre permutations compare equal; any pair whose corners or wings
    differ (built with real move sequences, commutators A B A' B') compares different."""
    corners, wings, centres = piece_stickers()
    rng = np.random.RandomState(2)
    np.random.seed(2)
    scrambles = colors(env.generate_states(300, (20, 40))[0])
    inverse = [env.moves.index(env.moves_rev[a]) for a in range(NUM_MOVES)]

    # centre permutations: equal, same hash
    for s in scrambles:
        t = shuffle_same_colour_centres(s, rng)
        assert Cube4State(s) == Cube4State(t) and hash(Cube4State(s)) == hash(Cube4State(t))

    # move-built pairs
    seen = {"corner": 0, "wing_only": 0}
    for i in range(2000):
        a = list(rng.randint(0, NUM_MOVES, rng.randint(1, 4)))
        b = list(rng.randint(0, NUM_MOVES, rng.randint(1, 3)))
        seq = a + b + [inverse[m] for m in a[::-1]] + [inverse[m] for m in b[::-1]]
        s = scrambles[i % len(scrambles)]
        t = apply_moves(env, s, seq)
        corner_changed = not np.array_equal(s[corners], t[corners])
        wing_changed = not np.array_equal(s[wings], t[wings])
        only_centres_within_colour = (not corner_changed and not wing_changed and
                                      np.array_equal(s[centres] // 16, t[centres] // 16))
        assert (Cube4State(s) == Cube4State(t)) == only_centres_within_colour
        if corner_changed or wing_changed:
            assert Cube4State(s) != Cube4State(t)
        seen["corner"] += corner_changed
        seen["wing_only"] += wing_changed and not corner_changed
    assert seen["corner"] > 0 and seen["wing_only"] > 0


def test_centre_permutations_collapse_in_sets(env, scrambled):
    rng = np.random.RandomState(3)
    for base in [env.goal_colors] + [s.colors for s in scrambled[:5]]:
        variants = [Cube4State(shuffle_same_colour_centres(base, rng)) for _ in range(50)]
        assert len({tuple(v.colors) for v in variants}) > 1  # the labels really differ
        assert len(set(variants)) == 1
        assert len({hash(v) for v in variants}) == 1
        closed = {variants[0]: 0.0}  # how A* uses states (closed_dict)
        assert all(v in closed for v in variants) and Cube4State(base) in closed


def test_single_move_from_solved_differs(env):
    goal = Cube4State(env.goal_colors)
    children = [env.next_state([goal], a)[0][0] for a in range(NUM_MOVES)]
    for a, child in enumerate(children):
        assert child != goal, env.moves[a]
    assert len(set(children) | {goal}) == NUM_MOVES + 1


def test_cube3_and_cube2_states_still_use_labels():
    # two labels of the same colour swapped: different states on 3x3 and 2x2 (no two pieces look identical there)
    s3 = np.arange(54, dtype=np.uint8)
    t3 = s3.copy(); t3[[0, 2]] = t3[[2, 0]]
    assert Cube3State(s3) != Cube3State(t3)
    assert hash(Cube3State(s3)) == hash(s3.tobytes()) and hash(Cube3State(t3)) == hash(t3.tobytes())
    s2 = np.arange(24, dtype=np.uint8)
    t2 = s2.copy(); t2[[0, 1]] = t2[[1, 0]]
    assert Cube2State(s2) != Cube2State(t2)
    assert hash(Cube2State(s2)) == hash(s2.tobytes())


# ---------------------------------------------------------------- Cube3 unchanged

# sha256 of Cube3's rotate_idxs_new/old (int64) for all 12 moves, computed from the unmodified upstream DeepCubeA code
CUBE3_PERM_SHA256 = "29b5dec3116ea5508eb715eae69a6710af754a6e32e1834503d86a434b431541"


def test_cube3_unchanged():
    cube3 = env_utils.get_environment("cube3")
    assert isinstance(cube3, Cube3)
    assert cube3.moves == ['U-1', 'U1', 'D-1', 'D1', 'L-1', 'L1', 'R-1', 'R1', 'B-1', 'B1', 'F-1', 'F1']
    h = hashlib.sha256()
    for m in cube3.moves:
        h.update(np.asarray(cube3.rotate_idxs_new[m], dtype=np.int64).tobytes())
        h.update(np.asarray(cube3.rotate_idxs_old[m], dtype=np.int64).tobytes())
    assert h.hexdigest() == CUBE3_PERM_SHA256

    np.random.seed(0)
    states, _ = cube3.generate_states(20, (5, 30))
    goal = cube3.generate_goal_states(1)
    assert cube3.is_solved(goal).tolist() == [True]
    for a in range(cube3.get_num_moves()):
        moved = states
        for _ in range(4):
            moved, _ = cube3.next_state(moved, a)
        assert np.array_equal(colors(moved), colors(states))
        assert np.array_equal(colors(cube3.prev_state(cube3.next_state(states, a)[0], a)), colors(states))
        assert not cube3.is_solved(cube3.next_state(goal, a)[0])[0]

    nnet_in = cube3.state_to_nnet_input(goal)[0]
    assert nnet_in.shape == (1, 54) and nnet_in.dtype == np.uint8
    assert np.array_equal(nnet_in[0], np.arange(54) // 9)
    assert hash(Cube3State(states[0].colors.copy())) == hash(states[0])
