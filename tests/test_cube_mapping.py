import itertools

import numpy as np
import pytest

from environments.cube2 import Cube2
from environments.cube3 import Cube3
from environments.cube4 import Cube4, Cube4State
from utils import cube_mapping as cm

NUM_RANDOM = 1000
GOAL4 = np.arange(96, dtype=np.uint8)


@pytest.fixture(scope="module")
def cube4():
    return Cube4()


class Case:
    def __init__(self, name, env, block_map, move_map, embed, project, is_reduced, centres_turn):
        self.name, self.env, self.block_map, self.move_map = name, env, block_map, move_map
        self.embed, self.project, self.is_reduced = embed, project, is_reduced
        self.centres_turn = centres_turn

    def same(self, a, b):
        """ Equality used for 4x4 states: up to centre-block turns for 3->4, exact labels for 2->4. """
        return cm.equal_up_to_centre_turns(a, b) if self.centres_turn else np.all(a == b, axis=1)


CASES = {
    "3to4": Case("3to4", Cube3(), cm.MAP_3TO4, cm.MOVE_3TO4, cm.embed_3to4, cm.project_4to3, cm.is_reduced_3, True),
    "2to4": Case("2to4", Cube2(), cm.MAP_2TO4, cm.MOVE_2TO4, cm.embed_2to4, cm.project_4to2, cm.is_reduced_2, False),
}


@pytest.fixture(params=list(CASES))
def case(request):
    return CASES[request.param]


def random_small_states(case, num=NUM_RANDOM, seed=0):
    np.random.seed(seed)
    states, _ = case.env.generate_states(num, (0, 60))
    return np.stack([s.colors for s in states])


def apply_mapped(cube4, case, states4, small_move):
    for a in case.move_map[small_move]:
        states4, _ = cube4._move_np(states4, a)
    return states4


def commutes(cube4, case, states_small) -> bool:
    embedded = case.embed(states_small)
    for i in range(case.env.get_num_moves()):
        lhs = case.embed(case.env._move_np(states_small, i)[0])
        rhs = apply_mapped(cube4, case, embedded, i)
        if not case.same(lhs, rhs).all():
            return False
    return True


# ---------------------------------------------------------------- move maps

def test_move_maps():
    assert cm.MOVE_3TO4 == [[i] for i in range(12)]
    assert cm.MOVE_2TO4 == [[i, i + 12] for i in range(12)]
    for i, (outer, inner) in enumerate(cm.MOVE_2TO4):
        assert Cube4.moves[outer] == Cube3.moves[i] and Cube4.moves[inner] == Cube3.moves[i][0].lower() + Cube3.moves[i][1:]


def test_block_structure(case):
    blocks = case.block_map.blocks
    assert sorted(np.concatenate(blocks).tolist()) == list(range(96))
    sizes = sorted(len(b) for b in blocks)
    if case.centres_turn:
        assert sizes == [1] * 24 + [2] * 24 + [4] * 6
        assert cm.CENTRE_BLOCKS_3TO4.shape == (6, 4)
    else:
        assert sizes == [4] * 24


# ---------------------------------------------------------------- commutation

def test_commutation(cube4, case):
    states = random_small_states(case)
    embedded = case.embed(states)
    for i in range(case.env.get_num_moves()):
        lhs = case.embed(case.env._move_np(states, i)[0])
        rhs = apply_mapped(cube4, case, embedded, i)
        assert case.same(lhs, rhs).all(), case.env.moves[i]
        if not case.centres_turn:
            assert np.array_equal(lhs, rhs), case.env.moves[i]


def test_3to4_exact_except_centres(cube4):
    """3->4: corners and edges commute exactly; only the turned face's centre block differs, by exactly one turn."""
    case = CASES["3to4"]
    states = random_small_states(case, 200)
    centre_pos = cm.CENTRE_BLOCKS_3TO4.reshape(-1)
    other = np.setdiff1d(np.arange(96), centre_pos)
    for i in range(12):
        lhs = case.embed(case.env._move_np(states, i)[0])
        rhs = apply_mapped(cube4, case, case.embed(states), i)
        assert np.array_equal(lhs[:, other], rhs[:, other])
        face = i // 2
        diff_faces = {int(f) for f in np.where((lhs[:, cm.CENTRE_BLOCKS_3TO4] != rhs[:, cm.CENTRE_BLOCKS_3TO4])
                                                .any(axis=(0, 2)))[0]}
        assert diff_faces == {face}
        assert not np.array_equal(lhs, rhs)  # labels cannot match exactly here, see the module docstring


# ---------------------------------------------------------------- solved, round trip

def test_solved_maps_to_solved(cube4, case):
    embedded = case.embed(case.env.goal_colors)
    assert np.array_equal(embedded, GOAL4)
    assert cube4.is_solved([Cube4State(embedded)])[0]
    assert case.is_reduced(GOAL4)
    assert np.array_equal(case.project(GOAL4), case.env.goal_colors)


def test_round_trip(case):
    states = random_small_states(case, seed=1)
    embedded = case.embed(states)
    assert embedded.shape == (NUM_RANDOM, 96) and embedded.dtype == np.uint8
    assert all(sorted(e.tolist()) == list(range(96)) for e in embedded[:50])
    assert np.array_equal(case.project(embedded), states)
    assert case.is_reduced(embedded).all()
    # single states work too
    assert np.array_equal(case.project(case.embed(states[0])), states[0])


# ---------------------------------------------------------------- random walks stay reduced

@pytest.mark.parametrize("start", ["solved", "embedded"])
def test_random_walks_stay_reduced(cube4, case, start):
    rng = np.random.RandomState(2)
    num = 200
    small = (np.tile(case.env.goal_colors, (num, 1)) if start == "solved"
             else random_small_states(case, num, seed=3))
    big = case.embed(small)
    for _ in range(60):
        moves = rng.randint(0, case.env.get_num_moves(), size=num)
        for m in np.unique(moves):
            idx = np.where(moves == m)[0]
            small[idx], _ = case.env._move_np(small[idx], m)
            big[idx] = apply_mapped(cube4, case, big[idx], m)
        assert case.is_reduced(big).all()
        assert case.same(big, case.embed(small)).all()
        assert np.array_equal(case.project(big), small)


# ---------------------------------------------------------------- negative checks

def test_3to4_inner_move_not_reduced(cube4):
    for a in range(12, 24):
        s4, _ = cube4._move_np(GOAL4[None], a)
        assert not cm.is_reduced_3(s4)[0], Cube4.moves[a]
        with pytest.raises(ValueError):
            cm.project_4to3(s4)


def test_2to4_single_layer_move_not_reduced(cube4):
    for a in range(24):  # outer-only (0-11) and inner-only (12-23) moves both break the 2x2 quadrants
        s4, _ = cube4._move_np(GOAL4[None], a)
        assert not cm.is_reduced_2(s4)[0], Cube4.moves[a]
        with pytest.raises(ValueError):
            cm.project_4to2(s4)


def test_project_raises_on_any_bad_state_in_batch(case):
    states = case.embed(random_small_states(case, 10))
    states[3] = Cube4().next_state([Cube4State(GOAL4)], 13)[0][0].colors
    assert case.is_reduced(states).tolist() == [True] * 3 + [False] + [True] * 6
    with pytest.raises(ValueError):
        case.project(states)


def test_unreachable_small_states_are_not_reduced():
    """Well-formed but unreachable small-cube states embed fine but are not reduced."""
    v3 = cm.MAP_3TO4.valid
    goal3 = np.arange(54)
    # two whole edges swapped (permutation parity mismatch: the 4x4 "PLL parity" case)
    s = goal3.copy(); (a0, a1), (b0, b1) = v3.edges[0], v3.edges[1]
    s[[a0, a1, b0, b1]] = s[[b0, b1, a0, a1]]
    # one flipped edge
    f = goal3.copy(); f[list(v3.edges[0])] = f[list(v3.edges[0][::-1])]
    # one twisted corner
    t = goal3.copy(); t[list(v3.corners[0])] = t[list(np.roll(v3.corners[0], 1))]
    # one mirrored corner (two stickers swapped)
    m = goal3.copy(); c = v3.corners[0]; m[[c[1], c[2]]] = m[[c[2], c[1]]]
    for bad in (s, f, t, m):
        assert not cm.is_reduced_3(cm.embed_3to4(bad))
        with pytest.raises(ValueError):
            cm.project_4to3(cm.embed_3to4(bad))

    v2 = cm.MAP_2TO4.valid
    t2 = np.arange(24); t2[list(v2.corners[0])] = t2[list(np.roll(v2.corners[0], 1))]
    assert not cm.is_reduced_2(cm.embed_2to4(t2))


def test_embed_rejects_malformed(case):
    n = 6 * case.env.cube_len ** 2
    with pytest.raises(ValueError):
        case.embed(np.zeros(n, dtype=np.uint8))  # not a permutation
    with pytest.raises(ValueError):
        case.embed(np.arange(n + 1))  # wrong shape
    if case.centres_turn:
        bad = np.arange(n); bad[[0, 1]] = bad[[1, 0]]  # corner label on an edge position
        with pytest.raises(ValueError):
            case.embed(bad)


# ---------------------------------------------------------------- 3x3 centre turns

def test_all_centre_turn_variants_project_to_same_state():
    s3 = random_small_states(CASES["3to4"], 1, seed=4)[0]
    s4 = cm.embed_3to4(s3)
    blocks = cm.CENTRE_BLOCKS_3TO4
    variants = np.tile(s4, (4 ** 6, 1))
    for v, turns in enumerate(itertools.product(range(4), repeat=6)):
        for f, r in enumerate(turns):
            variants[v, blocks[f]] = np.roll(s4[blocks[f]], r)
    assert len({tuple(x) for x in variants}) == 4 ** 6
    assert cm.is_reduced_3(variants).all()
    assert np.array_equal(cm.project_4to3(variants), np.tile(s3, (4 ** 6, 1)))
    assert cm.equal_up_to_centre_turns(variants, np.tile(s4, (4 ** 6, 1))).all()
    assert (variants == s4).all(axis=1).sum() == 1


def test_centre_block_swapped_not_turned_is_rejected():
    s4 = cm.embed_3to4(random_small_states(CASES["3to4"], 1, seed=5)[0])
    for f in range(6):
        b = cm.CENTRE_BLOCKS_3TO4[f]
        for i, j in itertools.combinations(range(4), 2):
            bad = s4.copy()
            bad[[b[i], b[j]]] = bad[[b[j], b[i]]]
            assert not cm.is_reduced_3(bad)
            assert not cm.equal_up_to_centre_turns(bad, s4)
            with pytest.raises(ValueError):
                cm.project_4to3(bad)
        mirrored = s4.copy()
        mirrored[b] = s4[b][::-1]  # a reflection is not a turn either
        assert not cm.is_reduced_3(mirrored)
    # centre blocks of two faces exchanged (each in a valid arrangement): 3x3 centres are not at home
    swapped = s4.copy()
    swapped[cm.CENTRE_BLOCKS_3TO4[0]], swapped[cm.CENTRE_BLOCKS_3TO4[1]] = \
        s4[cm.CENTRE_BLOCKS_3TO4[1]], s4[cm.CENTRE_BLOCKS_3TO4[0]]
    assert not cm.is_reduced_3(swapped)


def test_equal_up_to_centre_turns_is_exact_elsewhere():
    s4 = cm.embed_3to4(random_small_states(CASES["3to4"], 1, seed=6)[0])
    edge_block = next(b for b in cm.MAP_3TO4.blocks if len(b) == 2)
    bad = s4.copy()
    bad[edge_block] = bad[edge_block[::-1]]
    assert not cm.equal_up_to_centre_turns(bad, s4)
    assert cm.equal_up_to_centre_turns(s4, s4)


# ---------------------------------------------------------------- mutation check

@pytest.mark.parametrize("name,kind_size", [("3to4", 2), ("2to4", 4)])
def test_mutation_breaks_commutation(cube4, name, kind_size):
    """Swapping two members of one block (still a bijection, embed(goal) is still arange) must break commutation."""
    case = CASES[name]
    block_map = case.block_map
    states = random_small_states(case, 200, seed=7)
    original = [b.copy() for b in block_map.blocks]
    assert commutes(cube4, case, states)

    p = next(i for i, b in enumerate(original) if len(b) == kind_size)
    mutated = [b.copy() for b in original]
    mutated[p][[0, 1]] = mutated[p][[1, 0]]
    try:
        block_map.set_blocks(mutated)
        assert np.array_equal(case.embed(case.env.goal_colors), GOAL4)
        assert not commutes(cube4, case, states)
    finally:
        block_map.set_blocks(original)

    assert all(np.array_equal(a, b) for a, b in zip(block_map.blocks, original))
    assert commutes(cube4, case, states)
