""" Exact label-level maps between small cubes (3x3, 2x2) and the 4x4.

States are label arrays: a state of an n x n cube is a permutation of 0..6n^2-1 where entry i is the label of the
sticker at position i and label L is the sticker whose home position is L (solved = np.arange). Every map here works
on labels, not colours.

Embedding. Each small-cube sticker position p owns a block of 4x4 positions:
  3 -> 4: layers grouped {1, 2, 1} (rows/cols 0 -> {0}, 1 -> {1, 2}, 2 -> {3}); corner -> 1 sticker,
          edge -> 2 wings, centre -> 2x2 block.             3x3 move i  ->  4x4 outer move i.
  2 -> 4: layers grouped {2, 2}; every sticker -> a 2x2 quadrant.
          2x2 move i  ->  4x4 outer move i followed by the inner move of the same face and direction
          (a 2-move macro that counts as one move).
embed(s) puts, for every p, the block of 4x4 labels belonging to label s[p] onto the block of 4x4 positions of p,
member k onto member k. The order of the members inside each block comes from 3-D geometry (the sticker
coordinates used by compute_rotation_idxs) using axes attached to the cubie the sticker sits on: a corner sticker
orders by its cubie's two other face normals (taken right-handed), an edge sticker by the edge direction n x m.
These axes turn with the cubie, so every mapped move carries block member k to block member k, and embedding
commutes with moves exactly (on labels), with embed(solved) == np.arange(96).

Why 3x3 centres are only equal up to turning. A 3x3 face turn leaves its centre sticker in place with the same
label, but the corresponding 4x4 outer turn rotates the 2x2 centre block of that face by 90 degrees. The four labels
in the block are distinct, so no fixed order inside the block can be invariant under that rotation: a 3x3 state
simply does not record centre orientation, and a 4x4 state does. So for 3 -> 4:
  - corners and edges are exact on labels;
  - a centre block must hold exactly the 4 labels of one 3x3 centre in their cyclic order, turned by 0, 90, 180
    or 270 degrees (centre block members are listed in cyclic order around the face, so a turn is a cyclic shift);
  - embed_3to4 always writes the 0 degree turn; project_4to3 / is_reduced_3 accept any turn; compare 4x4 states
    with equal_up_to_centre_turns.
The 2 -> 4 map has no such freedom: every quadrant sits on a corner cubie, and it is exact on all labels.

Reduced. is_reduced_*(s4) is True exactly when s4 == embed(s) for some valid small-cube state s (up to centre turns
for 3 -> 4). Valid means reachable by the small cube's own moves: whole cubies with correctly ordered (not mirrored)
stickers, corner twist sum = 0 mod 3, and for the 3x3 also edge flip sum = 0 mod 2, equal corner and edge
permutation parity, and centres at home. Without this, e.g. a 4x4 with two whole edges swapped would count as
reduced although its 3x3 projection is unsolvable. project_4to* is the inverse of embed and raises ValueError on
states that are not reduced.

Colour-level check. The four centre pieces of one colour on a 4x4 are physically identical, so a cube reduced by
hand or by search almost never has them in the label arrangement embed_3to4 writes. is_reduced_3_colour /
project_4to3_colour keep corners and edges as strict as above but only require each centre block to be a single
colour. The strict is_reduced_3 / project_4to3 are kept unchanged.

All functions are vectorised over a batch (shape (B, n) or a single state of shape (n,)) using precomputed index
arrays, like Cube3._move_np.
"""
from typing import Dict, List, Tuple

import numpy as np

from environments.cube4 import Cube4, sticker_coords

CUBE_LEN_BIG: int = 4
NUM_BIG: int = 6 * CUBE_LEN_BIG ** 2
SMALL_MOVES: List[str] = ["%s%i" % (f, n) for f in ['U', 'D', 'L', 'R', 'B', 'F'] for n in [-1, 1]]

# small-cube move index -> list of 4x4 move indices
MOVE_3TO4: List[List[int]] = [[Cube4.moves.index(m)] for m in SMALL_MOVES]
MOVE_2TO4: List[List[int]] = [[Cube4.moves.index(m), Cube4.moves.index(m[0].lower() + m[1:])] for m in SMALL_MOVES]


def _cubie_groups(cube_len: int) -> Tuple[np.ndarray, np.ndarray, Dict[Tuple[int, ...], List[int]]]:
    points, normals = sticker_coords(cube_len)
    cubies: Dict[Tuple[int, ...], List[int]] = dict()
    for idx, c in enumerate(points - normals):
        cubies.setdefault(tuple(c), []).append(idx)
    return points, normals, cubies


def _compute_blocks(cube_len: int, layer_groups: List[List[int]]) -> Tuple[List[np.ndarray], List[str]]:
    """ For each small sticker: its ordered block of 4x4 sticker indices, and its kind (corner/edge/centre). """
    _, normals, cubies = _cubie_groups(cube_len)
    points_big, _ = sticker_coords(CUBE_LEN_BIG)
    cubie_of = {i: c for c, idxs in cubies.items() for i in idxs}
    num_face: int = cube_len ** 2

    blocks: List[np.ndarray] = []
    kinds: List[str] = []
    for idx in range(6 * num_face):
        face, rc = divmod(idx, num_face)
        row, col = divmod(rc, cube_len)
        members = [face * CUBE_LEN_BIG ** 2 + r * CUBE_LEN_BIG + c for r in layer_groups[row] for c in layer_groups[col]]

        n = normals[idx]
        others = [normals[j] for j in cubies[cubie_of[idx]] if j != idx]
        if len(others) == 2:
            m1, m2 = others
            if np.cross(n, m1).dot(m2) < 0:
                m1, m2 = m2, m1
            axes = [m1, m2]
            kinds.append('corner')
        elif len(others) == 1:
            axes = [np.cross(n, others[0])]
            kinds.append('edge')
        else:
            # centre: no cubie axes exist, and none can work (see module docstring). List the 2x2 block in cyclic
            # order around the face, (r0, c0), (r0, c1), (r1, c1), (r1, c0), so that a turn of the block is a cyclic
            # shift. This listing is the canonical 0 degree turn.
            kinds.append('centre')
            (r0, r1), (c0, c1) = layer_groups[row], layer_groups[col]
            base = face * CUBE_LEN_BIG ** 2
            cyclic = [base + r * CUBE_LEN_BIG + c for r, c in [(r0, c0), (r0, c1), (r1, c1), (r1, c0)]]
            blocks.append(np.array(cyclic, dtype=int))
            continue

        members.sort(key=lambda q: tuple(points_big[q].dot(a) for a in axes))
        blocks.append(np.array(members, dtype=int))

    return blocks, kinds


class _SmallCubeValidity:
    """ Vectorised check that small-cube label arrays are reachable by the small cube's face moves. """

    def __init__(self, cube_len: int):
        _, normals, cubies = _cubie_groups(cube_len)
        num_stickers: int = 6 * cube_len ** 2
        self.num_stickers = num_stickers

        def ordered(idxs: List[int]) -> List[int]:
            # reference sticker first: U/D face if present, else F/B face; corners then right-handed
            ref = sorted(idxs, key=lambda i: (abs(normals[i][1]) == 0, abs(normals[i][2]) == 0))[0]
            rest = [i for i in idxs if i != ref]
            if len(rest) == 2 and np.cross(normals[ref], normals[rest[0]]).dot(normals[rest[1]]) < 0:
                rest = rest[::-1]
            return [ref] + rest

        corners = [ordered(v) for v in cubies.values() if len(v) == 3]
        edges = [ordered(v) for v in cubies.values() if len(v) == 2]
        self.centres = np.array([v[0] for v in cubies.values() if len(v) == 1], dtype=int)

        self.corners = np.array(corners, dtype=int).reshape(-1, 3)
        self.edges = np.array(edges, dtype=int).reshape(-1, 2)

        # label -> (home cubie, position within the home cubie's ordered list); -1 for other piece types
        self.corner_of = np.full(num_stickers, -1)
        self.corner_k = np.zeros(num_stickers, dtype=int)
        for ci, c in enumerate(self.corners):
            self.corner_of[c] = ci
            self.corner_k[c] = np.arange(3)
        self.edge_of = np.full(num_stickers, -1)
        self.edge_k = np.zeros(num_stickers, dtype=int)
        for ei, e in enumerate(self.edges):
            self.edge_of[e] = ei
            self.edge_k[e] = np.arange(2)

    @staticmethod
    def _parity(perm: np.ndarray) -> np.ndarray:
        n = perm.shape[1]
        upper = np.triu(np.ones((n, n), dtype=bool), 1)
        inversions = (perm[:, :, None] > perm[:, None, :]) & upper
        return inversions.sum(axis=(1, 2)) % 2

    def __call__(self, states: np.ndarray) -> np.ndarray:
        ok = np.all(np.sort(states, axis=1) == np.arange(self.num_stickers), axis=1)

        lab = states[:, self.corners]
        home = self.corner_of[lab]
        k = self.corner_k[lab]
        ok &= np.all((home >= 0) & (home == home[:, :, :1]), axis=(1, 2))
        ok &= np.all(((k[:, :, 1] - k[:, :, 0]) % 3 == 1) & ((k[:, :, 2] - k[:, :, 0]) % 3 == 2), axis=1)
        ok &= k[:, :, 0].sum(axis=1) % 3 == 0

        if len(self.edges) > 0:
            lab = states[:, self.edges]
            home_e = self.edge_of[lab]
            ok &= np.all((home_e >= 0) & (home_e == home_e[:, :, :1]), axis=(1, 2))
            ok &= self.edge_k[lab][:, :, 0].sum(axis=1) % 2 == 0
            ok &= self._parity(home[:, :, 0]) == self._parity(home_e[:, :, 0])
        if len(self.centres) > 0:
            ok &= np.all(states[:, self.centres] == self.centres, axis=1)

        return ok


class _BlockMap:
    def __init__(self, cube_len: int, layer_groups: List[List[int]], turnable_centres: bool):
        self.cube_len = cube_len
        self.num_small: int = 6 * cube_len ** 2
        self.turnable_centres = turnable_centres
        self.valid = _SmallCubeValidity(cube_len)
        blocks, self.kinds = _compute_blocks(cube_len, layer_groups)
        self.set_blocks(blocks)

    def set_blocks(self, blocks: List[np.ndarray]) -> None:
        """ (Re)build all index arrays from the ordered blocks. """
        self.blocks = [np.asarray(b, dtype=int) for b in blocks]
        self.size = np.array([len(b) for b in self.blocks])
        self.table = np.zeros((self.num_small, self.size.max()), dtype=np.uint8)
        for p, b in enumerate(self.blocks):
            self.table[p, :len(b)] = b

        # embed: states4[:, pos] = table[states[:, src], k]
        self.pos = np.concatenate(self.blocks)
        self.src = np.repeat(np.arange(self.num_small), self.size)
        self.k = np.concatenate([np.arange(len(b)) for b in self.blocks])

        # inverse lookups on 4x4 labels
        self.small_of = np.zeros(NUM_BIG, dtype=int)
        self.k_of = np.zeros(NUM_BIG, dtype=int)
        for p, b in enumerate(self.blocks):
            self.small_of[b] = p
            self.k_of[b] = np.arange(len(b))

        # blocks grouped by (size, turnable) for vectorised projection
        self.groups = []
        for kind in sorted(set(self.kinds)):
            ps = np.array([p for p in range(self.num_small) if self.kinds[p] == kind])
            turnable = self.turnable_centres and kind == 'centre'
            self.groups.append((ps, np.stack([self.blocks[p] for p in ps]), turnable))

        self.centre_blocks = np.stack([self.blocks[p] for p in range(self.num_small) if self.kinds[p] == 'centre']) \
            if 'centre' in self.kinds else np.zeros((0, 4), dtype=int)

    def embed(self, states: np.ndarray) -> np.ndarray:
        states_np, single = _as_batch(states, self.num_small)
        if not np.all(np.sort(states_np, axis=1) == np.arange(self.num_small)):
            raise ValueError("states must be permutations of 0..%i" % (self.num_small - 1))
        if not np.all(self.size[states_np] == self.size[None, :]):
            raise ValueError("labels must sit on positions of their own piece type (corner/edge/centre)")

        states_big = np.zeros((states_np.shape[0], NUM_BIG), dtype=np.uint8)
        states_big[:, self.pos] = self.table[states_np[:, self.src], self.k]

        return states_big[0] if single else states_big

    def project_and_check(self, states_big: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
        states_np, single = _as_batch(states_big, NUM_BIG)
        states_small = np.zeros((states_np.shape[0], self.num_small), dtype=int)
        ok = np.ones(states_np.shape[0], dtype=bool)

        for ps, blocks, turnable in self.groups:
            size = blocks.shape[1]
            lab = states_np[:, blocks].astype(int)  # (B, num_blocks, size)
            small = self.small_of[lab[:, :, 0]]
            ok &= np.all(self.small_of[lab] == small[:, :, None], axis=(1, 2))
            ok &= np.all(self.size[small] == size, axis=1)
            k = self.k_of[lab]
            if turnable:
                ok &= np.all((k - k[:, :, :1]) % size == np.arange(size), axis=(1, 2))
            else:
                ok &= np.all(k == np.arange(size), axis=(1, 2))
            states_small[:, ps] = small

        ok &= self.valid(states_small)

        return states_small, ok, single

    def is_reduced(self, states_big: np.ndarray) -> np.ndarray:
        _, ok, single = self.project_and_check(states_big)
        return ok[0] if single else ok

    def project(self, states_big: np.ndarray) -> np.ndarray:
        states_small, ok, single = self.project_and_check(states_big)
        if not np.all(ok):
            bad = np.where(~ok)[0]
            raise ValueError("%i of %i states are not reduced %ix%ix%i states (first: index %i)" %
                             (len(bad), len(ok), self.cube_len, self.cube_len, self.cube_len, bad[0]))
        states_small = states_small.astype(np.uint8)
        return states_small[0] if single else states_small


def _as_batch(states: np.ndarray, num: int) -> Tuple[np.ndarray, bool]:
    states_np = np.asarray(states)
    single = states_np.ndim == 1
    states_np = np.atleast_2d(states_np)
    if states_np.ndim != 2 or states_np.shape[1] != num:
        raise ValueError("expected states of shape (B, %i) or (%i,), got %s" % (num, num, np.asarray(states).shape))
    return states_np, single


MAP_3TO4 = _BlockMap(3, [[0], [1, 2], [3]], turnable_centres=True)
MAP_2TO4 = _BlockMap(2, [[0, 1], [2, 3]], turnable_centres=False)

# (6, 4) 4x4 indices of the six 3x3-centre blocks, each listed in cyclic order around its face
CENTRE_BLOCKS_3TO4: np.ndarray = MAP_3TO4.centre_blocks


def embed_3to4(states3: np.ndarray) -> np.ndarray:
    return MAP_3TO4.embed(states3)


def embed_2to4(states2: np.ndarray) -> np.ndarray:
    return MAP_2TO4.embed(states2)


def is_reduced_3(states4: np.ndarray) -> np.ndarray:
    return MAP_3TO4.is_reduced(states4)


def is_reduced_2(states4: np.ndarray) -> np.ndarray:
    return MAP_2TO4.is_reduced(states4)


def project_4to3(states4: np.ndarray) -> np.ndarray:
    return MAP_3TO4.project(states4)


def project_4to2(states4: np.ndarray) -> np.ndarray:
    return MAP_2TO4.project(states4)


def _canonical_centres_3(states4: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
    """ Replace each 3x3-centre block that shows a single colour c by the canonical (0 degree) labels of colour c's
    centre block. Returns the new states, whether every block was single-coloured, and whether input was 1-D. """
    states_np, single = _as_batch(states4, NUM_BIG)
    states_np = states_np.copy()
    block_colours = states_np[:, CENTRE_BLOCKS_3TO4] // 16  # (B, 6, 4)
    one_colour = np.all(block_colours == block_colours[:, :, :1], axis=(1, 2))
    # the centre block of 3x3 face c holds the labels CENTRE_BLOCKS_3TO4[c] (centre labels of colour c)
    states_np[:, CENTRE_BLOCKS_3TO4] = CENTRE_BLOCKS_3TO4[block_colours[:, :, 0]]
    return states_np, one_colour, single


def is_reduced_3_colour(states4: np.ndarray) -> np.ndarray:
    """ Colour-level reduced check: corners and edge pairs form valid 3x3x3 pieces exactly as in is_reduced_3, and
    each centre block is a single colour (same-colour 4x4 centre pieces are physically identical, so which of them
    sits where does not matter). The projected 3x3x3 state must still be valid (reachable), so the centre colours
    must match the corners and edges. """
    canonical, one_colour, single = _canonical_centres_3(states4)
    _, ok, _ = MAP_3TO4.project_and_check(canonical)
    ok &= one_colour
    return ok[0] if single else ok


def project_4to3_colour(states4: np.ndarray) -> np.ndarray:
    """ Like project_4to3, but centres are taken from the colour of each centre block. Raises ValueError on states
    that fail is_reduced_3_colour. Agrees with project_4to3 on states that pass the strict check. """
    canonical, one_colour, single = _canonical_centres_3(states4)
    states_small, ok, _ = MAP_3TO4.project_and_check(canonical)
    ok &= one_colour
    if not np.all(ok):
        bad = np.where(~ok)[0]
        raise ValueError("%i of %i states are not colour-reduced 3x3x3 states (first: index %i)" %
                         (len(bad), len(ok), bad[0]))
    states_small = states_small.astype(np.uint8)
    return states_small[0] if single else states_small


def equal_up_to_centre_turns(states_a: np.ndarray, states_b: np.ndarray) -> np.ndarray:
    """ 4x4 states equal on every label, except that each 3x3-centre block may be turned by 0/90/180/270 degrees. """
    a, single = _as_batch(states_a, NUM_BIG)
    b, _ = _as_batch(states_b, NUM_BIG)
    centre_pos = CENTRE_BLOCKS_3TO4.reshape(-1)
    other = np.setdiff1d(np.arange(NUM_BIG), centre_pos)

    eq = np.all(a[:, other] == b[:, other], axis=1)
    a_c = a[:, CENTRE_BLOCKS_3TO4]
    b_c = b[:, CENTRE_BLOCKS_3TO4]
    turned = np.stack([np.all(np.roll(a_c, r, axis=2) == b_c, axis=2) for r in range(4)], axis=0)  # (4, B, 6)
    eq &= np.all(np.any(turned, axis=0), axis=1)

    return eq[0] if single else eq
