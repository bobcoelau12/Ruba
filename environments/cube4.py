from typing import List, Dict, Tuple, Union
import numpy as np
from torch import nn
from random import randrange

from utils.pytorch_models import ResnetModel
from .environment_abstract import Environment, State


# Sticker layout shared with Cube3: index = face * cube_len**2 + row * cube_len + col, faces ordered U, D, L, R, B, F.
# Each face is described in 3-D (x right, y up, z towards the front) by its outward normal and the directions in
# which its row and column indices increase. These frames are the unique ones under which the geometric
# construction in compute_rotation_idxs reproduces Cube3's permutations exactly for cube_len=3
# (checked in tests/test_cube4.py), so outer move i of this environment is the same move as Cube3 move i.
FACE_ORDER: List[str] = ['U', 'D', 'L', 'R', 'B', 'F']
FACE_FRAMES: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]] = {
    #     normal       row axis     col axis
    'U': ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    'D': ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    'L': ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    'R': ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    'B': ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
    'F': ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
}


def compute_rotation_idxs(cube_len: int, moves: List[str]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """ Compute sticker permutations for layer moves from 3-D geometry.

    A move is named <face><sign>, e.g. "U1" or "U-1". An upper-case face letter turns the outer layer of that face,
    a lower-case letter the layer one deeper (e.g. "u1"). Sign 1 is a clockwise quarter turn when looking at the face,
    -1 is anticlockwise (same as Cube3).

    Stickers are placed in doubled integer coordinates: the face plane sits at distance cube_len from the centre and
    sticker (row, col) sits at offset 2 * row - (cube_len - 1) along the row axis (same for col). The cubie a sticker
    belongs to is the sticker point minus its face normal. A layer move rotates the stickers of every cubie in that
    layer by 90 degrees about the face normal.

    @return: rotate_idxs_new, rotate_idxs_old, used as states_next[:, new] = states[:, old]
    """
    num_face: int = cube_len ** 2

    # sticker index <-> 3-D point
    points: np.ndarray = np.zeros((6 * num_face, 3), dtype=int)
    sticker_normals: np.ndarray = np.zeros((6 * num_face, 3), dtype=int)
    for face_idx, face in enumerate(FACE_ORDER):
        normal, row_axis, col_axis = (np.array(v) for v in FACE_FRAMES[face])
        for row in range(cube_len):
            for col in range(cube_len):
                idx: int = face_idx * num_face + row * cube_len + col
                points[idx] = (cube_len * normal + (2 * row - (cube_len - 1)) * row_axis +
                               (2 * col - (cube_len - 1)) * col_axis)
                sticker_normals[idx] = normal
    point_to_idx: Dict[Tuple[int, ...], int] = {tuple(p): i for i, p in enumerate(points)}
    cubies: np.ndarray = points - sticker_normals

    rotate_idxs_new: Dict[str, np.ndarray] = dict()
    rotate_idxs_old: Dict[str, np.ndarray] = dict()
    for move in moves:
        face: str = move[0].upper()
        depth: int = 0 if move[0].isupper() else 1
        sign: int = int(move[1:])
        normal: np.ndarray = np.array(FACE_FRAMES[face][0])

        # clockwise when looking at the face = rotation by -90 degrees about the outward normal
        rot_sign: int = -sign
        in_layer: np.ndarray = cubies.dot(normal) == (cube_len - 1) - 2 * depth
        idxs_old: np.ndarray = np.where(in_layer)[0]

        p: np.ndarray = points[idxs_old]
        p_rot: np.ndarray = (np.outer(p.dot(normal), normal) + rot_sign * np.cross(normal, p))
        idxs_new: np.ndarray = np.array([point_to_idx[tuple(x)] for x in p_rot], dtype=int)

        rotate_idxs_new[move] = idxs_new
        rotate_idxs_old[move] = idxs_old

    return rotate_idxs_new, rotate_idxs_old


class Cube4State(State):
    __slots__ = ['colors', 'hash']

    def __init__(self, colors: np.ndarray):
        self.colors: np.ndarray = colors
        self.hash = None

    def __hash__(self):
        if self.hash is None:
            self.hash = hash(self.colors.tobytes())

        return self.hash

    def __eq__(self, other):
        return np.array_equal(self.colors, other.colors)


class Cube4(Environment):
    # 0-11: outer layer moves, same names and order as Cube3. 12-23: inner layer (depth 1) moves.
    moves: List[str] = (["%s%i" % (f, n) for f in FACE_ORDER for n in [-1, 1]] +
                        ["%s%i" % (f.lower(), n) for f in FACE_ORDER for n in [-1, 1]])
    moves_rev: List[str] = (["%s%i" % (f, n) for f in FACE_ORDER for n in [1, -1]] +
                            ["%s%i" % (f.lower(), n) for f in FACE_ORDER for n in [1, -1]])

    def __init__(self):
        super().__init__()
        self.dtype = np.uint8
        self.cube_len = 4

        # solved state
        self.goal_colors: np.ndarray = np.arange(0, (self.cube_len ** 2) * 6, 1, dtype=self.dtype)

        # get idxs changed for moves
        self.rotate_idxs_new: Dict[str, np.ndarray]
        self.rotate_idxs_old: Dict[str, np.ndarray]

        self.rotate_idxs_new, self.rotate_idxs_old = compute_rotation_idxs(self.cube_len, self.moves)

    def next_state(self, states: List[Cube4State], action: int) -> Tuple[List[Cube4State], List[float]]:
        states_np = np.stack([x.colors for x in states], axis=0)
        states_next_np, transition_costs = self._move_np(states_np, action)

        states_next: List[Cube4State] = [Cube4State(x) for x in list(states_next_np)]

        return states_next, transition_costs

    def prev_state(self, states: List[Cube4State], action: int) -> List[Cube4State]:
        move: str = self.moves[action]
        move_rev_idx: int = np.where(np.array(self.moves_rev) == np.array(move))[0][0]

        return self.next_state(states, move_rev_idx)[0]

    def generate_goal_states(self, num_states: int, np_format: bool = False) -> Union[List[Cube4State], np.ndarray]:
        if np_format:
            goal_np: np.ndarray = np.expand_dims(self.goal_colors.copy(), 0)
            solved_states: np.ndarray = np.repeat(goal_np, num_states, axis=0)
        else:
            solved_states: List[Cube4State] = [Cube4State(self.goal_colors.copy()) for _ in range(num_states)]

        return solved_states

    def is_solved(self, states: List[Cube4State]) -> np.ndarray:
        # no fixed centres on a 4x4: solved when every face shows a single colour, in any orientation
        states_np = np.stack([state.colors for state in states], axis=0)
        face_colors: np.ndarray = (states_np // (self.cube_len ** 2)).reshape(-1, 6, self.cube_len ** 2)
        is_equal = np.equal(face_colors, face_colors[:, :, :1])

        return np.all(is_equal, axis=(1, 2))

    def state_to_nnet_input(self, states: List[Cube4State]) -> List[np.ndarray]:
        states_np = np.stack([state.colors for state in states], axis=0)

        representation_np: np.ndarray = states_np / (self.cube_len ** 2)
        representation_np: np.ndarray = representation_np.astype(self.dtype)

        representation: List[np.ndarray] = [representation_np]

        return representation

    def get_num_moves(self) -> int:
        return len(self.moves)

    def get_nnet_model(self) -> nn.Module:
        state_dim: int = (self.cube_len ** 2) * 6
        nnet = ResnetModel(state_dim, 6, 5000, 1000, 4, 1, True)

        return nnet

    def generate_states(self, num_states: int, backwards_range: Tuple[int, int]) -> Tuple[List[Cube4State], List[int]]:
        assert (num_states > 0)
        assert (backwards_range[0] >= 0)
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # Initialize
        scrambs: List[int] = list(range(backwards_range[0], backwards_range[1] + 1))
        num_env_moves: int = self.get_num_moves()

        # Get goal states
        states_np: np.ndarray = self.generate_goal_states(num_states, np_format=True)

        # Scrambles
        scramble_nums: np.array = np.random.choice(scrambs, num_states)
        num_back_moves: np.array = np.zeros(num_states)

        # Go backward from goal state
        moves_lt = num_back_moves < scramble_nums
        while np.any(moves_lt):
            idxs: np.ndarray = np.where(moves_lt)[0]
            subset_size: int = int(max(len(idxs) / num_env_moves, 1))
            idxs: np.ndarray = np.random.choice(idxs, subset_size)

            move: int = randrange(num_env_moves)
            states_np[idxs], _ = self._move_np(states_np[idxs], move)

            num_back_moves[idxs] = num_back_moves[idxs] + 1
            moves_lt[idxs] = num_back_moves[idxs] < scramble_nums[idxs]

        states: List[Cube4State] = [Cube4State(x) for x in list(states_np)]

        return states, scramble_nums.tolist()

    def expand(self, states: List[State]) -> Tuple[List[List[State]], List[np.ndarray]]:
        assert self.fixed_actions, "Environments without fixed actions must implement their own method"

        # initialize
        num_states: int = len(states)
        num_env_moves: int = self.get_num_moves()

        states_exp: List[List[State]] = [[] for _ in range(len(states))]

        tc: np.ndarray = np.empty([num_states, num_env_moves])

        # numpy states
        states_np: np.ndarray = np.stack([state.colors for state in states])

        # for each move, get next states, transition costs, and if solved
        move_idx: int
        move: int
        for move_idx in range(num_env_moves):
            # next state
            states_next_np: np.ndarray
            tc_move: List[float]
            states_next_np, tc_move = self._move_np(states_np, move_idx)

            # transition cost
            tc[:, move_idx] = np.array(tc_move)

            for idx in range(len(states)):
                states_exp[idx].append(Cube4State(states_next_np[idx]))

        # make lists
        tc_l: List[np.ndarray] = [tc[i] for i in range(num_states)]

        return states_exp, tc_l

    def _move_np(self, states_np: np.ndarray, action: int):
        action_str: str = self.moves[action]

        states_next_np: np.ndarray = states_np.copy()
        states_next_np[:, self.rotate_idxs_new[action_str]] = states_np[:, self.rotate_idxs_old[action_str]]

        transition_costs: List[float] = [1.0 for _ in range(states_np.shape[0])]

        return states_next_np, transition_costs
