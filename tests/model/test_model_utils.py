import numpy as np
import pytest

from src.simulation.simulation import Vector2
from src.model.model_utils import get_rotated_state


states_pre_rotation = [np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]) for _ in range(4)]
directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
states_post_rotation = [
    np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
    np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]),
    np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]]),
    np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
]



@pytest.mark.parametrize("state_pre,direction,state_post", [(states_pre_rotation[i], directions[i], states_post_rotation[i]) for i in range(len(states_pre_rotation))])
def test_get_rotated_state_2d(state_pre: np.ndarray, direction: Vector2, state_post: np.ndarray):
    assert np.all(get_rotated_state(state_pre, direction) == state_post)


def test_get_rotated_state_raises():
    with pytest.raises(ValueError):
        get_rotated_state(np.array([]), (0, 0))


one_dim_state = np.array([3, 4, 2, 1])
directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
states_post_rotation = [
    np.array([3, 4, 2 ,1]),
    np.array([2, 1, 3, 4]),
    np.array([4, 2, 1, 3]),
    np.array([1, 3, 4, 2])
]

@pytest.mark.parametrize("state_pre,direction,state_post", [(one_dim_state, directions[i], states_post_rotation[i]) for i in range(len(states_post_rotation))])
def test_get_rotated_state_1d(state_pre: np.ndarray, direction: Vector2, state_post: np.ndarray):
    rotated_state = get_rotated_state(state_pre, direction)

    print(rotated_state)

    assert np.all(rotated_state == state_post)

