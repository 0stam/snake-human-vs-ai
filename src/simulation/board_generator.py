from typing import Sequence

import numpy as np
from numpy._typing import ArrayLike

from src.simulation.simulation import Field, Vector2


def make_simple_board(dims: np.ndarray) -> np.ndarray:
    assert np.all(dims >= 3)

    board = np.full(dims, Field.EMPTY)

    board[[0, -1], :] = Field.WALL
    board[:, [0, -1]] = Field.WALL

    return board


if __name__ == "__main__":
    print(make_simple_board(np.array([10, 10])))
