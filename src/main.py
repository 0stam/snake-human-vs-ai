import random

import numpy as np
import keras
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)

from src.model.model_utils import get_rotated_state
from src.display.display import Display, GameQuit
from src.simulation.simulation import Simulation
from src.simulation.board_generator import make_simple_board


#def main() -> None:
#    simulation = Simulation(calculate_score)
#
#    states = [np.array([[0, 0, 0], [0, 5, 0], [0, 2, 0]]), np.array([[0, 0, 0], [0, 5, 2], [0, 0, 0]])]
#    moves = [(1, 0), (0, 1)]
#
#    model = keras.models.load_model("models/r1_16_16.keras")
#
#    X = np.array([get_rotated_state(state, move) for state, move in zip(states, moves)])
#
#    print(X)
#
#    print(model.predict(X))


def main() -> None:
    simulation = Simulation(calculate_score, True)

    model = keras.models.load_model("models/r7_simple_64x3_rb_1_3_e_100000_lr_001_23_snapshot.keras")
    view_type = "simple"

    display = Display()
    display.setup((1354, 1354))

    while True:
        simulation.reset(make_simple_board(np.array([15, 15])), 1, 1, 2)

        try:
            display.game_loop(simulation, model, view_type, 7, 8)
        except GameQuit:
            return


def calculate_score(ate_food: bool, died: bool) -> float:
    if died:
        return -20

    if ate_food:
        return 50

    return -1


if __name__ == "__main__":
    main()

