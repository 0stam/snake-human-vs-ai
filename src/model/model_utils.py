from collections import deque
from math import floor, ceil

import tensorflow as tf
import keras
import numpy as np

from src.simulation.simulation import Vector2


class ReplayBuffer:
    def __init__(
        self,
        priority_size: int,
        normal_size: int,
        default_split: float = 0.2,
        default_batch_size: int = 32,
        threshold_decay: float = 0.3,
        threshold_multiplier: float = 1.3
    ):
        self.priority_buffer = deque(maxlen=priority_size)
        self.normal_buffer = deque(maxlen=normal_size)

        self.default_split = default_split
        self.default_batch_size = default_batch_size

        self.loss_threshold = 0
        self.threshold_decay = threshold_decay
        self.threshold_multiplier = threshold_multiplier

        self.last_priority_idxs = []
        self.last_normal_idxs = []

    def append(self, experience: tuple):
        self.normal_buffer.append(experience)

    def sample(self, split: float = -1, batch_size: int = -1):
        if split < 1:
            split = self.default_split

        if batch_size < 1:
            batch_size = self.default_batch_size

        if len(self.priority_buffer) < batch_size * split:
            split = 0

        n_priority = ceil(split * batch_size)
        n_normal = floor((1 - split) * batch_size)

        batch = []

        if n_priority:
            idxs = np.random.randint(len(self.priority_buffer), size=n_priority)
            batch += [self.priority_buffer[i] for i in idxs]
            self.last_priority_idxs = idxs
        else:
            self.last_priority_idxs = []

        if n_normal:
            idxs = np.random.randint(len(self.normal_buffer), size=n_normal)
            batch += [self.normal_buffer[i] for i in idxs]
            self.last_normal_idxs = idxs
        else:
            self.last_normal_idxs = []
            
        return [
            [experience[field_idx] for experience in batch]
            for field_idx in range(6)
        ]

    def update_loss(self, losses: tf.Tensor, mean_loss: tf.Tensor):
        self.loss_threshold *= 1 - self.threshold_decay
        self.loss_threshold += self.threshold_decay * mean_loss

        for i, priority_idx in enumerate(self.last_priority_idxs):
            if losses[i] < self.loss_threshold * self.threshold_multiplier:
                if priority_idx != -1:
                    del self.priority_buffer[priority_idx]

                self.last_priority_idxs[self.last_priority_idxs == priority_idx] = -1
                self.last_priority_idxs[self.last_priority_idxs > priority_idx] -= 1

        for i, normal_idx in enumerate(self.last_normal_idxs, start=len(self.last_priority_idxs)):
            if losses[i] > self.loss_threshold * self.threshold_multiplier:
                self.priority_buffer.append(self.normal_buffer[normal_idx])

    def clear(self):
        self.priority_buffer.clear()
        self.normal_buffer.clear()

        self.loss_threshold = 0

        self.last_normal_idxs = []
        self.last_priority_idxs = []


# By default model faces right
def get_rotated_state(state: np.ndarray, move: Vector2):
    """
    Get state assuming the model faces in the direction specified by move.
    By default model faces right.

    If the input is 2D, it just rotates the view.

    If the input is 1D, it assumes that the input contains a series of four-element tuples with distances in directions:
        [RIGHT, DOWN, LEFT, UP]
    and changes the order inside each tuple.
    """
    if move == (0, 1):
        k = -1
    elif move == (-1, 0):
        k = 2
    elif move == (0, -1):
        k = 1
    elif move == (1, 0):
        k = 0
    else:
        raise ValueError(f"Invalid move: {move}")

    if state.ndim == 1:
        if state.size % 4 != 0:
            raise ValueError("Invalid 1D state passed to get_rotated_state")

        return np.reshape(np.roll(np.reshape(state, (-1, 4)), k, axis=1), -1)
    elif state.ndim == 2:
        return np.rot90(state, k)

    raise ValueError("State has more than two dimensions")


def get_model_moves(model: keras.Model, snakes_states: list[np.ndarray], snakes_possible_moves: list[list[Vector2]], process_mask: list[bool], softmax=False):
    return [prediction[0] for prediction in get_model_predictions(model, snakes_states, snakes_possible_moves, process_mask, softmax)]


def get_model_scores(model: keras.Model, snakes_states: list[np.ndarray], snakes_possible_moves: list[list[Vector2]], process_mask: list[bool], softmax=False):
    return [prediction[1] for prediction in get_model_predictions(model, snakes_states, snakes_possible_moves, process_mask, softmax)]
    
    
def get_model_predictions(model: keras.Model, snakes_states: list[np.ndarray], snakes_possible_moves: list[list[Vector2]], process_mask: list[bool], softmax=False):
    EMPTY_VAL = -2137  # -inf gives errors in some places

    to_be_batched = [
        get_rotated_state(single_state, move)
        for single_state, single_possible_moves, process in zip(snakes_states, snakes_possible_moves, process_mask)
        for move in single_possible_moves
        if process
    ]

    if not to_be_batched:
        return [((0, 0), EMPTY_VAL) for _ in range(len(snakes_states))]
    
    batch = np.stack(to_be_batched)

    scores = model.predict(batch, verbose=0)[:, 0]

    result = []

    i = 0

    for single_possible_moves, process in zip(snakes_possible_moves, process_mask):
        n_legal_moves = len(single_possible_moves)

        if not process:
            result.append(((0, 0), EMPTY_VAL))
            continue

#        print("Choosing moves:")
#        print(single_possible_moves)
#        print(scores[i:i+n_legal_moves])

        curr_scores = scores[i:i+n_legal_moves]
        
        if softmax:
            max_score = np.max(curr_scores)
            exp_shifted = np.exp(curr_scores - max_score)
            probas = exp_shifted / np.sum(exp_shifted)
            
            best_move_idx = np.random.choice(len(curr_scores), size=1, p=probas)[0]
        else:
            best_move_idx = np.argmax(curr_scores)

        best_score = curr_scores[best_move_idx]
        
        result.append((single_possible_moves[best_move_idx], best_score))

#        print(result[-1])

        i += n_legal_moves

    return result
