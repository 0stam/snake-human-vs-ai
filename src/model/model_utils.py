import tensorflow as tf
import keras
import numpy as np

from src.simulation.simulation import Vector2


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
