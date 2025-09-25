from collections import deque
from typing import Callable, Optional, Sequence
from enum import Enum, IntEnum
import random
import logging

logger = logging.getLogger("death_logger")
logger.setLevel(logging.ERROR)
std_handler = logging.StreamHandler()
std_handler.setLevel(logging.DEBUG)
logger.addHandler(std_handler)

import numpy as np


class Field(IntEnum):
    EMPTY = 0
    WALL = 1
    FOOD = 2
    SNAKE_BODY = 3
    SNAKE_HEAD = 4
    OWNER_BODY = 5
    OWNER_HEAD = 6


Vector2 = tuple[int, int]


VALID_DIRECTIONS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
DIAGONAL_DIRECTIONS = [(1, 1), (-1, 1), (-1, -1), (1, -1)]


class Simulation:
    def __init__(self, calculate_score: Callable[[bool, bool, int], float], use_timeout: bool=False) -> None:
        """
        Creates the simulation object, but doesn't create any game-related data.
        To start a game, use reset()

        Args:
            calculate_score (Callable[[bool, bool, int], float]): function that gives a score for a given state with args (snake_ate_food, snake_died, n_snakes_killed)
        """
        self.calculate_score = calculate_score
        self.use_timeout = use_timeout


    def reset(self, board: np.ndarray, snake_count: int, food_count: int, tail_len: int) -> None:
        if tail_len < 1:
            raise ValueError("Usupported tail_len. It would cause errors in next()")
        self.board: np.ndarray = board
        
        self.snakes: list[deque[Vector2]] = []
        self.previous_moves: list[Vector2] = []

        self._place_snakes(snake_count, tail_len)

        self.food_count: int = food_count

        self._snakes_alive = [True for _ in range(snake_count)]
        self.n_snakes_alive = snake_count
        self.finish_on_n_snakes = 0 if snake_count == 1 else 1
        
        self._place_on_empty(Field.FOOD, food_count)

        if self.use_timeout:
            self._reset_timeout()

        logger.debug("Game started")


    def next(self, moves: Sequence[Vector2]) -> tuple[Sequence[float], bool]:
        assert len(moves) == len(self.snakes)

        ate_food: list[bool] = [False for _ in self.snakes]
        died: list[bool] = [False for _ in self.snakes]
        n_snakes_killed = [0 for _ in self.snakes]

        missing_food = 0

        # Calculate where snakes are going
        # If snake is not eating an apple, remove it's tail so snakes can go there
        for i, (snake, move) in enumerate(zip(self.snakes, moves)):
            if not snake:
                continue

            assert move != (0, 0)

            head_x, head_y = snake[0]
            move_x, move_y = move

            prev_move_x, prev_move_y = self.previous_moves[i]

            if move_x == -prev_move_x and move_y == -prev_move_y:
                print("Invalid move, using the previous one")
                move_x = prev_move_x
                move_y = prev_move_y

            self.previous_moves[i] = (move_x, move_y)

            snake.appendleft((head_x + move_x, head_y + move_y))

            self.board[snake[1]] = Field.SNAKE_BODY  # Prevent snakes killing each other when one of them extends

            if self.board[snake[0]] == Field.FOOD:
                ate_food[i] = True
            else:
                self.board[snake.pop()] = Field.EMPTY

        # Check for collision and move snakes
        for i, (snake, move) in enumerate(zip(self.snakes, moves)):
            if not snake:
                continue

            # Check if snakes kill each other
            if self.board[snake[0]] == Field.SNAKE_HEAD:
                # Kill the other snake
                for j, colliding_snake in enumerate(self.snakes):
                    if not colliding_snake or colliding_snake is snake:
                        continue

                    if colliding_snake[0] == snake[0]:
                        logger.debug(f"{j} collided with other head, moved first")
                        died[j] = True
                        # Don't stop the search here, there can be many snakes on the same field
                
                # Kill current snake
                logger.debug(f"{i} collided with other head, moved second")
                died[i] = True

                continue

            if self.board[snake[0]] == Field.FOOD:
                missing_food += 1
            elif self.board[snake[0]] != Field.EMPTY: # Check if this snake dies
                # Kill this snake
                logger.debug(f"{i} collided with {Field(self.board[snake[0]]).name}")
                died[i] = True

                if self.board[snake[0]] == Field.SNAKE_BODY:
                    for j, colliding_snake in enumerate(self.snakes):
                        if snake[0] in colliding_snake:
                            if i == j:
                                continue

                            logger.debug(f"Killer {j}")
                            n_snakes_killed[j] += 1
                            break
                    else:
                        logger.debug("Snake killed itself")

                continue
            
            # Move snake forward
            self.board[snake[0]] = Field.SNAKE_HEAD

        for i, i_died in enumerate(died):
            if i_died:
                self._clear_snake(i, True)

        self._place_on_empty(Field.FOOD, missing_food)

        game_running = self.n_snakes_alive > self.finish_on_n_snakes

        if self.use_timeout:
            if missing_food:
                self._reset_timeout()

            if self.turns_before_timeout <= 0:
                game_running = False
                logger.debug("Timeout")

            self.turns_before_timeout -= 1
        
        return [self.calculate_score(a, d, n) for a, d, n in zip(ate_food, died, n_snakes_killed)], game_running

    def get_snake_view(self, snake_idx: int, view_type: str, view_range: int = 0):
        """
        Returns the snake view of selected type

        Args:
            snake_idx (int): idx of the selected snake
            view_type (str): "centered"/"full"/"simple"
                "centered": part of the board of size view_range * 2 + 1, centered on snake's head
                "full": full board
                "simple": series of distances up to view_range in directions [right, down, left, up] to different field types. If the distance exceeds the view_range, than view_range is returned.
                "diagonal": same as simple, but the directions are [right, down, left, up, right-down, down-left, left-up, up-right]
        """

        if view_type == "full":
            return self.get_snake_full_view(snake_idx)

        if view_range <= 0:
            raise ValueError(f"Can't return {view_type} view with view_range <= 0")

        if view_type == "centered":
            return self.get_snake_centered_view(snake_idx, view_range)
        
        if view_type == "simple":
            return self.get_snake_simple_view(snake_idx, view_range)

        if view_type == "diagonal":
            return self.get_snake_simple_view(snake_idx, view_range, True)

        raise ValueError("Invalid view_type")

    def get_snake_centered_view(self, snake_idx: int, view_range: int) -> np.ndarray:
        full_board_view = self.get_snake_full_view(snake_idx)

        if full_board_view.size == 0:
            return full_board_view

        head_x, head_y = self.snakes[snake_idx][0]

        view_start_x = head_x - view_range
        view_start_y = head_y - view_range

        view_end_x = head_x + view_range + 1
        view_end_y = head_y + view_range + 1

        #print(f"Head: ({head_x}, {head_y})\nView start: ({view_start_x}, {view_start_y})\nView end: ({view_end_x}, {view_end_y})")

        view = np.full((2 * view_range + 1, 2 * view_range + 1), Field.WALL)

        start_offset_x = max(-view_start_x, 0)
        start_offset_y = max(-view_start_y, 0)

        end_offset_x = min(full_board_view.shape[0] - view_end_x - 1, 0)
        end_offset_y = min(full_board_view.shape[1] - view_end_y - 1, 0)

        selection_end_x = end_offset_x if end_offset_x else view.shape[0]
        selection_end_y = end_offset_y if end_offset_y else view.shape[1]

        #print(f"View shape: f{view.shape}\nStart offset: ({start_offset_x}, {start_offset_y})\nEnd offset: ({end_offset_x}, {end_offset_y})")

        view[start_offset_x:selection_end_x, start_offset_y:selection_end_y] = full_board_view[
            view_start_x + start_offset_x:view_end_x + end_offset_x,
            view_start_y + start_offset_y:view_end_y + end_offset_y
        ]

        return view

    def get_snake_full_view(self, snake_idx: int) -> np.ndarray:
        if not self.snakes[snake_idx]:
            return np.array([])

        full_board_view = self.board.copy()

        for pos in self.snakes[snake_idx]:
            full_board_view[pos[0], pos[1]] += 2

        return full_board_view

    def get_snake_simple_view(self, snake_idx: int, view_range: int, diagonal: bool=False) -> np.ndarray:
        if not self.snakes[snake_idx]:
            return np.array([])

        result = []

        for field_type in range(1, len(Field) - 1):  # Don't check for empty and owner's head
            for dir in VALID_DIRECTIONS:
                result.append(self._find_in_direction(
                    snake_idx,
                    Field(field_type),
                    dir,
                    view_range
                ))
            
            if diagonal:
                for dir in DIAGONAL_DIRECTIONS:
                    result.append(self._find_in_direction(
                        snake_idx,
                        Field(field_type),
                        dir,
                        view_range
                    ))

        return np.array(result)

    def get_legal_moves(self, snake_idx: int) -> list[Vector2]:
        moves = [(1, 0), (-1, 0), (0, -1), (0, 1)]

        prev_x, prev_y = self.previous_moves[snake_idx]
        bad_move = (-prev_x, -prev_y)
        return [move for move in moves if move != bad_move]

    @property
    def n_snakes(self):
        return len(self.snakes)

    @property
    def snakes_alive(self):
        return self._snakes_alive.copy()

    def _place_snakes(self, n: int, tail_len: int) -> None:
        rand_x_min = tail_len
        rand_x_max = self.board.shape[0] - tail_len

        rand_y_min = tail_len
        rand_y_max = self.board.shape[1] - tail_len

        while n > 0:
            x = random.randrange(rand_x_min, rand_x_max)
            y = random.randrange(rand_y_min, rand_y_max)

            if np.any(self.board[x-tail_len:x+tail_len+1, y-tail_len:y+tail_len+1] != Field.EMPTY):
                continue

            self.board[x, y] = Field.SNAKE_HEAD
            self.snakes.append(deque([(x, y)]))

            dir_x, dir_y = random.choice(VALID_DIRECTIONS)

            for _ in range(tail_len):
                x -= dir_x
                y -= dir_y

                self.board[x, y] = Field.SNAKE_BODY
                self.snakes[-1].append((x, y))

            self.previous_moves.append((dir_x, dir_y))

            n -= 1

    def _place_on_empty(self, field_type: Field, n: int, return_positions: bool=False) -> Optional[Sequence[Vector2]]:
        empty_x, empty_y = np.where(self.board == Field.EMPTY)

        assert len(empty_x) >= n

        selection_idx = np.random.choice(len(empty_x), size=n, replace=False)
        positions = (empty_x[selection_idx], empty_y[selection_idx])

        self.board[positions] = field_type

        if return_positions:
            return [(positions[0][i], positions[1][i]) for i in range(len(positions[0]))]

    def _clear_snake(self, idx: int, clear_head: bool=False) -> None:
        logger.debug("Clearning snake")

        snake = self.snakes[idx]

        assert snake

        self._snakes_alive[idx] = False
        self.n_snakes_alive -= 1

        snake_iter = iter(snake)

        head = next(snake_iter)

        if clear_head and self.board[head] == Field.SNAKE_HEAD:
            self.board[head] = Field.EMPTY

        while True:
            try:
                self.board[next(snake_iter)] = Field.EMPTY
            except StopIteration:
                break

        snake.clear()

    def _find_in_direction(self, snake_idx: int, field_type: Field, direction: Vector2, max_distance: int):
        board = self.get_snake_full_view(snake_idx)

        x, y = self.snakes[snake_idx][0]
        dir_x, dir_y = direction

        for distance in range(1, max_distance + 1):
            x += dir_x
            y += dir_y

            if x < 0 or x >= board.shape[0] or y < 0 or y >= board.shape[1]:
                return max_distance

            if board[x][y] == field_type:
                return distance

        return max_distance

    def _reset_timeout(self):
        self.turns_before_timeout = 80 + max(len(snake) for snake in self.snakes) * 20

