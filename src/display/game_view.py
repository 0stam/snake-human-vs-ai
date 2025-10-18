from collections import deque
from itertools import islice

import pygame
import keras

from src.model.model_utils import get_model_moves
from src.simulation.simulation import Field, Simulation, Vector2
from src.display.view import View
from src.display.constants import EVENT_GAME_FINISHED, EVENT_GAMEPAD_LOCK_TIMEOUT, EVENT_HUMAN_STARTED, EVENT_HUMAN_TIMEOUT, EVENT_TICK, EVENT_POSSIBLE_HUMAN_TIMEOUT


class GameView(View):
    TIMEOUT_DURATION = 20_000

    COLOR_MAP = {
        Field.EMPTY: "black",
        Field.WALL: "gray",
        Field.FOOD: "purple",
        "empty_grid": pygame.Color(20, 20, 20)
    }

    SNAKE_HUE_START = 80
    SNAKE_HUE_END = 395

    SNAKE_SATURATION_HEAD = 100
    SNAKE_LIGHTNESS_HEAD = 70

    SNAKE_SATURATION_BODY = 100
    SNAKE_LIGHTNESS_BODY = 90

    AI_SATURATION_MUL = 0.18
    AI_LIGHTNESS_MUL = 0.7

    GAME_INPUT_KEYS = {
        pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d,
        pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT
    }

    GAMEPAD_DEADZONE = 0.25 ** 2  # Square for faster distance comparisons
    GAMEPAD_DEADZONE_RATIOS = [1.2, 1.3, 1.5]
    GAMEPAD_RATIO_THRESHOLDS = [0.7 ** 2, 0.5 ** 2, 0]
    GAMEPAD_AXES_HORIZONTAL = {0, 2}
    GAMEPAD_AXES_VERTICAL = {1, 3}
    GAMEPAD_OPPOSITE_LOCK_TIMEOUT = 200

    def __init__(self) -> None:
        super().__init__()

        self.simulation: Simulation = None

    def init_gui(self) -> None:
        super().init_gui()

        self.screen: pygame.Surface = pygame.display.get_surface()
        self.surface: pygame.Surface = pygame.Surface(self.screen.size, flags=pygame.SRCALPHA)
        self.size = self.screen.size


    def setup_game(self, simulation: Simulation, human_playing: bool, model: keras.Model, view_type: str, snake_view_range: int, turns_per_second: float, ticks_per_turn: int) -> None:
        self.simulation = simulation
        self.human_playing = human_playing
        self.model = model
        self.view_type = view_type
        self.snake_view_range = snake_view_range
        self.ticks_per_turn = ticks_per_turn
        self.tick_time = int(1000 / (turns_per_second * ticks_per_turn))
        self.ticks_remaining = 1

        self.previous_snakes: list[list|deque] = [list(s)[1:] for s in simulation.snakes]

        self.game_started = not human_playing
        self.input_queue: deque[Vector2] = deque(maxlen=2)

        if pygame.joystick.get_count():
            self.gamepad = pygame.Joystick(0)
        else:
            self.gamepad = None

        self.locked_gamepad_move = (0, 0)

        if self.game_started:
            pygame.time.set_timer(EVENT_TICK, self.tick_time)
        else:
            pygame.time.set_timer(EVENT_POSSIBLE_HUMAN_TIMEOUT, self.TIMEOUT_DURATION, 1)

        self.simulation_running = True

        self._display(self.simulation, 1)


    def process(self, delta: float, events: list[pygame.Event]) -> None:
        super().process(delta, events)

        # Test joystick
#        if pygame.joystick.get_count():
#            joystick = pygame.joystick.Joystick(0)
#
#            print("Axes:")
#
#            for i in range(joystick.get_numaxes()):
#                print(joystick.get_axis(i))
#
#            print("Buttons:")
#
#            for i in range(joystick.get_numbuttons()):
#                print(i, joystick.get_button(i))
#
#            print("Hats:")
#
#            for i in range(joystick.get_numhats()):
#                print(joystick.get_hat(i))
#        else:
#            print("Joystick unavailable")


        if not self.simulation:
            self.screen.blit(self.surface)
            return

        self._handle_human_moves(events, self.game_started)

        if not self.game_started:
            if self.human_playing:
                for event in events:
                    if self.input_queue:
                        self.game_started = True

                        pygame.event.post(pygame.Event(EVENT_TICK))

                        pygame.event.post(pygame.Event(EVENT_HUMAN_STARTED))
                        pygame.time.set_timer(EVENT_TICK, self.tick_time)

                    if event.type == EVENT_POSSIBLE_HUMAN_TIMEOUT:
                        pygame.event.post(pygame.Event(EVENT_HUMAN_TIMEOUT))

                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        pygame.event.post(pygame.Event(EVENT_HUMAN_TIMEOUT))
                        pygame.time.set_timer(EVENT_POSSIBLE_HUMAN_TIMEOUT, 0)

            self.screen.blit(self.surface)
            return

        for event in events:
            if event.type == EVENT_TICK:
                self.ticks_remaining -= 1

        if self.ticks_remaining > 0:
            self._display(self.simulation, (self.ticks_per_turn - self.ticks_remaining) / (self.ticks_per_turn - 1))
            self.screen.blit(self.surface)
            return

        self.ticks_remaining = self.ticks_per_turn

        moves = []

        if self.human_playing:
            if self.input_queue:
                moves.append(self.input_queue.popleft())
            else:
                moves.append(self.simulation.previous_moves[0])

        moves += self._get_ai_moves(self.human_playing)

        self.previous_snakes = [deque(s) for s in self.simulation.snakes]

        _, sim_running = self.simulation.next(moves)

        if self.human_playing and not self.simulation.snakes_alive[0]:
            sim_running = False

        if self.simulation_running ^ sim_running:
            pygame.event.post(pygame.Event(EVENT_GAME_FINISHED))

        self.simulation_running = sim_running

        self._display(self.simulation, 0)
        self.screen.blit(self.surface)

    def _display(self, simulation: Simulation, trans_perc: float) -> None:
        self.surface.fill((0, 0, 0, 0))

        n_tiles_x = simulation.board.shape[0]
        n_tiles_y = simulation.board.shape[1]

        tile_size = round(min(
            self.size[0] / n_tiles_x,
            self.size[1] / n_tiles_y
        ))

        snake_offset = round(tile_size * 0.1)
        snake_size = tile_size - 2 * snake_offset

        apple_offset = round(tile_size * 0.2)
        apple_size = tile_size - 2 * apple_offset

        image_width = tile_size * n_tiles_x
        image_height = tile_size * n_tiles_y

        margin_x = (self.size[0] - image_width) // 2
        margin_y = (self.size[1] - image_height) // 2

        self.surface.fill(
            self.COLOR_MAP[Field.EMPTY],
            (margin_x, margin_y, image_width, image_height)
        )

        snake_colors: list[tuple[pygame.Color, pygame.Color]] = []

        hue_step = (self.SNAKE_HUE_END - self.SNAKE_HUE_START) / ((len(simulation.snakes) - 1) or 1)

        for i in range(len(simulation.snakes)):
            head_color = pygame.Color(0, 0, 0)
            body_color = pygame.Color(0, 0, 0)

            hue = self.SNAKE_HUE_START + i * hue_step

            if hue > 355:
                hue -= 355

            if i == 0 or not self.human_playing:
                s_mul = 1
                l_mul = 1
            else:
                s_mul = self.AI_SATURATION_MUL
                l_mul = self.AI_LIGHTNESS_MUL

            head_color.hsva = (hue, self.SNAKE_SATURATION_HEAD * s_mul, self.SNAKE_LIGHTNESS_HEAD * l_mul, 100)
            body_color.hsva = (hue, self.SNAKE_SATURATION_BODY * s_mul, self.SNAKE_LIGHTNESS_BODY * l_mul, 100)

            snake_colors.append((head_color, body_color))

        # Draw tiles except for snakes
        for x in range(n_tiles_x):
            for y in range(n_tiles_y):
                size = tile_size
                offset_x = margin_x
                offset_y = margin_y

                if self.human_playing and (x + (1 if y % 2 else 0)) % 2:
                    pygame.draw.rect(
                        self.surface,
                        self.COLOR_MAP["empty_grid"],
                        [offset_x + x * tile_size, offset_y + y * tile_size, size, size]
                    )

                if simulation.board[x, y] == Field.FOOD:
                    size = apple_size
                    offset_x += apple_offset
                    offset_y += apple_offset

                if simulation.board[x, y] in {Field.SNAKE_BODY, Field.SNAKE_HEAD, Field.EMPTY}:
                    continue

                pygame.draw.rect(
                    self.surface,
                    self.COLOR_MAP[simulation.board[x, y]],
                    [offset_x + x * tile_size, offset_y + y * tile_size, size, size]
                )

        # Draw snakes
        for snake, previous_snake, (head_color, body_color) in zip(simulation.snakes, self.previous_snakes, snake_colors):
            if not snake:
                continue

            snake_len = len(snake)

            curr_iter = islice(snake, 1, len(snake) - 1)
            prev_iter = islice(previous_snake, 1, len(snake) - 1)

            for pos1, pos2 in zip(curr_iter, prev_iter):
                self._draw_snake_segment(pos1, pos2, margin_x, margin_y, tile_size, snake_size, snake_offset, body_color)

            # Draw tail

            prev_tail_x, prev_tail_y = previous_snake[-1]
            curr_tail_x, curr_tail_y = snake[-1]

            tail_diff_x = (curr_tail_x - prev_tail_x) * trans_perc
            tail_diff_y = (curr_tail_y - prev_tail_y) * trans_perc

            prev_tail_x += tail_diff_x
            prev_tail_y += tail_diff_y

            self._draw_snake_segment(snake[-1], (prev_tail_x, prev_tail_y), margin_x, margin_y, tile_size, snake_size, snake_offset, body_color)

            # Draw head

            prev_head_x, prev_head_y = previous_snake[0]
            curr_head_x, curr_head_y = snake[0]

            head_diff_x = (curr_head_x - prev_head_x) * trans_perc
            head_diff_y = (curr_head_y - prev_head_y) * trans_perc

            curr_head_x = prev_head_x + head_diff_x
            curr_head_y = prev_head_y + head_diff_y

            self._draw_snake_segment(previous_snake[0], (curr_head_x, curr_head_y), margin_x, margin_y, tile_size, snake_size, snake_offset, body_color)

            pygame.draw.rect(
                self.surface,
                head_color,
                [margin_x + snake_offset + curr_head_x * tile_size, margin_y + snake_offset + curr_head_y * tile_size, snake_size, snake_size]
            )

            self.surface = self.surface.convert_alpha()

    def _draw_snake_segment(self, pos1: Vector2, pos2: Vector2, margin_x: int, margin_y: int, tile_size: int, snake_size: int, snake_offset: int, color: pygame.Color):
        x1, y1 = pos1
        x2, y2 = pos2

        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))

        x_start = round(margin_x + snake_offset + x1 * tile_size)
        y_start = round(margin_y + snake_offset + y1 * tile_size)

        x_size = round(snake_size if x2 - x1 == 0 else snake_size + (x2 - x1) * (snake_size + snake_offset * 2))
        y_size = round(snake_size if y2 - y1 == 0 else snake_size + (y2 - y1) * (snake_size + snake_offset * 2))

        pygame.draw.rect(
            self.surface,
            color,
            [x_start, y_start, x_size, y_size]
        )


    def _handle_human_moves(self, events: list[pygame.Event], discard_forward: bool=True) -> None:
        actual_last_move = self.simulation.previous_moves[0]

        if self.input_queue:
            prev_move = self.input_queue[-1]
        else:
            prev_move = actual_last_move

        input_moves = []

        joystick_move = self.get_joystick_move(prev_move)

        if joystick_move != (0, 0):
            input_moves.append(joystick_move)

        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_d, pygame.K_RIGHT):
                    input_moves.append((1, 0))
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    input_moves.append((-1, 0))
                elif event.key in (pygame.K_w, pygame.K_UP):
                    input_moves.append((0, -1))
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    input_moves.append((0, 1))

            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == 1:
                    input_moves.append((1, 0))
                elif event.button == 3:
                    input_moves.append((-1, 0))
                elif event.button == 4:
                    input_moves.append((0, -1))
                elif event.button == 0:
                    input_moves.append((0, 1))

            if event.type == pygame.JOYHATMOTION:
                if event.value != (0, 0):
                    if event.value[0]:
                        input_moves.append((event.value[0], 0))
                    else:
                        input_moves.append((0, -event.value[1]))

            if event.type == EVENT_GAMEPAD_LOCK_TIMEOUT:
                self.locked_gamepad_move = (0, 0)

        for new_move in input_moves:
            if new_move[0] == -prev_move[0] and new_move[1] == -prev_move[1]:
                self.input_queue.clear()
                print("Input buffer cleared because of oposite move")

            if len(self.input_queue) < self.input_queue.maxlen:
                if discard_forward and new_move == prev_move:
                    continue

                self.input_queue.append(new_move)
                prev_move = new_move

        if self.input_queue:
            actual_next_x, actual_next_y = self.input_queue[0]

            if actual_next_x == -actual_last_move[0] and actual_next_y == -actual_last_move[1]:
                self.input_queue.clear()

    def get_joystick_move(self, last_move: Vector2) -> Vector2:
        if not self.gamepad:
            return (0, 0)

        joystick_x = 0
        joystick_y = 0

        abs_x = 0
        abs_y = 0

        for axis in self.GAMEPAD_AXES_HORIZONTAL:
            val = self.gamepad.get_axis(axis)
            abs_val = abs(val)

            if abs_val > abs_x:
                joystick_x = val
                abs_x = abs_val

        for axis in self.GAMEPAD_AXES_VERTICAL:
            val = self.gamepad.get_axis(axis)
            abs_val = abs(val)

            if abs_val > abs_y:
                joystick_y = val
                abs_y = abs_val

        len_squared = joystick_x ** 2 + joystick_y ** 2

        if len_squared <= self.GAMEPAD_DEADZONE:
            return (0, 0)

        greater, smaller = (abs_x, abs_y) if abs_x > abs_y else (abs_y, abs_x)
        
        for threshold, ratio in zip(self.GAMEPAD_RATIO_THRESHOLDS, self.GAMEPAD_DEADZONE_RATIOS):
            if len_squared >= threshold:
                if smaller != 0 and greater / smaller <= ratio:
                    return (0, 0)

        if abs_x > abs_y:
            move = (1, 0) if joystick_x > 0 else (-1, 0)
        else:
            move = (0, 1) if joystick_y > 0 else (0, -1)

        if move == self.locked_gamepad_move:
            return (0, 0)

        self.locked_gamepad_move = (-move[0], -move[1])
        pygame.time.set_timer(EVENT_GAMEPAD_LOCK_TIMEOUT, self.GAMEPAD_OPPOSITE_LOCK_TIMEOUT, 1)

        return move

    def _get_ai_moves(
        self,
        human_playing: bool,
    ) -> list[Vector2]:
        start = 1 if human_playing else 0

        states = [
            self.simulation.get_snake_view(i, self.view_type, self.snake_view_range)
            for i in range(start, self.simulation.n_snakes)
        ]

        possible_moves = [
            self.simulation.get_surviving_moves(i) for i in range(start, self.simulation.n_snakes)
        ]

        return get_model_moves(self.model, states, possible_moves, self.simulation.snakes_alive[start:], True)

