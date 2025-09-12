from collections import deque

import pygame
import keras

from src.model.model_utils import get_model_moves
from src.simulation.simulation import Field, Simulation, Vector2
from src.display.view import View
from src.display.constants import EVENT_GAME_FINISHED, EVENT_HUMAN_STARTED, EVENT_HUMAN_TIMEOUT, EVENT_NEXT_TURN, EVENT_POSSIBLE_HUMAN_TIMEOUT


class GameView(View):
    TIMEOUT_DURATION = 20_000

    COLOR_MAP = {
        Field.EMPTY: "black",
        Field.WALL: "gray",
        Field.FOOD: "purple",
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

    def __init__(self) -> None:
        super().__init__()

        self.simulation = None

    def init_gui(self) -> None:
        super().init_gui()

        self.screen: pygame.Surface = pygame.display.get_surface()
        self.surface: pygame.Surface = pygame.Surface(self.screen.size, flags=pygame.SRCALPHA)
        self.size = self.screen.size


    def setup_game(self, simulation: Simulation, human_playing: bool, model: keras.Model, view_type: str, snake_view_range: int, fps: float) -> None:
        self.simulation = simulation
        self.human_playing = human_playing
        self.model = model
        self.view_type = view_type
        self.snake_view_range = snake_view_range
        self.frame_time = int(1000 / fps)

        self.game_started = not human_playing
        self.input_queue: deque[Vector2] = deque(maxlen=2)

        if self.game_started:
            pygame.time.set_timer(EVENT_NEXT_TURN, self.frame_time)
        else:
            pygame.time.set_timer(EVENT_POSSIBLE_HUMAN_TIMEOUT, self.TIMEOUT_DURATION)

        self.simulation_running = True

        self._display(self.simulation)


    def process(self, delta: float, events: list[pygame.Event]) -> None:
        super().process(delta, events)

        if not self.simulation:
            self.screen.blit(self.surface)
            return

        if not self.game_started:
            if self.human_playing:
                if any(
                    event.type == pygame.KEYDOWN and event.key in self.GAME_INPUT_KEYS
                    for event in events
                ):
                    self.game_started = True
                    self._handle_human_moves(events)

                    pygame.event.post(pygame.Event(EVENT_NEXT_TURN))
                    pygame.event.post(pygame.Event(EVENT_HUMAN_STARTED))
                    pygame.time.set_timer(EVENT_NEXT_TURN, self.frame_time)

                if any(event.type == EVENT_POSSIBLE_HUMAN_TIMEOUT for event in events):
                    pygame.event.post(pygame.Event(EVENT_HUMAN_TIMEOUT))

            self.screen.blit(self.surface)
            return

        self._handle_human_moves(events)

        if not any(event.type == EVENT_NEXT_TURN for event in events):
            self.screen.blit(self.surface)
            return

        moves = []

        if self.human_playing:
            if self.input_queue:
                moves.append(self.input_queue.popleft())
            else:
                moves.append(self.simulation.previous_moves[0])

        moves += self._get_ai_moves(self.human_playing)

        _, sim_running = self.simulation.next(moves)

        if self.human_playing and not self.simulation.snakes_alive[0]:
            sim_running = False

        if self.simulation_running ^ sim_running:
            pygame.event.post(pygame.Event(EVENT_GAME_FINISHED))

        self.simulation_running = sim_running

        self._display(self.simulation)
        self.screen.blit(self.surface)

    def _display(self, simulation) -> None:
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

                if simulation.board[x, y] == Field.FOOD:
                    size = apple_size
                    offset_x += apple_offset
                    offset_y += apple_offset

                pygame.draw.rect(
                    self.surface,
                    self.COLOR_MAP.get(simulation.board[x, y], self.COLOR_MAP[Field.EMPTY]),
                    [offset_x + x * tile_size, offset_y + y * tile_size, size, size]
                )

        # Draw snakes
        for snake, (head_color, body_color) in zip(simulation.snakes, snake_colors):
            if not snake:
                continue

            offset_iter = iter(snake)
            head_x, head_y = next(offset_iter)

            for (x1, y1), (x2, y2) in zip(snake, offset_iter):
                x1, x2 = sorted((x1, x2))
                y1, y2 = sorted((y1, y2))

                x_start = margin_x + snake_offset + x1 * tile_size
                y_start = margin_y + snake_offset + y1 * tile_size

                x_size = snake_size if x2 - x1 == 0 else 2 * (snake_size + snake_offset)
                y_size = snake_size if y2 - y1 == 0 else 2 * (snake_size + snake_offset)

                pygame.draw.rect(
                    self.surface,
                    body_color,
                    [x_start, y_start, x_size, y_size]
                )

            pygame.draw.rect(
                self.surface,
                head_color,
                [margin_x + snake_offset + head_x * tile_size, margin_y + snake_offset + head_y * tile_size, snake_size, snake_size]
            )

    def _handle_human_moves(self, events: list[pygame.Event]) -> None:
        actual_last_move = self.simulation.previous_moves[0]

        if self.input_queue:
            prev_move = self.input_queue[-1]
        else:
            prev_move = actual_last_move

        for event in events:
            if event.type != pygame.KEYDOWN:
                continue

            new_move = (0, 0)
            
            if event.key in (pygame.K_d, pygame.K_RIGHT):
                new_move = (1, 0)
            elif event.key in (pygame.K_a, pygame.K_LEFT):
                new_move = (-1, 0)
            elif event.key in (pygame.K_w, pygame.K_UP):
                new_move = (0, -1)
            elif event.key in (pygame.K_s, pygame.K_DOWN):
                new_move = (0, 1)

            if new_move == (0, 0):
                continue

            if new_move[0] == -prev_move[0] and new_move[1] == -prev_move[1]:
                self.input_queue.clear()
                self.input_queue.append(new_move)

            if len(self.input_queue) < self.input_queue.maxlen:
                self.input_queue.append(new_move)
                prev_move = new_move

        if self.input_queue:
            actual_next_x, actual_next_y = self.input_queue[0]

            if actual_next_x == -actual_last_move[0] and actual_next_y == -actual_last_move[1]:
                self.input_queue.clear()

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
            self.simulation.get_legal_moves(i) for i in range(start, self.simulation.n_snakes)
        ]

        return get_model_moves(self.model, states, possible_moves, self.simulation.snakes_alive[start:], True)

