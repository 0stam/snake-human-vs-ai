from itertools import product
import random
import time

import pygame
from pygame.event import Event
import keras

from src.simulation.simulation import Simulation, Field, Vector2
from src.model.model_utils import get_model_moves


class Display:
    COLOR_MAP = {
        Field.EMPTY: "black",
        Field.WALL: "gray",
        Field.FOOD: "purple",
    }

    SNAKE_HUE_START = 40
    SNAKE_HUE_END = 240

    SNAKE_SATURATION_HEAD = 100
    SNAKE_LIGHTNESS_HEAD = 80

    SNAKE_SATURATION_BODY = 100
    SNAKE_LIGHTNESS_BODY = 90

    def __init__(self) -> None:
        pass

    def setup(self, size: tuple[int, int]=(0,0)) -> None:
        pygame.init()

        if size == (0, 0):
            screen_info = pygame.display.Info()
            size = (screen_info.current_w, screen_info.current_h)

            self.screen = pygame.display.set_mode(size, pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(size)

        self.size = size

        self.clock = pygame.time.Clock()

    def main_loop(self) -> None:
        views = []

    def game_loop(self, simulation: Simulation, model: keras.Model, view_type: str, snake_view_range: int, fps: float) -> None:
        running = True

        self.display(simulation)
        last_human_move = (0, 0)

        while running:
            self.clock.tick(fps)

            if pygame.event.get(pygame.QUIT):
                raise GameQuit()

            keydown_events = pygame.event.get(pygame.KEYDOWN)
            pygame.event.pump()

            for event in keydown_events:
                if event.key == pygame.K_SPACE:
                    running = False

            moves = self._get_moves(simulation, model, view_type, snake_view_range, False, last_human_move, keydown_events)
            last_human_move = moves[0]

            _, sim_running = simulation.next(moves)

            #input()

            running &= sim_running

            self.display(simulation)


    def display(self, simulation: Simulation) -> None:
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

        self.screen.fill(self.COLOR_MAP[Field.EMPTY])

        snake_colors: list[tuple[pygame.Color, pygame.Color]] = []

        hue_step = (self.SNAKE_HUE_END - self.SNAKE_HUE_START) / ((len(simulation.snakes) - 1) or 1)

        for i in range(len(simulation.snakes)):
            head_color = pygame.Color(0, 0, 0)
            body_color = pygame.Color(0, 0, 0)

            hue = self.SNAKE_HUE_START + i * hue_step

            head_color.hsva = (hue, self.SNAKE_SATURATION_HEAD, self.SNAKE_LIGHTNESS_HEAD, 100)
            body_color.hsva = (hue, self.SNAKE_SATURATION_BODY, self.SNAKE_LIGHTNESS_BODY, 100)

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
                    self.screen,
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
                    self.screen,
                    body_color,
                    [x_start, y_start, x_size, y_size]
                )

            pygame.draw.rect(
                self.screen,
                head_color,
                [margin_x + snake_offset + head_x * tile_size, margin_y + snake_offset + head_y * tile_size, snake_size, snake_size]
            )


        pygame.display.flip()


    def _get_moves(
        self,
        simulation: Simulation,
        model: keras.Model,
        view_type: str,
        snake_view_range: int,
        human_playing: bool,
        last_human_move: Vector2,
        key_events: list[Event]
    ) -> list[Vector2]:
        moves = []

        if human_playing:
            for event in key_events:
                if event.key in (pygame.K_d, pygame.K_RIGHT):
                    last_human_move = (1, 0)
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    last_human_move = (-1, 0)
                elif event.key in (pygame.K_w, pygame.K_UP):
                    last_human_move = (0, -1)
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    last_human_move = (0, 1)
            else:
                if last_human_move == (0, 0):
                    last_human_move = (0, 1)

            moves.append(last_human_move)

        start = 1 if human_playing else 0

        states = [simulation.get_snake_view(i, view_type, snake_view_range) for i in range(start, simulation.n_snakes)]
        possible_moves = [simulation.get_legal_moves(i) for i in range(start, simulation.n_snakes)]

        moves.extend(get_model_moves(model, states, possible_moves, simulation.snakes_alive, True))

        return moves


class GameQuit(Exception):
    pass


if __name__ == "__main__":
    display = Display()
    display.setup((500, 500))


    while True:
        display.clock.tick(10)
    
