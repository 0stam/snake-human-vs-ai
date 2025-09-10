from itertools import product
from pathlib import Path
import random
import time

import pygame
from pygame.event import Event
import keras
import numpy as np

from src.display.background import Background
from src.display.text import Text, TextFactory
from src.display.constants import EVENT_GAME_FINISHED, EVENT_GAME_STARTED, EVENT_HUMAN_STARTED
from src.simulation.board_generator import make_simple_board
from src.display.view import ParentView, View
from src.display.game_view import GameView
from src.simulation.simulation import Simulation, Field, Vector2
from src.model.model_utils import get_model_moves


class Display(ParentView):
    PROMPT_TEXT = "Press any key to start"

    def init_gui(self, size: tuple[int, int]=(0,0)) -> None:
        super().init_gui()

        pygame.init()

        if size == (0, 0):
            screen_info = pygame.display.Info()
            size = (screen_info.current_w, screen_info.current_h)

            self.screen = pygame.display.set_mode(size, pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(size)

        self.size = size

        self.scale = min(size[0] / 1920, size[1] / 1080)
        print(size, self.scale)

        # Background
        self.background = Background(pygame.Color(22, 27, 22))
        self.add_view(self.background)

        # Game View
        self.game_view = GameView()
        self.add_view(self.game_view)

        # Text
        self.text_factory = TextFactory(Path("assets/fonts/tiny5/Tiny5-Regular.ttf"), self.scale)

        self.center_label: Text = self.text_factory.create(self.PROMPT_TEXT, "center", (0, 0), 80)
        self.add_view(self.center_label)

        self.bottom_label: Text = self.text_factory.create("", "bottom_center", (0, -100), 50)
        self.add_view(self.bottom_label)

        # Start demo game
        self.human_playing = False
        self._start_game()

    def process(self, delta: float, events: list[pygame.Event]) -> None:
        super().process(delta, events)

        for event in events:
            if event.type == EVENT_GAME_FINISHED:
                if self.human_playing:
                    pygame.time.set_timer(EVENT_GAME_STARTED, 2000, 1)
                else:
                    self._start_game()
                continue

            if event.type == EVENT_GAME_STARTED:
                self._start_game()
                continue

            if not self.human_playing and event.type == pygame.KEYDOWN:
                pygame.time.set_timer(EVENT_GAME_STARTED, 0)

                self.human_playing = True
                self._start_game()
                continue

            if event.type == EVENT_HUMAN_STARTED:
                self.bottom_label.text = ""

    def _start_game(self) -> None:
        calculate_score = lambda x, y: 0

        simulation = Simulation(calculate_score, not self.human_playing)

        model = keras.models.load_model("models/r7_simple_rb_1_3_e_100000_lr_001_timeout_v2_186_snapshot.keras")
        view_type = "simple"

        simulation.reset(make_simple_board(np.array([15, 15])), 1, 1, 2)

        self.game_view.setup_game(simulation, self.human_playing, model, view_type, 7, 20)

        if self.human_playing:
            self.center_label.text = ""
            self.bottom_label.text = self.PROMPT_TEXT
        else:
            self.center_label.text = self.PROMPT_TEXT
            self.bottom_label.text = ""


#    def game_loop(self, simulation: Simulation, model: keras.Model, view_type: str, snake_view_range: int, fps: float) -> None:
#        running = True
#
#        self.display(simulation)
#        last_human_move = (0, 0)
#
#        while running:
#            self.clock.tick(fps)
#
#            if pygame.event.get(pygame.QUIT):
#                raise GameQuit()
#
#            keydown_events = pygame.event.get(pygame.KEYDOWN)
#            pygame.event.pump()
#
#            for event in keydown_events:
#                if event.key == pygame.K_SPACE:
#                    running = False
#
#            last_human_move = simulation.previous_moves[0]
#            moves = self._get_moves(simulation, model, view_type, snake_view_range, True, last_human_move, keydown_events)
#
#            _, sim_running = simulation.next(moves)
#
#            #input()
#
#            running &= sim_running
#
#            self.display(simulation)




