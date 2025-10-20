from itertools import product
from pathlib import Path
import random
import time
from math import ceil

import pygame
from pygame.event import Event
import keras
import numpy as np

from src.model.litert_wrapper import LiteRTWrapper
from src.display.background import Background
from src.display.text import Text, TextFactory
from src.display.constants import EVENT_GAME_FINISHED, EVENT_GAME_STARTED, EVENT_HUMAN_STARTED, EVENT_HUMAN_TIMEOUT
from src.simulation.board_generator import make_simple_board
from src.display.view import ParentView, View
from src.display.game_view import GameView
from src.simulation.simulation import Simulation, Field, Vector2
from src.model.model_utils import get_model_moves


class Display(ParentView):
    PROMPT_IDLE = "Press any key to start"
    PROMPT_MOVE_KEYBOARD = "WSAD / Arrows to move"
    PROMPT_MOVE_GAMEPAD = "D-PAD / Joystick to move"

    PROMPT_PLAYER_COLOR = "Player is green"

    CHR_FILLED_POINT = u"\u25CF"
    CHR_EMPTY_POINT = u"\u25CB"

    HUMAN_SCORED = "You won!"
    AI_SCORED = "AI won!"
    DRAW = "Draw!"

    HUMAN_WON = "You're still better than AI!"
    AI_WON = "AI took your job!"

    NEW_GAME_WAIT_TIME = 2000

    GAME_STARTING_INPUTS = {pygame.KEYDOWN, pygame.JOYBUTTONDOWN, pygame.JOYHATMOTION}

    def __init__(self, model_config: dict[str, dict]) -> None:
        super().__init__()

        self.model_config = model_config

        # TODO: implement file loading
        self.total_human_score = 0
        self.total_ai_score = 0

        self.curr_human_score = 0
        self.curr_ai_score = 0

        self.max_curr_score = 1

        self.ai_model_name = "demo"
        self.curr_model_name = "normal"
        self.model: LiteRTWrapper = None
        self.loaded_model_filename = ""

    def init_gui(self, size: tuple[int, int]=(0,0)) -> None:
        super().init_gui()

        pygame.init()

        if size == (0, 0):
            screen_info = pygame.display.Info()
            size = (screen_info.current_w, screen_info.current_h)

            self.screen = pygame.display.set_mode(size, pygame.FULLSCREEN, vsync=1)
        else:
            self.screen = pygame.display.set_mode(size, vsync=1)

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

        self.center_label: Text = self.text_factory.create(self.PROMPT_IDLE, "center", (0, 0), 80)
        self.add_view(self.center_label)

        self.bottom_label: Text = self.text_factory.create("", "bottom_center", (0, -100), 50)
        self.add_view(self.bottom_label)

        self.bottom_label_above: Text = self.text_factory.create("", "bottom_center", (0, -160), 50)
        self.bottom_label_above.color = (153, 229, 0)
        self.add_view(self.bottom_label_above)

        self.human_score_label: Text = self.text_factory.create("", "top_left", (10, 10), 50)
        self.add_view(self.human_score_label)

        self.human_score_label_lower: Text = self.text_factory.create("", "top_left", (10, 60), 60)
        self.add_view(self.human_score_label_lower)

        self.ai_score_label: Text = self.text_factory.create("", "top_right", (-10, 10), 50)
        self.add_view(self.ai_score_label)

        self.ai_score_label_lower: Text = self.text_factory.create("", "top_right", (-10, 60), 60)
        self.add_view(self.ai_score_label_lower)

        # Start demo game
        self.human_playing = False
        self._start_game()

    def process(self, delta: float, events: list[pygame.Event]) -> None:
        super().process(delta, events)

        for event in events:
            if event.type == EVENT_GAME_FINISHED:
                if self.human_playing:
                    pygame.time.set_timer(EVENT_GAME_STARTED, self.NEW_GAME_WAIT_TIME, 1)

                    simulation = self.game_view.simulation
                    assert simulation

                    if simulation.snakes_alive[0]:
                        self._human_won()
                    elif simulation.n_snakes_alive > 0:
                        self._ai_won()
                    else:
                        self._draw()
                else:
                    pygame.time.set_timer(EVENT_GAME_STARTED, 1500, 1)
                continue

            if event.type == EVENT_GAME_STARTED:
                self._start_game()
                continue

            if not self.human_playing and (event.type in self.GAME_STARTING_INPUTS):
                self.human_playing = True

                self._reset_curr_score()
                self._start_game()
                continue

            if event.type == EVENT_HUMAN_STARTED:
                self.bottom_label.text = ""

            if event.type == EVENT_HUMAN_TIMEOUT:
                self.human_playing = False
                self._start_game()

    def _start_game(self) -> None:
        if self.human_playing:
            config = self.model_config[self.curr_model_name]
        else:
            config = self.model_config[self.ai_model_name]

        pygame.time.set_timer(EVENT_GAME_STARTED, 0)

        calculate_score = lambda x, y, z, p, q, r: 0

        simulation = Simulation(calculate_score, not self.human_playing)

        max_batch_size = config["snake_count"] * 3

        if self.loaded_model_filename != config["file"]:
            self.model = LiteRTWrapper(Path("assets/models") / config["file"], max_batch_size)
            self.loaded_model_filename = config["file"]
        else:
            self.model.set_max_batch_size(max_batch_size)

        view_type = config["view_type"]
        view_range = config["view_range"]
        fps = config["fps"]

        self.max_curr_score = 2

        simulation.reset(make_simple_board(np.array([15, 15])), config["snake_count"], config["food_count"], 2)

        self.game_view.setup_game(simulation, self.human_playing, self.model, view_type, view_range, fps, ceil(config["tps"] / fps))

        if self.human_playing:
            self.center_label.text = ""

            if pygame.joystick.get_count():
                self.bottom_label.text = self.PROMPT_MOVE_GAMEPAD
            else:
                self.bottom_label.text = self.PROMPT_MOVE_KEYBOARD
            self.bottom_label_above.text = self.PROMPT_PLAYER_COLOR
        else:
            self.center_label.text = self.PROMPT_IDLE
            self.bottom_label.text = ""
            self.bottom_label_above.text = ""
            self._update_score_labels(False)

    def _reset_curr_score(self):
        self.curr_human_score = 0
        self.curr_ai_score = 0

        self._update_score_labels(True)

    def _human_won(self):
        self.curr_human_score += 1
        self._update_score_labels(True)

        if self.curr_human_score >= self.max_curr_score:
            self.total_human_score += 1
            self.center_label.text = self.HUMAN_WON
            pygame.time.set_timer(EVENT_HUMAN_TIMEOUT, self.NEW_GAME_WAIT_TIME, 1)
            pygame.time.set_timer(EVENT_GAME_STARTED, 0)
        else:
            self.center_label.text = self.HUMAN_SCORED

    def _ai_won(self):
        self.curr_ai_score += 1
        self._update_score_labels(True)
        
        if self.curr_ai_score >= self.max_curr_score:
            self.total_ai_score += 1
            self.center_label.text = self.AI_WON
            pygame.time.set_timer(EVENT_HUMAN_TIMEOUT, self.NEW_GAME_WAIT_TIME, 1)
            pygame.time.set_timer(EVENT_GAME_STARTED, 0)
        else:
            self.center_label.text = self.AI_SCORED

    def _draw(self):
        self.center_label.text = self.DRAW

    def _update_score_labels(self, current: bool):
        if current:
            human_sc = self.curr_human_score
            ai_sc = self.curr_ai_score

            self.human_score_label.text = "Human"
            self.human_score_label_lower.text = self.CHR_FILLED_POINT * human_sc + self.CHR_EMPTY_POINT * (self.max_curr_score - human_sc)

            self.ai_score_label.text = "AI"
            self.ai_score_label_lower.text = self.CHR_EMPTY_POINT * (self.max_curr_score - ai_sc) + self.CHR_FILLED_POINT * ai_sc 
        else:
            self.human_score_label.text = f"Human: {self.total_human_score}"
            self.ai_score_label.text = f"{self.total_ai_score} :AI"

            self.human_score_label_lower.text = ""
            self.ai_score_label_lower.text = ""

