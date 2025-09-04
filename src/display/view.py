import pygame

from src.display.button import Button
from src.simulation.simulation import Vector2


class View:
    def __init__(self) -> None:
        self.buttons: list[Button] = []

    def process(self, delta: float, key_events: list[pygame.Event]) -> None:
        pass

    def on_mouse_pressed(self, pos: Vector2) -> bool:
        for button in self.buttons:
            if button.on_mouse_pressed(pos):
                return True

        return False
