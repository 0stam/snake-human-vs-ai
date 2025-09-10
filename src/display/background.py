import pygame
from src.display.view import View


class Background(View):
    def __init__(self, color: pygame.Color) -> None:
        super().__init__()

        self.color = color


    def process(self, delta: float, events: list[pygame.Event]) -> None:
        super().process(delta, events)

        pygame.display.get_surface().fill(self.color)

