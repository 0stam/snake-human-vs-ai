from pathlib import Path

import pygame
import pygame.freetype

from src.simulation.simulation import Vector2
from src.display.view import View


class Text(View):
    def __init__(self, text: str, anchor: str, offset: Vector2, font: pygame.freetype.Font, f_size: float) -> None:
        super().__init__()

        self._text = text
        self.anchor = anchor
        self.offset = offset
        self.font = font
        self.f_size = f_size

        self.text_surface: pygame.Surface = None
        self.text_rect: pygame.Rect = None
        self.screen: pygame.Surface = pygame.display.get_surface()

    def _display(self):
        self.text_surface, self.text_rect = self.font.render(self.text, (255, 255, 255), size=self.f_size)

    def init_gui(self) -> None:
        super().init_gui()

        self._display()

    def process(self, delta: float, events: list[pygame.Event]) -> None:
        super().process(delta, events)

        self.screen.blit(self.text_surface, self._calculate_pos())

    def _calculate_pos(self) -> Vector2:
        scr_x, scr_y = self.screen.size
        text_x, text_y = self.text_rect.size

        if self.anchor.startswith("top"):
            y = 0
        elif self.anchor.startswith("center"):
            y = (scr_y - text_y) // 2
        elif self.anchor.startswith("bottom"):
            y = scr_y - text_y
        else:
            raise ValueError("Usupported anchor type")

        if self.anchor.endswith("left"):
            x = 0
        elif self.anchor.endswith("center"):
            x = (scr_x - text_x) // 2
        elif self.anchor.endswith("right"):
            x = scr_x - text_x
        else:
            raise ValueError("Usupported anchor type")

        x += self.offset[0]
        y += self.offset[1]

        return x, y

    @property
    def text(self) -> str:
        return self._text
    
    @text.setter
    def text(self, val: str) -> None:
        self._text = val
        self._display()


class TextFactory:
    def __init__(self, font_path: Path, scale: float):
        self.font = pygame.freetype.Font(font_path)
        self.scale = scale

    def create(self, text: str, anchor: str, offset: Vector2, size: float) -> Text:
        scaled_offset = (round(offset[0] * self.scale), round(offset[1] * self.scale))

        return Text(text, anchor, scaled_offset, self.font, size * self.scale)
