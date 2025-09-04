from collections.abc import Callable

from src.simulation.simulation import Vector2


class Button:
    def __init__(self, callback: Callable[[], None]) -> None:
        self.callback = callback
        self.rect_start: Vector2 = (0, 0)
        self.rect_end: Vector2 = (0, 0)

    def on_mouse_pressed(self, pos: Vector2) -> bool:
        if self.rect_start[0] <= pos[0] <= self.rect_end[0] and self.rect_start[1] <= pos[1] <= self.rect_end[1]:
            self.callback()
            return True

        return False
