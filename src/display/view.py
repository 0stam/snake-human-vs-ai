import pygame

from src.display.button import Button
from src.simulation.simulation import Vector2


class View:
    MOUSE_EVENTS = [
        pygame.MOUSEWHEEL,
        pygame.MOUSEMOTION,
        pygame.MOUSEBUTTONUP,
        pygame.MOUSEBUTTONDOWN
    ]

    def __init__(self) -> None:
        self.buttons: list[Button] = []

    def init_gui(self) -> None:
        pass

    def process(self, delta: float, events: list[pygame.Event]) -> None:
        pass

    def on_mouse_pressed(self, pos: Vector2) -> bool:
        for button in self.buttons:
            if button.on_mouse_pressed(pos):
                return True

        return False

    def main_loop(self) -> None:
        clock = pygame.time.Clock()

        while True:
            delta = clock.tick(60)

            if pygame.event.get(pygame.QUIT):
                return

            process_events = pygame.event.get(exclude=self.MOUSE_EVENTS)

            self.process(delta, process_events)

            mouse_events = pygame.event.get(self.MOUSE_EVENTS)

            # TODO: implement mouse presses

            pygame.display.flip()


class ParentView(View):
    def __init__(self) -> None:
        super().__init__()

        self.views: list[View] = []

    def add_view(self, view: View) -> None:
        self.views.append(view)
        view.init_gui()

    def process(self, delta: float, events: list[pygame.Event]) -> None:
        super().process(delta, events)

        for view in self.views:
            view.process(delta, events)

