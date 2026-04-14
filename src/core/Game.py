import chess
import pygame

from agents.RandomAgent import RandomAgent
from core.Controller import Controller
from core.GameState import GameState
from core.Model import Model
from core.View import View

DEFAULT_AGENT_CONFIG = (None, None)  ##pvp(in engine), left is white

"""
Game constructs the engine and sustains ownership over the whole VMC
"""


class Game:
    def __init__(
        self,
        init_state=GameState(),
        agents=DEFAULT_AGENT_CONFIG,
        window_size=(640, 640),
        comms=None,
    ):
        pygame.init()
        pygame.display.set_caption("basic chess view")
        model = Model(init_state, agents)
        view = View(window_size, model.get_board())
        self.controller = Controller(model, view, comms)

    def start(self):
        self.controller.play()

    def reset(self):
        self.controller = Controller()
        self.start()

    @staticmethod
    def random_game():
        Game(agents=(RandomAgent(), RandomAgent())).start()
