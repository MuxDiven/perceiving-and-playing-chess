import time
from collections import deque

import chess
import pygame

from agents.CameraAgent import CameraAgent
from core.GameState import GameState

FPS = 60
DEFAULT_AGENT_CONFIG = (None, None)  ##pvp, left is white

"""
Model keeps track of the game of chess in the engine, being enacted upon and mutated by the Controller
"""


class Model:
    def __init__(self, game_state=GameState(), agents=DEFAULT_AGENT_CONFIG):
        if len(agents) != 2:
            raise ValueError("Too many or not enough agents exception")
        self.current_state = game_state
        self.stack = deque()
        self.agents = agents  ## none for player 0 -> black, 1 -> white
        self.is_camera = False

    def current_player(self):
        return self.agents[
            int(not self.get_turn())
        ]  ## chess.WHITE == True, player 1 is index 0

    def player_move(self, move):
        if move not in self.current_state.board.legal_moves:
            return None

        self.stack.appendleft(self.current_state)
        self.current_state = self.current_state.move(move)

    def agent_move(self, call=None):
        agent = self.current_player()

        if agent is None:
            raise ValueError("problem in turn control flow")

        self.is_camera = agent.__class__ == CameraAgent

        if self.is_camera and agent.it_frame is None:
            agent.it_frame = call

        start = time.time()
        self.stack.appendleft(self.current_state)
        self.current_state = agent.move(self.current_state)
        print(f"time taken to choose move: ~{int(time.time() - start)} seconds")

    def get_board(self):
        return self.current_state.board

    def get_turn(self):
        return self.get_board().turn

    def turn_str(self):
        return "WHITE" if self.get_turn() == chess.WHITE else "BLACK"
