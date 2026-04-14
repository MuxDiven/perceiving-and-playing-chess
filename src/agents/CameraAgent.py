import sys
from time import sleep

import chess
import pygame


class CameraAgent:
    def __init__(self, comms):
        self.comms = comms
        self.it_frame = None

    def move(self, game_state):
        ns = game_state.next_states()  # for parent moves
        while not self.comms["halt"].is_set():
            self.it_frame()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.comms["halt"].set()

            try:
                data = self.comms[
                    "q_from_cv"
                ].get_nowait()  ## blocking call doesn't waste cycles

                if data[0] == game_state.fen() or data[1] is None:
                    continue

                for s in ns:
                    if s.parent_move.uci() == data[1]:
                        return s
            except Exception:
                sleep(0.001)

        if self.comms["halt"].is_set():
            sys.exit(0)
