import time

import chess
import pygame

from core.GameState import GameState
from core.Model import Model
from core.View import View

FPS = 60
DEFAULT_AGENT_CONFIG = (None, None)  ##pvp, left is white

"""
The Controller acts as the main bridge between the Engine UI and Logic. Its goal is to ensure that all communication between these process are smooth
for engine use, agents and camera systems
"""


class Controller:
    def __init__(self, model, view, comms=None):
        self.model = model
        self.view = view
        self.comms = comms
        self.clock = pygame.time.Clock()

    @property
    def comms_check(self):
        return True if self.comms is None else not self.comms["halt"].is_set()

    def terminal_game_state(self):
        return self.model.current_state.terminal()

    def it_frame(self):
        self.view.draw_board(self.model.get_board())
        pygame.display.flip()
        self.clock.tick(FPS)

    def move_generator(self, origin_sqr, current_square):
        piece = self.model.get_board().piece_at(origin_sqr)
        if piece and piece.piece_type == chess.PAWN:
            if (
                piece.color == chess.WHITE and chess.square_rank(current_square) == 7
            ) or (
                piece.color == chess.BLACK and chess.square_rank(current_square) == 0
            ):
                return chess.Move(origin_sqr, current_square, promotion=chess.QUEEN)
        return chess.Move(origin_sqr, current_square)

    def play(self):
        running = True
        origin_sqr = None
        move = None
        n_moves = 0

        print(self.model.get_turn())

        agent = self.model.current_player()
        start = time.time()
        # main gameplay loop
        while running and self.comms_check and not self.terminal_game_state():
            self.it_frame()
            agent_input = agent is not None

            # process agent input
            if agent_input:
                self.model.agent_move(self.it_frame)
                n_moves += 1
                print(f"{self.model.turn_str()}'s Move\ntotal moves played: {n_moves}")

                if self.comms:
                    fen = self.model.current_state.fen()
                    p_move = self.model.current_state.parent_move
                    self.comms["q_to_cv"].put((fen, p_move, self.model.is_camera))

                ## switch player turn
                agent = self.model.current_player()

            ##process player input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                    if self.comms:
                        self.comms["halt"].set()

                elif event.type == pygame.MOUSEBUTTONDOWN and not agent_input:
                    x, y = pygame.mouse.get_pos()
                    col, row = x // self.view.SQUARE_SIZE, y // self.view.SQUARE_SIZE
                    current_square = GameState.get_square(row, col)

                    ##move selection
                    if origin_sqr is None:
                        origin_sqr = current_square
                        print(f"x,y:{x,y}\nsquare:{chess.square_name(origin_sqr)}\n")
                    else:
                        move = self.move_generator(origin_sqr, current_square)

                        if move not in self.model.get_board().legal_moves:
                            print(f"Illegal move: {move.uci()}")
                            move = None
                        origin_sqr = None

            ## we have full human input
            if agent is None and move is not None:
                self.model.player_move(move)
                move = None
                n_moves += 1
                agent = self.model.current_player()
                print(f"{self.model.turn_str()}'s Move\ntotal moves played: {n_moves}")

        print(f"\nGAME TIME: {time.time() - start}\n")

        ## allow use to view end of game state
        if self.terminal_game_state():
            running = True
            if self.model.current_state.checkmate():
                print(f"CHECKMATE {self.model.turn_str()}")
            else:
                print("DRAW")

            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        if self.comms:
                            self.comms["halt"].set()

                self.it_frame()

        pygame.quit()
