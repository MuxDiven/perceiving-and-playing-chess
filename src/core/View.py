import chess
import pygame

from core.GameState import GameState

"""
View presents the UI of the chess engine and primarly communicates to and through the Controller
"""


class View:
    def __init__(self, size_tup=(640, 640), current_state=chess.Board()):
        ## dimensions
        HEIGHT, WIDTH = size_tup

        ## TODO: implement padding
        if HEIGHT != WIDTH:
            raise NotImplementedError(
                "non square boards will be implemented with menu options"
            )

        self.ROWS, self.COLS = 8, 8
        self.SQUARE_SIZE = HEIGHT // self.COLS
        FONT_SIZE = int(self.SQUARE_SIZE * 0.8)

        ## colours
        self.LIGHT = (240, 217, 181)
        self.DARK = (181, 136, 99)
        self.TEXT = (0, 0, 0)

        ## screen
        self.screen = pygame.display.set_mode(size_tup)
        self.font = pygame.font.Font(
            "../assets/NotoSansSymbols2-Regular.ttf", FONT_SIZE
        )  ## run in src or use relative path
        self.draw_board(current_state)

    def draw_board(self, board):
        for row in range(self.ROWS):
            for col in range(self.COLS):
                colour = self.LIGHT if (row + col) % 2 == 0 else self.DARK

                ## starting point
                x = col * self.SQUARE_SIZE
                y = row * self.SQUARE_SIZE

                ## x,y -> from topleft , SQUARE_SIZE -> square dims
                pygame.draw.rect(
                    self.screen, colour, (x, y, self.SQUARE_SIZE, self.SQUARE_SIZE)
                )

                piece = board.piece_at(GameState.get_square(row, col))
                if piece:
                    face = chess.UNICODE_PIECE_SYMBOLS[piece.symbol()]
                    text_surface = self.font.render(face, True, self.TEXT)
                    text_rect = text_surface.get_rect(
                        center=(x + self.SQUARE_SIZE / 2, y + self.SQUARE_SIZE / 2)
                    )
                    self.screen.blit(text_surface, text_rect)
