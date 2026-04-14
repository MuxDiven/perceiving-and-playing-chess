import chess

"""
GameState is a small wrapper used in tree searchs, abstracting the chess.Board object to only the actions we require
"""


class GameState:
    def __init__(self, board=chess.Board(), parent_move=None):
        self.board = board
        self.parent_move = parent_move
        self.turn = self.board.turn

    def __str__(self):
        return (
            "\n"
            + str(self.board)
            + "\nlegal moves:"
            + (" WHITE" if self.turn else " BLACK")
            + "\n"
            + str([move.uci() for move in self.board.legal_moves])
            + "\n"
        )

    def check(self):
        return self.board.is_check()

    def checkmate(self):
        return self.board.is_checkmate()

    def draw(self):
        return self.board.is_stalemate() or self.board.is_insufficient_material()

    def terminal(self):
        return self.checkmate() or self.draw()

    def next_states(self, pseudo=False):
        legal_moves = (
            self.board.legal_moves if not pseudo else self.board.pseudo_legal_moves
        )
        children = [self.move(move) for move in legal_moves]
        return children

    def copy_board(self):
        return chess.Board(self.fen())

    def copy_state(self):
        return GameState(board=self.copy_board())

    ##TTable keys
    def fen(self):
        return self.board.fen()

    @property
    def n_legal_moves(self):
        return self.board.legal_moves.count()

    def legal_moves(self):
        return self.board.legal_moves

    def move(self, board_move):
        ## board_move is of type chess.Move
        self.board.push(board_move)
        child = self.copy_state()
        child.parent_move = board_move
        self.board.pop()
        return child

    @staticmethod
    def get_square(row, col):
        rank = 7 - row
        file = col
        return chess.square(file, rank)
