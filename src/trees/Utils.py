import math
import random

import chess
from core.GameState import GameState


## family of game and tree scoring functions
class Game_fn:
    ##MCTS
    @staticmethod
    def rollout(game_state):
        alligance = game_state.turn
        while not game_state.terminal():
            game_state = random.choice(game_state.next_states())

        return 1 if game_state.turn != alligance else -1

    @staticmethod
    def bounded_rollout(n_rolls=50, parent_state=GameState()):
        v = 0

        for _ in range(n_rolls):
            node = parent_state
            v += Game_fn.rollout(node)

        return v / n_rolls

    ##Minimax


class Eval_fn:
    PIECE_VALUES = {
        "P": 100,
        "N": 320,
        "B": 330,
        "R": 500,
        "Q": 900,
        "K": 0,
        "p": -100,
        "n": -320,
        "b": -330,
        "r": -500,
        "q": -900,
        "k": 0,
    }

    @staticmethod
    def ucb(mcts_node):
        if mcts_node.n_plays == 0:
            return float("inf")

        C = math.e
        v = mcts_node.avg_win / mcts_node.n_plays
        return v + (C * math.sqrt(math.log(mcts_node.root_plays() / mcts_node.n_plays)))

    @staticmethod
    def eval_material(game_state):
        total = 0
        for _, p in game_state.board.piece_map().items():
            sign = 1 if p.color == game_state.turn else -1
            total += sign * abs(Eval_fn.PIECE_VALUES.get(p.symbol(), 0))
        return total

    @staticmethod
    def eval_mobility(game_state):
        player_moves = sum(1 for _ in game_state.board.legal_moves)
        player_is_check = game_state.check()
        game_state.board.push(chess.Move.null())
        opponent_moves = sum(1 for _ in game_state.board.legal_moves)
        opponent_is_check = game_state.check()
        game_state.board.pop()

        mobility_score = player_moves - opponent_moves

        CHECK_BALANCE = 3
        if player_is_check:
            mobility_score -= CHECK_BALANCE
        if opponent_is_check:
            mobility_score += CHECK_BALANCE

        return mobility_score

    @staticmethod
    def eval_center(game_state):
        center_sqrs = [chess.E4, chess.E5, chess.D4, chess.D5]
        W = 0.01

        score = 0
        for s in center_sqrs:
            friendly_attacks = game_state.board.attackers(game_state.turn, s)
            hostile_attacks = game_state.board.attackers(not game_state.turn, s)

            for sq in friendly_attacks:
                score += W * abs(
                    Eval_fn.PIECE_VALUES[game_state.board.piece_at(sq).symbol()]
                )

            for sq in hostile_attacks:
                score -= W * abs(
                    Eval_fn.PIECE_VALUES[game_state.board.piece_at(sq).symbol()]
                )

        return score

    @staticmethod
    def eval_pos(game_state):
        gs = game_state
        mat = Eval_fn.eval_material(gs)
        mob = Eval_fn.eval_mobility(gs)
        ctr = Eval_fn.eval_center(gs)

        return mat + (0.1 * mob) + (0.2 * ctr)
