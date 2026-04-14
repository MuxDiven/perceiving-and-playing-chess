import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import chess


class Minimax:
    def __init__(self, eval_fn, null_move=True, worker_ts=6):
        self.eval_fn = eval_fn
        self.t_table = defaultdict(lambda: None)
        self.R = 2 if null_move else float("inf")
        self.tt_lock = threading.Lock()
        self.t_pool = ThreadPoolExecutor(max_workers=worker_ts)

    def min_val(
        self,
        state,
        alpha=float("-inf"),
        beta=float("inf"),
        depth=float("inf"),
    ):
        ## evaluation
        if state.checkmate():
            return float("-inf")
        if state.draw():
            return 0
        if depth == 0:
            return self.eval_fn(state)

        ## null move heuristic
        if depth >= self.R + 1 and not state.check():
            v = self.max_val(
                state.move(chess.Move.null()), alpha, beta, depth - self.R - 1
            )

            if v <= alpha:
                return alpha

        ## normal search
        v = float("inf")
        for s in state.next_states():
            key = s.fen()
            v_0 = None

            ## TODO: maybe multiply values by depth as simple heuristic
            if key in self.t_table:
                v_0, stored_depth = self.t_table[key]

                if stored_depth < depth:
                    v_0 = self.max_val(s, alpha, beta, depth - 1)

            else:
                v_0 = self.max_val(s, alpha, beta, depth - 1)

            v = min(v, v_0)
            if v_0 <= alpha:
                break
            beta = min(v, beta)

        ## t_table accounts for depth
        k = state.fen()
        entry = self.t_table[k]
        if entry is None or entry[1] < depth:
            with self.tt_lock:
                self.t_table[k] = (v, depth)
        return v

    def max_val(
        self,
        state,
        alpha=float("-inf"),
        beta=float("inf"),
        depth=float("inf"),
    ):
        ##evaluate leaf nodes
        if state.checkmate():
            return float("inf")
        if state.draw():
            return 0
        if depth == 0:
            return self.eval_fn(state)

        # Null move Heuristic
        if depth >= self.R + 1 and not state.check():
            v = self.min_val(
                state.move(chess.Move.null()), alpha, beta, depth - self.R - 1
            )

            if v >= beta:
                return beta

        ##normal search
        v = float("-inf")
        ##find terminal states/evaluate to start back propegating orderings
        for s in state.next_states():
            key = s.fen()
            v_0 = None

            if key in self.t_table:
                v_0, stored_depth = self.t_table[key]

                if stored_depth < depth:
                    v_0 = self.min_val(s, alpha, beta, depth - 1)

            else:
                v_0 = self.min_val(s, alpha, beta, depth - 1)

            v = max(v, v_0)
            if v_0 >= beta:
                break
            alpha = max(alpha, v)

        ## t_table accounts for depth
        k = state.fen()
        entry = self.t_table[k]
        if entry is None or entry[1] < depth:
            with self.tt_lock:
                self.t_table[k] = (v, depth)
        return v

    def root_search(self, children, depth=float("inf")):
        futures = {
            self.t_pool.submit(self.max_val, child, depth=depth): child
            for child in children
        }

        best_move = None
        best_value = float("-inf")

        for f in as_completed(futures):
            move = futures[f]
            value = f.result()
            if value > best_value:
                best_move = move
                best_value = value

        return best_move, best_value

    def id_search(self, game_state, depth):
        best_move = game_state
        best_value = float("-inf")
        children = game_state.next_states()

        for d in range(1, depth + 1):
            move, value = self.root_search(children, d)
            if value > best_value:
                best_value = value
                best_move = move

            children.sort(key=lambda x: 1 if x == best_move else 0)

        return best_move

    def search(self, game_state, depth):
        return self.id_search(game_state, depth)
