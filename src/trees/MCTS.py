import math
import random
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import chess

from trees.node import MCTS_node
from trees.Utils import Eval_fn


class MCTS:
    def __init__(
        self, eval_func, game_func, search_lim, worker_ts=4, mt=sys._is_gil_enabled()
    ):
        ## game_func should take a GameState in to evaluate
        self.eval_func = eval_func
        self.game_func = game_func
        self.search_lim = search_lim
        self.t_table = defaultdict()
        self.mt = mt
        self.t_pool = ThreadPoolExecutor(worker_ts)
        self.tt_lock = threading.Lock()

    ## TODO: change eval_pos to some neural assited function
    ## EVAL FUNCS
    def puct(self, child):
        C = 0.75
        v = 0 if child.n_plays == 0 else child.avg_win / child.n_plays
        key = child.current.fen()

        p = None
        with self.tt_lock:
            if key in self.t_table:
                p = self.t_table[key]
            else:
                p = Eval_fn.eval_pos(child.current)
                self.t_table[key] = p

        return v + (C * p * (math.sqrt(child.root_plays() / (1 + child.n_plays))))

    def ucbh(self, child):
        if child.n_plays == 0:
            return float("inf")

        key = child.current.fen()
        p = None
        with self.tt_lock:
            if key in self.t_table:
                p = self.t_table[key]
            else:
                p = math.tanh(Eval_fn.eval_pos(child.current))
                self.t_table[key] = p

        C = 0.1 
        Q = p * (child.avg_win / child.n_plays)
        return Q + (C * math.sqrt(math.log(child.root_plays() / child.n_plays)))

    def best_child(self, node):
        return max(node.children, key=self.ucbh)


    ## SEARCH
    def expand(self, node):
        if not node.children:
            if node.n_plays == 0:
                v = self.game_func(node.current)
                node.back_prop(v)
            else:
                node.spawn_children()

                if node.children:
                    ## TODO: proper child evaluation
                    child = random.choice(node.children)
                    v = self.game_func(child.current)
                    child.back_prop(v)
        else:
            best_child = self.best_child(node)
            self.expand(best_child)

    def bounded_expansion(self, node):
        for _ in range(self.search_lim):
            self.expand(node)

    def mt_search(self, game_state):
        root = MCTS_node(current=game_state)
        root.spawn_children()
        ## expand children to give initial values to root for eval
        for c in root.children:
            self.expand(c)

        futures = [self.t_pool.submit(self.bounded_expansion, c) for c in root.children]
        res = [f.result() for f in as_completed(futures)]

        return max(root.children, key=lambda x: x.avg_win / x.n_plays)

    def normal_search(self, game_state):
        root = MCTS_node(current=game_state)
        root.spawn_children()
        self.bounded_expansion(root)
        played_children = [c for c in root.children if c.n_plays != 0]
        best_child = max(played_children, key=lambda x: x.avg_win / x.n_plays)
        return best_child.current

    def search(self, game_state):
        return self.mt_search(game_state) if self.mt else self.normal_search(game_state)
