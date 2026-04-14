from trees import Minimax, Utils


class HA_Agent:
    def __init__(
        self,
        eval_func=Utils.Eval_fn.eval_pos,
        depth=5,
        max_worker_ts=4,
    ):
        self.n_moves = 0
        self.minimax = Minimax.Minimax(eval_func, worker_ts=max_worker_ts)
        self.depth = depth

    def move(self, game_state):
        return self.minimax.search(game_state, self.depth)
