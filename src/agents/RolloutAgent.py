from trees import MCTS, Utils


class Rollout_Agent:
    def __init__(
        self,
        eval_func=Utils.Eval_fn.ucb,
        game_func=Utils.Game_fn.rollout,
        search_lim=5,
        max_worker_ts=4,
    ):
        self.n_moves = 0  ##we can tailor openings/heuristics with this
        self.mcts = MCTS.MCTS(eval_func, game_func, search_lim, max_worker_ts)

    def move(self, game_state):
        self.n_moves += 1
        return self.mcts.search(game_state)

    def set_search_lim(self, search_lim):
        self.mcts.search_lim = search_lim
