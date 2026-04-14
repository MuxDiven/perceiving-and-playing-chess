from trees.nnMCTS import nnMCTS


class NA_Agent:
    def __init__(self, model, n_rollouts=15):
        self.tree = nnMCTS(model)
        self.n_moves = 0
        self.n_rollouts = n_rollouts

    def move(self, game_state):
        return self.tree.search(game_state, self.n_rollouts)
