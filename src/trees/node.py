from core.GameState import GameState


class MCTS_node:
    def __init__(self, parent=None, current=GameState()):
        self.parent = parent
        self.current = current
        self.children = []
        self.avg_win = 0
        self.n_plays = 0
        self.P = float("-inf")  ##policy estimation for this node as a succesor

    def spawn_children(self):
        ns = self.current.next_states()
        self.children = [MCTS_node(parent=self, current=n) for n in ns]

    def root_plays(self):
        if self.parent is not None:
            self.parent.root_plays()
        return self.n_plays

    def back_prop(self, wins):
        self.avg_win += wins
        self.n_plays += 1

        if self.parent is not None:
            self.parent.back_prop(wins)

    @property
    def Q(self):
        return self.avg_win / self.n_plays if self.n_plays > 0 else 0


class NA_node(MCTS_node):
    def __init__(self, parent=None, current=GameState()):
        super().__init__(parent, current)
