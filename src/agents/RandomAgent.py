import random


class RandomAgent:
    def __init__(self):
        pass

    def move(self, game_state):
        return random.choice(game_state.next_states())
