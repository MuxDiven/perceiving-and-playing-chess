from collections import defaultdict

import chess
import torch

from trees.chess_net import Encoder, VPNet
from trees.node import MCTS_node


class VPNode:
    def __init__(self, policy_probs=None, policy_indeces=None, value=None):
        self.policy_probs = policy_probs
        self.policy_indeces = policy_indeces
        self.value = value


class nnMCTS:
    def __init__(self, file_path):
        self.t_table = defaultdict(VPNode)  ##FEN-str -> Value, P_probs respectively
        self.net = VPNet()

        self.net.eval()  # load weights
        state_dict = torch.load(
            file_path, map_location=torch.device("cpu")
        )  # or 'cuda'
        self.net.load_state_dict(state_dict)
        self.action_space = [
            Encoder.decode_az_4672(i) for i in range(4672)
        ]  ##hopefully no Nones come flying out this
        self.K = 30

    #typical UCT function for this style of neural assisted search
    def puct(self, node):
        assert node.children

        Q = torch.tensor([c.Q for c in node.children], dtype=torch.float32)
        N = torch.tensor([c.n_plays for c in node.children], dtype=torch.float32)
        P = torch.tensor([c.P for c in node.children], dtype=torch.float32)

        N_parent = N.sum()

        c_puct = 1.5  ##TODO:tune

        puct_scores = Q + c_puct * P * torch.sqrt(N_parent) / (1 + N)
        return puct_scores

    def best_child(self, node):
        best_idx = torch.argmax(self.puct(node)).item()
        return node.children[best_idx]

    #construct policy over current states top legal moves
    def create_child_nodes(self, node, topk_i, topk_p):
        valid_probs = []
        for i in range(self.K):
            action = self.action_space[topk_i[i].item()]
            if action in node.current.legal_moves():
                child_state = node.current.move(action)
                child_node = MCTS_node(node, child_state)
                valid_probs.append(topk_p[i])
                node.children.append(child_node)

        for child, prob in zip(node.children, valid_probs):
            child.P = prob / sum(valid_probs)

    ## SEARCH
    def expand(self, node):
        current_fen = node.current.fen()
        if not node.children:
            if current_fen in self.t_table:
                entry = self.t_table[current_fen]
                self.create_child_nodes(node, entry.policy_indeces, entry.policy_probs)
                node.back_prop(entry.value)
            else:
                board_tensor = Encoder.vectorise(current_fen)
                p, v = self.net.predict(board_tensor)

                ##get the value out of the tensor
                v = v.unsqueeze(0).item()

                ##sample probabilities and normalise
                topk_p, topk_i = torch.topk(p, self.K)
                topk_p = topk_p / topk_p.sum()

                topk_p = topk_p.squeeze(0)
                topk_i = topk_i.squeeze(0)

                self.create_child_nodes(node, topk_i, topk_p)
                self.t_table[current_fen] = VPNode(topk_p, topk_i, v)

        else:
            best_child = self.best_child(node)
            self.expand(best_child)

    def bounded_expansion(self, node, n_rollouts):
        for _ in range(n_rollouts):
            self.expand(node)

    def search(self, game_state, n_rollouts):
        node = MCTS_node(current=game_state)
        self.bounded_expansion(node, n_rollouts)
        return max(node.children, key=lambda c: c.n_plays).current
