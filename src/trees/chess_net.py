import time

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader, Dataset

C = 13
INPUT_SHAPE = (C, 8, 8)

PIECE_TO_PLANE = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("CUDA available:", torch.cuda.is_available())  # should be True
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0))


SLIDING_DIRS = [
    (-1, 0),  # N
    (1, 0),  # S
    (0, 1),  # E
    (0, -1),  # W
    (-1, 1),  # NE
    (-1, -1),  # NW
    (1, 1),  # SE
    (1, -1),  # SW
]

KNIGHT_DIRS = [
    (2, 1),
    (2, -1),
    (-2, 1),
    (-2, -1),
    (1, 2),
    (1, -2),
    (-1, 2),
    (-1, -2),
]

UNDERPROMOS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

"""
Encoder is used to encode/decode from the az4672 policy as well as normalise state information for the neural net
"""


class Encoder:
    @staticmethod
    def vectorise(fen):
        board, to_move = fen.split()[:2]

        tensor = np.zeros(INPUT_SHAPE, dtype=np.float32)

        rows = board.split("/")

        for i, row in enumerate(rows):
            j = 0
            for char in row:
                if char.isdigit():
                    j += int(char)
                else:
                    plane = PIECE_TO_PLANE[char]
                    tensor[plane, i, j] = 1.0
                    j += 1

        if to_move == "w":
            tensor[12, :, :] = 1.0

        return torch.from_numpy(tensor)

    @staticmethod
    def encode_az_4672(move: chess.Move) -> int:
        from_sq = move.from_square
        to_sq = move.to_square

        assert 0 <= move.from_square < 64
        assert 0 <= move.to_square < 64

        fr, fc = divmod(from_sq, 8)
        tr, tc = divmod(to_sq, 8)

        dr = tr - fr
        dc = tc - fc

        if move.promotion in UNDERPROMOS:
            promo_idx = UNDERPROMOS.index(move.promotion)
            # forward, forward-left, forward-right
            if dc == 0:
                dir_idx = 0
            elif dc == -1:
                dir_idx = 1
            elif dc == 1:
                dir_idx = 2
            else:
                raise ValueError("Invalid underpromotion direction")

            move_type = 64 + dir_idx * 3 + promo_idx
            assert 0 <= move_type < 73

            ans = from_sq * 73 + move_type
            return ans

        if (dr, dc) in KNIGHT_DIRS:
            move_type = 56 + KNIGHT_DIRS.index((dr, dc))
            assert 0 <= move_type < 73
            return from_sq * 73 + move_type

        for dir_idx, (sdr, sdc) in enumerate(SLIDING_DIRS):
            for dist in range(1, 8):
                if dr == sdr * dist and dc == sdc * dist:
                    move_type = dir_idx * 7 + (dist - 1)
                    assert 0 <= move_type < 73
                    return from_sq * 73 + move_type

        raise ValueError(f"Unencodable move: {move}")

    @staticmethod
    def decode_az_4672(index: int) -> chess.Move | None:
        """Decode a label to a move. Returns None if move goes off the board."""
        from_sq = index // 73
        move_type = index % 73
        fr, fc = divmod(from_sq, 8)

        to_sq = None
        promotion = None

        if 64 <= move_type < 73:  # underpromotion
            up_idx = move_type - 64
            dir_idx, promo_idx = divmod(up_idx, 3)
            tc = fc
            if dir_idx == 1:
                tc = fc - 1
            elif dir_idx == 2:
                tc = fc + 1
            tr = fr + 1
            promotion = UNDERPROMOS[promo_idx]
            to_sq = tr * 8 + tc

        elif 56 <= move_type < 64:  # knight
            k_idx = move_type - 56
            dr, dc = KNIGHT_DIRS[k_idx]
            tr, tc = fr + dr, fc + dc
            to_sq = tr * 8 + tc

        else:  # sliding
            for dir_idx, (sdr, sdc) in enumerate(SLIDING_DIRS):
                for dist in range(1, 8):
                    if dir_idx * 7 + (dist - 1) == move_type:
                        tr, tc = fr + sdr * dist, fc + sdc * dist
                        to_sq = tr * 8 + tc
                        break

        # --- centralized clipping ---
        if to_sq is None or not (0 <= from_sq < 64) or not (0 <= to_sq < 64):
            return None

        return chess.Move(from_sq, to_sq, promotion=promotion)


class ChessDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.X = torch.from_numpy(data["states"]).float()
        self.yp = torch.from_numpy(data["policy"]).long()
        self.yv = torch.from_numpy(data["values"]).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, yp, yv = self.X[idx], self.yp[idx], self.yv[idx]
        return x, yp, yv


class VPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.N_POSSIBLE_MOVES = 8 * 8 * 73

        # trunk is responsible for positioning/spatial analysis on the board state
        self.trunk = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # the policy is a probability dist over the parent state's legal moves
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, self.N_POSSIBLE_MOVES),
        )

        # The value is a score of a current choice that seeks to answer the question "are we currently winning/losing?"
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        self.policy_loss = nn.CrossEntropyLoss()
        self.value_loss = nn.MSELoss()

    def forward(self, x):
        features = self.trunk(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value

    ##NOTE: this is the main function we will be using
    def predict(self, x):
        self.eval()
        x = x.unsqueeze(0)
        policy, value = self(x)
        p_sigma = F.softmax(policy, dim=1)
        v_sigma = F.tanh(value)
        return p_sigma, v_sigma

    def fit(self, train_dataset, epochs=5):
        LAMBDA = 1.0  # loss ofset
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        start = time.time()
        for i in range(epochs):
            self.train()
            print(f"Starting Epoch {i}")
            for X, yp, yv in train_loader:
                X, yp, yv = (
                    X.to(device),
                    yp.to(device),
                    yv.to(device),
                )
                self.optimizer.zero_grad()

                policy_logits, value = self(X)

                p_loss = self.policy_loss(policy_logits, yp)
                v_loss = self.value_loss(value.squeeze(), yv)
                loss = p_loss + LAMBDA * v_loss
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {i}")

        print(f"training took: {time.time() - start}")

    # TEST FUNCTION
    def evaluate(self, test_dataset):
        self.eval()
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        p_correct = 0
        p_topk_correct = 0
        v_sign_correct = 0
        v_mse_sum = 0

        total = 0

        with torch.no_grad():
            for X, yp, yv in test_loader:
                X, yp, yv = (
                    X.to(device),
                    yp.to(device),
                    yv.to(device),
                )

                p_logits, value_pred = self(X)
                value_pred = value_pred.squeeze(1)

                # policy top-1
                preds = p_logits.argmax(dim=1)
                p_correct += (preds == yp).sum().item()

                # policy topk
                topk_preds = p_logits.topk(5, dim=1).indices
                p_topk_correct += (
                    (topk_preds == yp.unsqueeze(1)).any(dim=1).sum().item()
                )

                v_mse_sum += F.mse_loss(value_pred, yv, reduction="sum").item()

                v_sign_correct += (
                    (torch.sign(value_pred) == torch.sign(yv)).sum().item()
                )

                total += yp.size(0)

        return {
            "policy_top1": 100 * p_correct / total,
            "policy_topk": 100 * p_topk_correct / total,
            "value_mse": 100 * v_mse_sum / total,
            "value_sign_acc": 100 * v_sign_correct / total,
        }
