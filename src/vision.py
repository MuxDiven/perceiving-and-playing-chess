from collections import defaultdict

import chess
import cv2
import numpy as np

"""
ChessAnalyst is used to extract information from a current frame of the board
to filter down the next move that was made relative to the internal state of the engine
"""


class ChessAnalyst:
    def __init__(self):
        self.density_map = {}  # sqr => density
        self.mean_colour_map = {}

    def position_delta(self, frame, square_map, state):
        # CV analysis config
        ZOOM_SCALE = 0.6
        NATURAL_LIGHT = 1000
        CONFIG = self.density_map is None and self.mean_colour_map is None

        state = state.copy()
        current_density_map = {}  # sqr -> real
        hsv_map = {}  # sqr -> real^3

        predicted_squares = set()
        edges = self.edge_extraction(frame)
        mask = np.zeros(edges.shape, dtype=np.uint8)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # narrows down later evaluation while taking information from the board
        from_sqr_candidates = defaultdict(list)

        # indicdators as to most likely cadidates during a normal move
        nmv_sqrs = set()

        # Filter useful information from the physical board
        for sqr in chess.SQUARES:
            mask.fill(0)
            occupent = state.piece_at(sqr)
            sqr_points = np.asarray(square_map[sqr], dtype=np.float32)
            center = np.mean(sqr_points, axis=0)

            zoomed_roi = center + (sqr_points - center) * ZOOM_SCALE
            zoomed_roi = zoomed_roi.astype(np.int32)

            cv2.fillPoly(mask, [zoomed_roi], 255)
            roi_edges = cv2.bitwise_and(edges, mask)
            mean_hsv = cv2.mean(hsv_frame, mask=mask)[:3]

            edge_count = np.sum(roi_edges > 0)
            pixel_count = np.sum(mask > 0)
            density = edge_count / pixel_count if pixel_count > 0 else 0

            if occupent is not None and self.is_low_contrast(sqr, occupent):
                density += 0.05

            if CONFIG:
                self.density_map[sqr] = density
                self.mean_colour_map[sqr] = mean_hsv
            current_density_map[sqr] = density
            hsv_map[sqr] = mean_hsv

            # detected a piece on the physical board
            if self.hard_sigmoid(density) >= NATURAL_LIGHT:
                if occupent is None:  # empty square is now "filled"
                    nmv_sqrs.add(sqr)
                predicted_squares.add(chess.square_name(sqr))
            elif (
                occupent is not None
            ):  # this is the event where we check the to sqr againts our candidates
                from_sqr_candidates[sqr] = [
                    move for move in state.legal_moves if move.from_square == sqr
                ]

            print(f"sigmoid({chess.square_name(sqr)}) = {self.hard_sigmoid(density)}\n")

        print(f"normal move square candidates {list(map(chess.square_name,nmv_sqrs))}")
        print(
            f"from square candidates {list(map(chess.square_name,from_sqr_candidates.keys()))}\n"
        )

        # score likely cadidates, max => model's best guess
        capture = not nmv_sqrs
        diffs = []
        for sqr in from_sqr_candidates.keys():
            for m in from_sqr_candidates[sqr]:
                if state.is_capture(m) != capture:
                    continue

                # evaluate normal moves based off new and old density snapshot
                to_sqr = m.to_square
                to_diff = abs(self.density_map[to_sqr] - current_density_map[to_sqr])
                from_diff = abs(self.density_map[sqr] - current_density_map[sqr])
                delta = to_diff * from_diff

                # evaluate capture on colour change as pieces will always contrast
                nabla = abs(self.mean_colour_map[to_sqr][2] - hsv_map[to_sqr][2])

                # store the errors
                diffs.append((delta, nabla, m))

                print(f"density diff @ {m.uci()[:4]} = {delta}")
                print(f"value diff @ {m.uci()[2:4]} = {nabla}\n")

        # switch on: capture -> colour diff, normal -> density delta product
        # bg = mux(diffs,max(mux(capture,colour,density)),None) ode to syrup
        best_guess = (max(diffs, key=lambda x: x[int(capture)]))[-1] if diffs else None

        if best_guess is not None:  # prevents pushing a null move
            state.push(best_guess)

        # update old snapshot
        self.density_map = current_density_map
        self.mean_colour_map = hsv_map

        print("complete")
        print(f"\nstate:\n{state}\n")

        predicted_squares = set(
            [
                chess.square_name(sqr)
                for sqr in chess.SQUARES
                if state.piece_at(sqr) is not None
            ]
        )
        return predicted_squares, state, best_guess

    def edge_extraction(self, frame, d_itr=1):
        SIGMA = 0.33
        img = frame.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_img = clahe.apply(gray)
        blurred = cv2.GaussianBlur(contrast_img, (5, 5), 0)
        v = np.median(blurred)
        lower = int(max(0, (1 - SIGMA) * v))
        upper = int(min(255, (1 + SIGMA) * v))
        edges_a = cv2.Canny(contrast_img, lower, upper)
        d_edges = cv2.dilate(edges_a, None, iterations=d_itr)
        return d_edges

    #
    def is_low_contrast(self, sqr, piece):
        if piece is None:
            return False

        return piece.color == (((sqr // 8) + (sqr % 8)) % 2 != 1)

    # creates greater differences when evaluating density thresholds, good for distinguishing pieces from noise
    def hard_sigmoid(self, x, weight=100, bias=0.145):
        return 1 / (np.exp(-weight * (x - bias)))


"""
The vision main loop and its helpers are used to call upon board analysis
"""


# Sort points based on their coordinates to get TL, TR, BR, BL order
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-Left
    rect[2] = pts[np.argmax(s)]  # Bottom-Right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-Right
    rect[3] = pts[np.argmax(diff)]  # Bottom-Left
    return rect


def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        img = params[0]
        points = params[1]
        if len(points) < 4:
            points.append((x, y))
            # Draw the point immediately so you can see it
            cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
            print(f"Point {len(points)} captured at: {x}, {y}")


def remap_square(r, c, cam_rot):
    match cam_rot:
        case 90:
            return c, 7 - r
        case 180:
            return 7 - r, 7 - c
        case 270:
            return 7 - c, r
        case _:
            return r, c


# create a map of internal model squares => 4 point pixle square regions
def map_squares(dst_grid, rotation):
    pixle_squares = defaultdict(list)
    for r in range(8):
        for c in range(8):
            r_prime, c_prime = remap_square(r, c, rotation)

            file = chr(ord("a") + c_prime)
            rank = 8 - r_prime
            sqr_name = f"{file}{rank}"
            sqr = chess.parse_square(sqr_name)

            tl = dst_grid[r_prime, c_prime].tolist()
            tr = dst_grid[r_prime, c_prime + 1].tolist()
            br = dst_grid[r_prime + 1, c_prime + 1].tolist()
            bl = dst_grid[r_prime + 1, c_prime].tolist()
            pixle_squares[sqr] = [tl, tr, br, bl]

    return pixle_squares


def draw_grid(dst_grid, frame, rotation, predicted_set=set()):
    # fill display grid
    for i in range(9):
        cv2.polylines(frame, [dst_grid[i, :, :].astype(int)], False, (255, 0, 0), 1)
        cv2.polylines(frame, [dst_grid[:, i, :].astype(int)], False, (255, 0, 0), 1)

    # cv2.imshow("polyLines", frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # should look good on the paper

    files = "abcdefgh"
    ranks = "87654321"
    font = cv2.FONT_HERSHEY_COMPLEX
    for r in range(8):
        for c in range(8):
            r_std, c_std = remap_square(r, c, rotation)
            # r_std, c_std = r,c
            file_letter = files[c_std]
            rank_char = ranks[r_std]
            label = f"{file_letter}{rank_char}"

            center = (
                dst_grid[r_std, c_std]
                + (dst_grid[r_std + 1, c_std + 1] - dst_grid[r_std, c_std]) / 2
            )
            cx, cy = int(center[0]), int(center[1])
            cv2.putText(
                frame,
                label,
                (cx - 12, cy + 5),
                font,
                0.5,
                (0, 255, 0) if label in predicted_set else (0, 255, 255),
                1,
                cv2.LINE_AA,
            )


def show_controls():
    print("chess vision controls\n")
    print("q -> quit")
    print("c -> infer next position")
    print("p -> push to engine")
    print("r -> reset board config")
    print("h -> prints this again\n")


def main_loop(comms, device=0, camera_rotation=90):
    CAMERA_ROTATION = camera_rotation

    board_analyser = ChessAnalyst()
    predicted_set = set()  # board configuration guessed by the inference model
    points = []  # The four corners of that make up the whole sqr of the board
    pixle_squares = {}  # set of four corners for each model square
    calibration_data = [None, points]
    prev_state = chess.Board()
    state_mutation = False
    parent_move = None

    cap = cv2.VideoCapture(device)
    cv2.namedWindow("Board Analysis")
    cv2.setMouseCallback("Board Analysis", click_event, calibration_data)

    H = None
    dst_grid = None

    print(f"the initial internal state \n{prev_state}\n")

    show_controls()

    while not comms["halt"].is_set():
        ret, frame = cap.read()
        frame_c = frame.copy()

        if not ret:
            comms["halt"].set()
            break

        # try to recieve the updated state from the engine and update internally
        try:
            fen, parent_move, camera_input = comms["q_to_cv"].get_nowait()

            prev_state = chess.Board(fen)

            predicted_set = set(
                [
                    chess.square_name(sqr)
                    for sqr in chess.SQUARES
                    if prev_state.piece_at(sqr) is not None
                ]
            )

            if (
                not camera_input
                and board_analyser.density_map
                and board_analyser.mean_colour_map
                and parent_move
            ):
                from_s, to_s = chess.parse_square(parent_move[:2]), chess.parse_square(
                    parent_move[2:4]
                )

                temp = board_analyser.density_map[from_s]
                board_analyser.density_map[from_s] = board_analyser.density_map[to_s]
                board_analyser.density_map[to_s] = temp

                temp = board_analyser.mean_colour_map[from_s]
                board_analyser.mean_colour_map[from_s] = board_analyser.mean_colour_map[
                    to_s
                ]
                board_analyser.mean_colour_map[to_s] = temp

        except Exception:
            # the queue is empty, there is no update
            pass

        calibration_data[0] = frame

        # user enters 4 corners of their chess board to define cv grid
        if len(points) < 4:
            for p in points:
                cv2.circle(frame, p, 10, (0, 255, 0), -1)

        # compute grid(once), draw grid, listen for input
        elif len(points) == 4:
            if H is None:
                points = np.asarray(points)
                points = order_points(points)

                pts = np.array(points, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

                assert CAMERA_ROTATION % 90 == 0
                src = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=np.float32)
                src = np.roll(src, shift=(CAMERA_ROTATION / 90) % 4, axis=0)

                # Homography matrix
                H = cv2.getPerspectiveTransform(src, points)

                # init grid
                src_grid = np.array(
                    [[[x, y] for x in range(9)] for y in range(9)], dtype=np.float32
                )
                dst_grid = cv2.perspectiveTransform(
                    src_grid.reshape(-1, 1, 2), H
                ).reshape(9, 9, 2)
                pixle_squares = map_squares(dst_grid, CAMERA_ROTATION)

            # draw grid to board
            draw_grid(dst_grid, frame, CAMERA_ROTATION, predicted_set)

        cv2.imshow("Board Analysis", frame)

        key = cv2.waitKey(1) & 0xFF

        # quit
        if key == ord("q"):
            comms["halt"].set()
            break

        # push to chess engine can only properly progress when state is updated
        elif key == ord("p") and H is not None:
            comms["q_from_cv"].put((prev_state.fen(), parent_move, True))

        # reset grid config
        elif key == ord("r") and H is not None:
            predicted_set = set()
            points = []
            calibration_data[1] = points
            H = None
            dst_grid = None

        # capture current snapshot and evaluate
        elif key == ord("c") and H is not None:
            predicted_set, prev_state, delta_move = board_analyser.position_delta(
                frame_c, pixle_squares, prev_state
            )
            if delta_move is not None:
                parent_move = delta_move.uci()
            print(predicted_set)

        # help
        elif key == ord("h"):
            show_controls()

    # clean up
    cap.release()
    cv2.destroyAllWindows()
