import multiprocessing as mp

import chess

import agents.CameraAgent as C

# import agents.HA_Agent as M
import agents.NA_Agent as N
import vision
from core.Game import Game
from core.GameState import GameState


def basic_playground(agent_conf, ws=(640, 640)):
    GAME_STATE = GameState(chess.Board())
    Game(init_state=GAME_STATE, agents=agent_conf, window_size=ws).start()


if __name__ == "__main__":

    agent_conf = []
    hardware_acc = False
    DEVICE = -1  # what camera do we wish to use
    CAMERA_ROTATION = None
    comms = None

    print("enter your agent configuration, first white then black")
    print("camera:        c")
    print("neural-net:    n")
    print("player:        p[default]")

    while len(agent_conf) != 2:

        action = input("(agent type) > ")

        match action.lower():
            case "c":
                if not hardware_acc:
                    DEVICE = int(input("(what capture device do you wish to use) > "))
                    # TODO better instructions
                    print(
                        "input the camera rotation offset: white at the bottom is seen as 0 deg\nkings closest to the bottom being 90 deg"
                    )

                    CAMERA_ROTATION = int(input("(camera rotation) > "))

                    # shared memeory pipeline between the engine and the camera system
                    comms = {
                        "q_from_cv": mp.Queue(),
                        "q_to_cv": mp.Queue(),
                        "halt": mp.Event(),
                    }
                    hardware_acc = True

                agent_conf.append(
                    C.CameraAgent(comms)
                )  # this will be the cv system init
            case "n":
                # can build a config in this case
                agent_conf.append(
                    N.NA_Agent(n_rollouts=250, model="../models/model.pth")
                )
            case _:
                agent_conf.append(
                    None
                )  # None represents the mouse as input for the engine

    if hardware_acc:
        CV_PROCESS = mp.Process(
            target=vision.main_loop, args=(comms, DEVICE, CAMERA_ROTATION)
        )
        CHESS_ENGINE = Game(GameState(), tuple(agent_conf), (640, 640), comms=comms)

        # FULL SYSTEM UP AND RUNNING
        CV_PROCESS.start()
        CHESS_ENGINE.start()
    else:
        # normal engine only process no MP required
        basic_playground(tuple(agent_conf))
