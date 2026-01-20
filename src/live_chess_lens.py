import os
from dotenv import load_dotenv
load_dotenv()  # noqa

from Utils import ChessUtils
import ChessLens
from ImageProvider.ImageProvider import ImageProvider

import numpy as np
import cv2
import torch
from torchvision import transforms

from PIL import Image

import time
import sys

from UI.server import FenServer

server = FenServer(port=8000)
server.start()

camera = ImageProvider(
    interval=0.2,
    # camera="files",
    # interval=0,
    # data_dir="/mnt/D/University/Thesis_Dataset/Temp/photos2/photos"
)

algorithm = "cnn_onnx_static"
dirname = "game_fens"
# with open(filename, "w") as f:
#    f.write("rnbqkbnr")
game = ChessLens.ChessLensGame(algorithm, config={
    "game_out_path": dirname,
    # "is_detect_occlusion": False,
    # "is_detect_wakeup": False,
    # "context_delay": 0,
    # "context_continous": False,
    "context_delay": 60,
    "context_continous": True,

    "fen_update": server.update_fen
})
is_running = True


def stop_game():
    global is_running
    is_running = False


server.stop_game = stop_game

try:
    frame_times = []
    while is_running:
        # for t in range(500):
        img = camera.take_image()
        if img is None:
            break
        t1 = time.perf_counter()
        # game.set_img(img, verbose=True)
        game.set_img(img)
        t2 = time.perf_counter()

        frame_times.append(t2-t1)

    avg_frame = sum(frame_times)/len(frame_times)
    total_time = sum(frame_times)
    game.bind()
    history = game.get_history(True)

    print(f"Avg Image Loading: {game.avg_times["load"]*1e3 / len(frame_times):.4f}ms | Avg Board Detection: {game.avg_times["board_detection"]*1e3 / len(frame_times):.4f}ms | Avg Wakeup: {game.avg_times["wakeup"]*1e3 / len(frame_times):.4f}ms | Avg Occlusion: {game.avg_times["occlusion"]*1e3 / len(frame_times):.4f}ms | Avg Piece Recognition: {game.avg_times["piece_recognition"]*1e3 / len(frame_times):.4f}ms | Avg HMM: {game.avg_times["HMM"]*1e3 / len(frame_times):.4f}ms")
    print(
        f"Avg Frame: {avg_frame*1e3:.4f}ms | Frame Count: {len(frame_times)} | Total Time: {total_time*1e3:.4f}ms")

    fens = []
    for i in range(history.shape[0]):
        fens.append(ChessUtils.ChessTensorUtils.tensorToFEN_MAX(
            history[[i], ::-1]))
        # ChessUtils.fen_to_png(fens[-1], "Game", f"game_{i}.png")

    with open("Game/game_out.txt", "w") as f:
        pgn = ChessUtils.ChessTensorUtils.fens_to_pgn(fens)
        f.write(f"{'\n'.join(fens)}\n\n{pgn}")

    print(pgn)
except KeyboardInterrupt:
    print("Stopped Manually")
finally:
    camera.quit()
