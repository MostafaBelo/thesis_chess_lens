import os
from dotenv import load_dotenv
load_dotenv()  # noqa

from Utils import ChessUtils
import ChessLens

import torch

import time


def take_image() -> torch.Tensor:
    pass


algorithm = "cnn_onnx_static"
game = ChessLens.ChessLensGame(algorithm, config={
    "game_out_path": "game_fens.csv",
    "is_detect_occlusion": False,
    "is_detect_wakeup": False
})
frame_times = []
frame_paths = []
# while True:
for t in range(5000):
    img = take_image()
    t1 = time.perf_counter()
    # game.set_img(img, verbose=True)
    is_wake_up = game.set_img(img)
    t2 = time.perf_counter()

    frame_times.append(t2-t1)

avg_frame = sum(frame_times)/len(frame_times)
total_time = sum(frame_times)
game.bind()
history = game.get_history(True)

print(f"Avg Image Loading: {game.avg_times["load"]*1e3:.4f}ms | Avg Board Detection: {game.avg_times["board_detection"]*1e3:.4f}ms | Avg Piece Recognition: {game.avg_times["piece_recognition"]*1e3:.4f}ms | Avg HMM: {game.avg_times["HMM"]*1e3:.4f}ms")
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
