import os
from dotenv import load_dotenv
load_dotenv()  # noqa

from tqdm import tqdm

from Dataset.DataSetLoaders.ChessDataset import GameDataset
from Utils import ChessUtils
import ChessLens

import time

dataset = GameDataset(
    config={
        "img_size": (640, 640),
        "include_only": "valid;occlusion"
    }
)

algorithm = "cnn"
game = ChessLens.ChessLensGame(algorithm, config={
    # "is_detect_occlusion": False,
    # "is_detect_wakeup": False
})
frame_times = []
for t in tqdm(range(len(dataset))):
    img, _ = dataset[t]
    t1 = time.perf_counter()
    # game.set_img(img, verbose=True)
    game.set_img(img)
    t2 = time.perf_counter()
    # game.process_img(True)

    frame_times.append(t2-t1)

avg_frame = sum(frame_times)/len(frame_times)
total_time = sum(frame_times)
game.bind()
history = game.get_history(True)

print(f"Avg Image Loading: {game.avg_times["load"]*1e3:.4f}ms | Avg Board Detection: {game.avg_times["board_detection"]*1e3:.4f}ms | Avg Piece Recognition: {game.avg_times["piece_recognition"]*1e3:.4f}ms | Avg HMM: {game.avg_times["HMM"]*1e3:.4f}ms")
print(
    f"Avg Frame: {avg_frame*1e3:.4f}ms | Frame Count: {len(dataset)} | Total Time: {total_time*1e3:.4f}ms")

fens = []
for i in range(history.shape[0]):
    fens.append(ChessUtils.ChessTensorUtils.tensorToFEN_MAX(
        history[[i], ::-1]))
    # ChessUtils.fen_to_png(fens[-1], "Game", f"game_{i}.png")

with open("Game/game_out.txt", "w") as f:
    pgn = ChessUtils.ChessTensorUtils.fens_to_pgn(fens)
    f.write(f"{'\n'.join(fens)}\n\n{pgn}")


print(pgn)
