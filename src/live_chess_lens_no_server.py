from dotenv import load_dotenv
load_dotenv()  # noqa

from Utils import ChessUtils
import ChessLens

import time
import requests

# from UI.server import FenServer

# server = FenServer(port=8000)
# server.start()


def update_fen(fen: str):
    requests.get(f"10.42.0.0:8000/update_fen?fen={fen}")

    # server.update_fen(fen)


algorithm = "cnn_onnx_static"
dirname = "game_fens"
game1 = ChessLens.ChessLensGame1(algorithm, config={
    "camera_interval": 0.2
})
game2 = ChessLens.ChessLensGame2(config={
    # "context_delay": 0,
    # "context_continous": False,
    "context_delay": 5,
    "context_continous": True,

    "game_out_path": dirname,
    "fen_update": update_fen
})
is_running = True


def stop_game():
    global is_running
    is_running = False


# server.stop_game = stop_game

try:
    frame_times = []
    while is_running:
        t1 = time.perf_counter()
        probs = game1.operate()
        game2.update_bindings()
        if probs is False:
            break
        elif probs is None:
            continue
        game2.operate(probs)
        t2 = time.perf_counter()

        frame_times.append(t2-t1)

    avg_frame = sum(frame_times)/len(frame_times)
    total_time = sum(frame_times)
    game2.bind()
    fens = game2.get_history(True)

    print(f"Avg Image Loading: {game1.avg_times["load"]*1e3 / len(frame_times):.4f}ms | Avg Board Detection: {game1.avg_times["board_detection"]*1e3 / len(frame_times):.4f}ms | Avg Wakeup: {game1.avg_times["wakeup"]*1e3 / len(frame_times):.4f}ms | Avg Occlusion: {game1.avg_times["occlusion"]*1e3 / len(frame_times):.4f}ms | Avg Piece Recognition: {game1.avg_times["piece_recognition"]*1e3 / len(frame_times):.4f}ms | Avg HMM: {game2.avg_times["HMM"]*1e3 / len(frame_times):.4f}ms")
    print(
        f"Avg Frame: {avg_frame*1e3:.4f}ms | Frame Count: {len(frame_times)} | Total Time: {total_time*1e3:.4f}ms")

    fens = [fen.split(" - ")[0] for fen in fens]
    with open("Game/game_out.txt", "w") as f:
        pgn = ChessUtils.ChessTensorUtils.fens_to_pgn(fens)
        f.write(f"{'\n'.join(fens)}\n\n{pgn}")

    print(pgn)
except KeyboardInterrupt:
    print("Stopped Manually")
finally:
    game1.quit()
    game2.quit()
