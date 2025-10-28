import os
from dotenv import load_dotenv
load_dotenv()  # noqa

import numpy as np
from matplotlib import pyplot as plt

import torch

import cv2

from tqdm import tqdm

import os
from Dataset.data import DataPath
from Dataset.DataSetLoaders.ChessDataset import ChessDataset, GameDataset
from HMM import ChessHMM
from Utils import ChessUtils
import ChessLens

import time

dataset = GameDataset(
    config={
        "img_size": (640, 640),
        "include_only": "valid"
    }
)

algorithm = "cnn"
img = ChessLens.ChessLensImage(piece_detector=algorithm)
hmm = ChessHMM.ChessHMM(30)


def prep_probs(piece_matrix: torch.Tensor):
    probs = piece_matrix

    if algorithm == "yolo":
        probs = torch.zeros(1, 13, 8, 8, dtype=torch.float32) + .1

        i = torch.arange(8).unsqueeze(1).expand(8, 8)
        j = torch.arange(8).unsqueeze(0).expand(8, 8)
        probs[0, piece_matrix.squeeze().to(torch.int32), i, j] = .9

    return -np.log(torch.rot90(probs, k=-1, dims=(2, 3)).squeeze().permute(1, 2, 0).numpy()[::-1]+(1e-7))


print(f"Devices | Board Detection: {ChessLens.BoardDetection.Bounded_Saddle_Yolo.bd.model.device} | Piece Recognition (CNN): {"None" if ChessLens.PieceDetection.PieceDetection_CNN.piece_detection_model is None else next(ChessLens.PieceDetection.PieceDetection_CNN.piece_detection_model.parameters()).device} | Piece Recognition (YOLO): {"None" if ChessLens.PieceDetection.PieceDetection_YOLO.model is None else ChessLens.PieceDetection.PieceDetection_YOLO.model.device}")

avg_times = {
    "load": 0,
    "board_detection": 0,
    "piece_recognition": 0,
    "HMM": 0
}
for t in tqdm(range(len(dataset))):
    t1 = time.perf_counter()
    img.load_image(dataset[t][0])
    t2 = time.perf_counter()
    img.detect_board()
    t3 = time.perf_counter()
    img.recognize_pieces()
    t4 = time.perf_counter()

    hmm.set_probs(t+1, prep_probs(img.piece_matrix))

    if (t % 5 == 0 and t >= 5):
        hmm.bind(t-4)
    t5 = time.perf_counter()

    avg_times["load"] += t2-t1
    avg_times["board_detection"] += t3-t2
    avg_times["piece_recognition"] += t4-t3
    avg_times["HMM"] += t5-t4

hmm.bind(len(dataset))

history = hmm.get_history()

avg_times["load"] /= len(dataset)
avg_times["board_detection"] /= len(dataset)
avg_times["piece_recognition"] /= len(dataset)
avg_times["HMM"] /= len(dataset)
avg_frame = avg_times["load"] + avg_times["board_detection"] + \
    avg_times["piece_recognition"] + avg_times["HMM"]
total_time = avg_frame * len(dataset)
print(f"Avg Image Loading: {avg_times["load"]*1e3:.4f}ms | Avg Board Detection: {avg_times["board_detection"]*1e3:.4f}ms | Avg Piece Recognition: {avg_times["piece_recognition"]*1e3:.4f}ms | Avg HMM: {avg_times["HMM"]*1e3:.4f}ms")
print(
    f"Avg Frame: {avg_frame*1e3:.4f}ms | Frame Count: {len(dataset)} | Total Time: {total_time*1e3:.4f}ms")

fens = []
for i in tqdm(range(history.shape[0])):
    fens.append(ChessUtils.ChessTensorUtils.tensorToFEN_MAX(
        history[[i], ::-1]))
    # ChessUtils.fen_to_png(fens[-1], "Game", f"game_{i}.png")

with open("Game/game_out.txt", "w") as f:
    pgn = ChessUtils.ChessTensorUtils.fens_to_pgn(fens)
    f.write(f"{'\n'.join(fens)}\n\n{pgn}")
