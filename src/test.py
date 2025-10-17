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

dataset = GameDataset(
    config={
        "img_size": (640, 640)
    }
)

img = ChessLens.ChessLensImage()
hmm = ChessHMM.ChessHMM(20)


def prep_probs(piece_matrix):
    return -np.log(torch.rot90(piece_matrix, k=-1, dims=(2, 3)).squeeze().permute(1, 2, 0).numpy()[::-1]+(1e-7))


for t in range(len(dataset)):
    img.load_image(dataset[t])
    img.detect_board()
    img.recognize_pieces()

    hmm.set_probs(t+1, prep_probs(img.piece_matrix))

    if (t % 5 == 0 and t >= 5):
        hmm.bind(t-4)

hmm.bind(len(dataset))

history = hmm.get_history()

# fens = []
# for i in tqdm(range(history.shape[0])):
#     fens.append(ChessUtils.ChessTensorUtils.tensorToFEN_MAX(
#         history[[i], ::-1]))
#     ChessUtils.fen_to_png(fens[-1], "Game", f"game_{i}.png")

# with open("Game/game_out.txt", "w") as f:
#     f.write("\n".join(fens))
