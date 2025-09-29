# %%
import numpy as np
from matplotlib import pyplot as plt

import torch

import cv2

import os
from Dataset.data import DataPath

import importlib
import ChessLens
importlib.reload(ChessLens)

# %%
# img = ChessLens.ChessLensImage(os.path.join(DataPath, 'data_manual/1741715439429.jpg'))
img = ChessLens.ChessLensImage(os.path.join(
    DataPath, 'data_manual/1741715439470.jpg'))

# %%
img.detect_board()
# img.recognize_clock()
img.recognize_pieces()
img.save_fen_image()

# %%
plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)
plt.imshow(img.img.permute(1, 2, 0))
plt.subplot(1, 2, 2)
plt.imshow(plt.imread("out_fen.png"))
img.fen
