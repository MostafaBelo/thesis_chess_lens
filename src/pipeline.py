import sys
import os

import numpy as np
import ChessLens

img_path = sys.argv[1]
img = ChessLens.ChessLensImage(img_path)

img.detect_board()
# img.recognize_clock()
img.recognize_pieces()
img.save_fen_image()
