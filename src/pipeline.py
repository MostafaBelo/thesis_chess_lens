import sys
import os

import numpy as np
import ChessLens

img_path = sys.argv[1]
if len(sys.argv) >= 3:
    piece_detector = sys.argv[2]
else:
    piece_detector = None
img = ChessLens.ChessLensImage(img_path, piece_detector)

img.detect_board()
# img.recognize_clock()
img.recognize_pieces()
img.save_fen_image()
