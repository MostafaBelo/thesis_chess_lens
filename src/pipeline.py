import sys
import os

import numpy as np
import ChessLens

import time

img_path = sys.argv[1]
if len(sys.argv) >= 3:
    piece_detector = sys.argv[2]
else:
    piece_detector = None

img = ChessLens.ChessLensImage(img_path, piece_detector)
img.detect_board()
img.recognize_pieces()

print(f"Devices | Board Detection: {ChessLens.BoardDetection.bd.model.device} | Piece Recognition (CNN): {next(ChessLens.PieceDetection.PieceDetection_CNN.piece_detection_model.parameters()).device} | Piece Recognition (YOLO): {ChessLens.PieceDetection.PieceDetection_YOLO.model.device}")

start_time = time.perf_counter()
img = ChessLens.ChessLensImage(img_path, piece_detector)
end_time = time.perf_counter()
print(f"Image Loading: {(end_time-start_time)*1e3:.6f} ms")

start_time = time.perf_counter()
img.detect_board()
end_time = time.perf_counter()
print(f"Board Detection: {(end_time-start_time)*1e3:.6f} ms")

# img.recognize_clock()

start_time = time.perf_counter()
img.recognize_pieces()
end_time = time.perf_counter()
print(f"Piece Recognition: {(end_time-start_time)*1e3:.6f} ms")

img.save_fen_image()
