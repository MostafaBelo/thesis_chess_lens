import numpy as np
from matplotlib import pyplot as plt

import torch
import cv2

from BoardDetector_Yolo import main_yolo as board_yolo
from Piece_Detection import main_yolo as piece_yolo

from Utils import ChessUtils


class ChessLensImage:
    def __init__(self, img: str | torch.Tensor | np.ndarray | None = None):
        self.clear()
        self.load_image(img)

    def clear(self):
        self.img = None

        self.board_detection = None
        self.clock_time = None
        self.piece_bboxs = None
        self.piece_matrix = None
        self.fen = None

    def is_img_loaded(self) -> bool:
        return not (self.img is None)

    def is_board_detected(self) -> bool:
        return not (self.board_detection is None)

    def is_clock_recognized(self) -> bool:
        return not (self.clock_time is None)

    def is_pieces_detected(self) -> bool:
        return not (self.piece_bboxs is None or self.piece_matrix is None or self.fen is None)

    def load_image(self, img: str | torch.Tensor | np.ndarray):
        if (type(img) == str):
            img = torch.tensor(plt.imread(img))
        elif (type(img) == np.ndarray):
            img = torch.tensor(img)

        self.img = img

    def detect_board(self):
        if not self.is_img_loaded():
            raise Exception("No image loaded")

        # run board detection

        self.board_detection = board_yolo.detect_board_img(self.img)

    def recognize_clock(self):
        if not self.is_img_loaded():
            raise Exception("No image loaded")

        # run clock recognition

    def recognize_pieces(self):
        if not self.is_img_loaded():
            raise Exception("No image loaded")

        # run yolo piece detection and set piece_bboxs
        self.piece_bboxs = piece_yolo.detect_pieces_img(self.img)

        if not self.is_board_detected():
            raise Exception("Board not detected")

        # align piece bboxs to squares using detected board
        sqs, self.piece_matrix = piece_yolo.align_boxes_to_board(
            self.piece_bboxs, self.board_detection)
        # convert piece matrix to fen
        self.fen = ChessUtils.ChessTensorUtils().tensorToFEN_MAX(
            self.piece_matrix.unsqueeze(0))

    def save_fen_image(self):
        if not self.is_pieces_detected():
            raise Exception("Pieces not detected")

        fen_img = ChessUtils.fen_to_png(self.fen, ".", "out_fen.png")

    def apply(self):
        self.detect_board()
        self.recognize_clock()
        self.recognize_pieces()
