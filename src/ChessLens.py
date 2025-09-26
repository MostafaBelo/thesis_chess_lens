import numpy as np
from matplotlib import pyplot as plt

import torch
from torchvision import transforms
import cv2

from PIL import Image

from BoardDetection import BoardDetection
from PieceDetection import PieceDetection

from Utils import ChessUtils


class ChessLensImage:
    def __init__(self, img: str | torch.Tensor | np.ndarray | None = None):
        self.clear()
        self.load_image(img)

    def clear(self):
        self.img = None

        self.board_detection = None
        self.clock_time = None
        self.piece_matrix = None
        self.fen = None

    def is_img_loaded(self) -> bool:
        return not (self.img is None)

    def is_board_detected(self) -> bool:
        return not (self.board_detection is None)

    def is_clock_recognized(self) -> bool:
        return not (self.clock_time is None)

    def is_pieces_detected(self) -> bool:
        return not (self.piece_matrix is None or self.fen is None)

    def load_image(self, img: str | torch.Tensor | np.ndarray):
        if (type(img) == str):
            img = transforms.ToTensor()(Image.open(img).convert("RGB").resize((640, 640)))
        elif (type(img) == np.ndarray):
            img = torch.tensor(img)

        self.img = img

    def detect_board(self):
        if not self.is_img_loaded():
            raise Exception("No image loaded")

        # run board detection

        BoardDetection.board_extractor.set_img(self.img)
        self.board_detection, conf = BoardDetection.board_extractor.extract_board()
        self.board_detection = torch.tensor(self.board_detection)

    def recognize_clock(self):
        if not self.is_img_loaded():
            raise Exception("No image loaded")

        # run clock recognition

    def recognize_pieces(self):
        if not self.is_img_loaded():
            raise Exception("No image loaded")

        if not self.is_board_detected():
            raise Exception("No board detected")

        PieceDetection.piece_detector.set_img(self.img, self.board_detection)
        PieceDetection.piece_detector.preprocess()
        self.piece_matrix = PieceDetection.piece_detector.predict()

        # convert piece matrix to fen
        self.fen = ChessUtils.ChessTensorUtils().tensorToFEN_MAX(
            self.piece_matrix)

    def save_fen_image(self):
        if not self.is_pieces_detected():
            raise Exception("Pieces not detected")

        fen_img = ChessUtils.fen_to_png(self.fen, ".", "out_fen.png")

    def preview_board(self):
        plt.imshow(self.img)
        plt.imshow(self.board_detection[2], alpha=self.board_detection[2])

    def preview_pieces(self):
        self.pieces_yolo.show()

    def apply(self):
        self.detect_board()
        self.recognize_clock()
        self.recognize_pieces()
