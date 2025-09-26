import torch
import numpy as np
import cv2

from PieceDetection.Piece_Detection_CNN import PieceDetection_CNN
from PieceDetection.Piece_Detection_YOLO import PieceDetection_YOLO

from typing import Literal


class PieceDetector:
    def __init__(self, method: Literal["cnn", "yolo"] = "cnn"):
        self.img = None
        self.corners = None

        if method == "cnn":
            self.piece_detector = PieceDetection_CNN.PieceDetector()
        elif method == "yolo":
            self.piece_detector = PieceDetection_YOLO.PieceDetector()
        else:
            raise Exception("Piece Detection Method not found")

    def set_img(self, img: torch.Tensor, corners: torch.Tensor):
        self.img = img  # 3, H, W
        self.corners = corners  # 4, 2

        self.piece_detector.set_img(self.img, self.corners)

    def preprocess(self):
        self.piece_detector.preprocess()

    def predict(self):
        return self.piece_detector.predict()


# piece_detector = PieceDetector()
piece_detector = PieceDetector("yolo")
