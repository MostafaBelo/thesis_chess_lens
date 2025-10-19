import torch
import numpy as np
import cv2

from PieceDetection.PieceDetection_CNN import PieceDetection_CNN
from PieceDetection.PieceDetection_YOLO import PieceDetection_YOLO

from typing import Literal


class PieceDetector:
    def __init__(self, method: Literal["cnn", "yolo",
                                       "cnn_onnx", "cnn_onnx_dynamic", "cnn_onnx_static",
                                       "yolo_onnx", "yolo_onnx_dynamic", "yolo_onnx_static"] = "yolo"):
        self.img = None
        self.corners = None

        match method:
            case "cnn":
                self.piece_detector = PieceDetection_CNN.PieceDetector("torch")

            case "yolo":
                self.piece_detector = PieceDetection_YOLO.PieceDetector()

            case "cnn_onnx":
                self.piece_detector = PieceDetection_CNN.PieceDetector("onnx")

            case "cnn_onnx_dynamic":
                self.piece_detector = PieceDetection_CNN.PieceDetector(
                    "onnx_dynamic")

            case "cnn_onnx_static":
                self.piece_detector = PieceDetection_CNN.PieceDetector(
                    "onnx_static")

            case "yolo_onnx":
                self.piece_detector = PieceDetection_YOLO.PieceDetector("onnx")

            case "yolo_onnx_dynamic":
                self.piece_detector = PieceDetection_YOLO.PieceDetector(
                    "onnx_dynamic")

            case "yolo_onnx_static":
                self.piece_detector = PieceDetection_YOLO.PieceDetector(
                    "onnx_static")

        # else:
        #     raise Exception("Piece Detection Method not found")

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
