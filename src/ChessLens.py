from dotenv import load_dotenv
load_dotenv()  # noqa

import numpy as np
from matplotlib import pyplot as plt

import torch
from torchvision import transforms
import cv2

from PIL import Image

from BoardDetection import BoardDetection
from OcclusionDetection import OcclusionDetectionCNN
from WakeupDetection import WakeupModule
from PieceDetection import PieceDetection
from ContextAwareModels.HMM import ChessHMM

from Utils import ChessUtils

from typing import Literal

import time


class ChessLensImage:
    def __init__(self, img: str | torch.Tensor | np.ndarray | None = None, piece_detector: Literal["cnn", "yolo",
                                                                                                   "cnn_onnx", "cnn_onnx_dynamic", "cnn_onnx_static", "cnn_prunned",
                                                                                                   "yolo_onnx", "yolo_onnx_dynamic", "yolo_onnx_static"] | None = None):
        self.clear()
        self.load_image(img)

        if piece_detector is None:
            self.piece_detector = PieceDetection.PieceDetector()
        else:
            self.piece_detector = PieceDetection.PieceDetector(piece_detector)

        self.occlusion_model = OcclusionDetectionCNN.OcclusionDetectorCNN()

    def clear(self):
        self.img = None
        self.warped_img = None

        self.board_detection = None
        self.clock_time = None
        self.piece_matrix = None
        self.orientation = None
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
        if self.img is not None:
            BoardDetection.board_extractor.set_img(self.img)

    def detect_board(self, verbose=False):
        if not self.is_img_loaded():
            raise Exception("No image loaded")

        # run board detection

        # BoardDetection.board_extractor.set_img(self.img)
        self.board_detection, conf = BoardDetection.board_extractor.extract_board(
            verbose)
        self.board_detection = torch.tensor(self.board_detection)
        return self.board_detection, conf

    def warp(self):
        if not self.is_board_detected():
            raise Exception("Board not detected")

        warpped_img, M = BoardDetection.board_extractor.warp(
            self.board_detection.numpy())
        self.warped_img = warpped_img
        self.M = M
        return warpped_img, M

    def is_occluded(self):
        warped, _ = self.warp()
        warped_tensor = torch.from_numpy(warped).permute(
            2, 0, 1).unsqueeze(0).float() / 255.0
        self.occlusion_model.set_img(warped_tensor.to("cuda"))
        pred, conf = self.occlusion_model.is_occluded()
        return pred

    def recognize_clock(self):
        if not self.is_img_loaded():
            raise Exception("No image loaded")

        # run clock recognition

    def recognize_pieces(self, verbose=False):
        if not self.is_img_loaded():
            raise Exception("No image loaded")

        if not self.is_board_detected():
            raise Exception("No board detected")

        self.piece_detector.set_img(self.img, self.board_detection)

        if verbose:
            start_time = time.perf_counter()
        self.piece_detector.preprocess()
        if verbose:
            end_time = time.perf_counter()
            print(
                f"Piece Recognition - Preprocessing {(end_time-start_time)*1e3:.6f} ms")

        if verbose:
            start_time = time.perf_counter()
        self.piece_matrix = self.piece_detector.predict()
        if verbose:
            end_time = time.perf_counter()
            print(
                f"Piece Recognition - Processing {(end_time-start_time)*1e3:.6f} ms")
        self.orientation = self.piece_detector.guess_orientation()

        # convert piece matrix to fen
        self.fen = ChessUtils.ChessTensorUtils().tensorToFEN_MAX(
            self.piece_matrix)

    def save_fen_image(self, file_name="out_fen.png"):
        if not self.is_pieces_detected():
            raise Exception("Pieces not detected")

        fen_img = ChessUtils.fen_to_png(self.fen, ".", file_name)

    def preview_board(self):
        plt.imshow(self.img)
        plt.imshow(self.board_detection[2], alpha=self.board_detection[2])

    def preview_pieces(self):
        self.pieces_yolo.show()

    def apply(self):
        self.detect_board()
        self.recognize_clock()
        self.recognize_pieces()


class ChessLensGame:
    def __init__(self, piece_detector: Literal["cnn", "yolo",
                                               "cnn_onnx", "cnn_onnx_dynamic", "cnn_onnx_static"
                                               "yolo_onnx", "yolo_onnx_dynamic", "yolo_onnx_static"] | None = None, config=None):

        if (config is not None) and ("bd_period" in config):
            self.bd_period = config["bd_period"]
        else:
            self.bd_period = 5

        if (config is not None) and ("is_detect_occlusion" in config):
            self.is_detect_occlusion = config["is_detect_occlusion"]
        else:
            self.is_detect_occlusion = True

        if (config is not None) and ("is_detect_wakeup" in config):
            self.is_detect_wakeup = config["is_detect_wakeup"]
        else:
            self.is_detect_wakeup = True

        if (config is not None) and ("context_bredth" in config):
            self.context_bredth = config["context_bredth"]
        else:
            self.context_bredth = 50

        if (config is not None) and ("context_delay" in config):
            self.context_delay = config["context_delay"]
        else:
            self.context_delay = 120

        if (config is not None) and ("context_bind_period" in config):
            self.context_bind_period = config["context_bind_period"]
        else:
            self.context_bind_period = 1

        self.current_img = ChessLensImage(piece_detector=piece_detector)
        self.clear()
        self.piece_detector = piece_detector
        self.wakeup_module = WakeupModule.WakeupModule()

        self.avg_times = {
            "load": 0,
            "board_detection": 0,
            "piece_recognition": 0,
            "HMM": 0
        }

    def clear(self):
        self.board_detection = None
        self.orientation = None

        self.context_model = ChessHMM(
            self.context_bredth, self.context_delay, self.context_bind_period)
        self.pgn = None

        self.t = 0

    def set_img(self, img: str | torch.Tensor | np.ndarray, verbose=False):
        t1 = time.perf_counter()
        self.current_img.load_image(img)
        img = self.current_img
        self.process_img(verbose)
        self.t += 1
        t2 = time.perf_counter()

        self.avg_times["load"] += t2-t1

    def calc_orientation(self):
        piece_matrix = self.current_img.piece_matrix.clone().detach().squeeze()

        if not ("yolo" in self.piece_detector):
            piece_matrix = piece_matrix.argmax(dim=0)

        whites = (piece_matrix < 6).float()

        correct_r: int = (whites[:, 4:]).sum().item() - \
            (whites[:, :4]).sum().item()
        correct_l: int = -correct_r

        correct_t: int = (whites[:4, :]).sum().item() - \
            (whites[4:, :]).sum().item()
        correct_b: int = -correct_t

        vals = [correct_r, correct_l, correct_t, correct_b]
        orientations = ["r", "l", "t", "b"]
        self.orientation = orientations[vals.index(max(vals))]

    def detect_occlusion(self) -> bool:
        return self.current_img.is_occluded()

    def detect_wakeup(self) -> bool:
        warped_img, _ = self.current_img.warp()
        return self.wakeup_module.is_wakeup(warped_img)

    def prep_probs(self, probs) -> np.ndarray:
        # probs = self.current_img.piece_matrix

        if "yolo" in self.piece_detector:
            probs = torch.zeros(1, 13, 8, 8, dtype=torch.float32) + .1

            i = torch.arange(8).unsqueeze(1).expand(8, 8)
            j = torch.arange(8).unsqueeze(0).expand(8, 8)
            probs[0, self.current_img.piece_matrix.squeeze().to(
                torch.int32), i, j] = .9

        if self.orientation == "r":
            k = -1  # r
        elif self.orientation == "l":
            k = 1  # l
        elif self.orientation == "t":
            k = 2
        elif self.orientation == "b":
            k = 0
        else:
            raise Exception("Invalid Orientation")

        return -np.log(torch.rot90(probs, k=k, dims=(2, 3)).squeeze().permute(1, 2, 0).numpy()[::-1]+(1e-7))

    def process_img(self, verbose=False):
        t2 = time.perf_counter()
        img = self.current_img

        # Board Detection
        if self.t % self.bd_period == 0:
            try:
                img.detect_board()
                old_detection = self.board_detection
                new_detection = img.board_detection

                if old_detection is None:
                    self.board_detection = new_detection
                else:
                    self.board_detection = (new_detection + old_detection) / 2
            except:
                pass
        img.board_detection = self.board_detection

        # Wakup Detection
        if self.is_detect_wakeup:
            is_wakeup = self.detect_wakeup()
            if not is_wakeup:
                return

        # Occlusion Detection
        if self.is_detect_occlusion:
            is_occluded = self.detect_occlusion()
            if is_occluded:
                return
        t3 = time.perf_counter()

        # Frame Processing
        img.recognize_pieces()

        # Orientation
        if self.orientation is None:
            self.calc_orientation()
        t4 = time.perf_counter()

        if verbose:
            # print(img.fen)
            img.save_fen_image(f"game_fens/fen_{self.t}.png")

        # Context Awareness
        self.context_model.set_probs(
            self.context_model.model.top_t()+1, self.prep_probs(img.piece_matrix))
        t5 = time.perf_counter()

        self.avg_times["board_detection"] += t3-t2
        self.avg_times["piece_recognition"] += t4-t3
        self.avg_times["HMM"] += t5-t4

    def bind(self):
        if self.context_model.model.top_t() != self.context_model.model.top_bind_t():
            self.context_model.bind()

    def get_history(self, include_non_bound: bool = False):
        return self.context_model.get_history(include_non_bound)
