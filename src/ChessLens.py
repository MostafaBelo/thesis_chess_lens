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

import os
import time


to_tensor = transforms.Compose([
    lambda x: Image.fromarray(x),
    transforms.ToTensor()
])


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

    def detect_board(self, verbose=False):
        if not self.is_img_loaded():
            raise Exception("No image loaded")

        # run board detection

        self.board_detection, conf = BoardDetection.board_extractor.extract_board(
            self.img,
            verbose)
        if type(self.board_detection) != torch.Tensor:
            self.board_detection = torch.tensor(self.board_detection)
        return self.board_detection, conf

    def warp(self):
        if not self.is_board_detected():
            raise Exception("Board not detected")

        warpped_img, M = BoardDetection.board_extractor.warp(
            self.img,
            self.board_detection.numpy())
        self.warped_img = warpped_img
        self.M = M
        return warpped_img, M

    def is_occluded(self):
        warped, _ = self.warp()
        warped_tensor = to_tensor(warped)
        pred, conf = self.occlusion_model.is_occluded(warped_tensor)
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

        if verbose:
            start_time = time.perf_counter()
        # self.piece_detector.preprocess()
        # self.piece_matrix = self.piece_detector.predict()
        self.piece_matrix = self.piece_detector.process(
            self.img, self.board_detection)
        # if verbose:
        #     end_time = time.perf_counter()
        #     print(
        #         f"Piece Recognition - Preprocessing {(end_time-start_time)*1e3:.6f} ms")
        # if verbose:
        #     start_time = time.perf_counter()
        if verbose:
            end_time = time.perf_counter()
            # print(f"Piece Recognition - Processing {(end_time-start_time)*1e3:.6f} ms")
            print(f"Piece Recognition {(end_time-start_time)*1e3:.6f} ms")
        # self.orientation = self.piece_detector.guess_orientation()

        # convert piece matrix to fen
        self.fen = ChessUtils.ChessTensorUtils().tensorToFEN_MAX(
            self.piece_matrix)

    def get_fen_img(self):
        if not self.is_pieces_detected():
            raise Exception("Pieces not detected")

        fen_img = ChessUtils.fen_to_png(
            self.fen, ".", file_name="", is_write=False)
        return fen_img

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

        if (config is not None) and ("wakeup_period" in config):
            self.wakeup_period = config["wakeup_period"]
        else:
            self.wakeup_period = 10

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

        if (config is not None) and ("context_continous" in config):
            self.context_continous = config["context_continous"]
        else:
            self.context_continous = False

        if (config is not None) and ("game_out_path" in config):
            self.game_out_path = config["game_out_path"]
        else:
            self.game_out_path = None

        if (config is not None) and ("fen_update" in config):
            self.fen_update = config["fen_update"]
        else:
            self.fen_update = None

        self.current_img = ChessLensImage(piece_detector=piece_detector)
        self.clear()
        self.piece_detector = piece_detector
        self.wakeup_module = WakeupModule.WakeupModule()

        self.avg_times = {
            "load": 0,
            "board_detection": 0,
            "wakeup": 0,
            "occlusion": 0,
            "piece_recognition": 0,
            "HMM": 0
        }

        self.broadcasted_fens = []
        self.last_num = 0
        self.latest_bound_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

    def clear(self):
        self.board_detection = None
        self.orientation = None

        self.context_model = ChessHMM(
            self.context_bredth, self.context_delay, self.context_bind_period)
        self.pgn = None

        self.t = 0
        self.last_wakeup = 0

    def set_img(self, img: str | torch.Tensor | np.ndarray, verbose=False):
        t1 = time.perf_counter()
        self.current_img.load_image(img)
        img = self.current_img
        t2 = time.perf_counter()
        is_wake_up = self.process_img(verbose)
        self.t += 1
        t3 = time.perf_counter()

        self.avg_times["load"] += t2-t1

        return (is_wake_up == True)

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
        print("Orientation:", self.orientation)

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
        img = self.current_img

        isbound = self.context_model.check_bind(self.t)
        print("bound", isbound, self.t)
        if isbound or self.context_continous:
            self.get_latest_fens()

        t1 = time.perf_counter()
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
            except Exception as e:
                pass
                # raise e
                # print(f"Failed to detect board - {e}")
        img.board_detection = self.board_detection
        t2 = time.perf_counter()

        # Wakup Detection
        if self.is_detect_wakeup:
            if self.last_wakeup - self.t >= self.wakeup_period:
                is_wakeup = True
            else:
                is_wakeup = self.detect_wakeup()
            if not is_wakeup:
                print("Not Awake")
                if (not verbose):
                    return
            else:
                print("Awake")
                self.last_wakeup = self.t
        t3 = time.perf_counter()

        # Occlusion Detection
        if self.is_detect_occlusion:
            is_occluded = self.detect_occlusion()
            if is_occluded:
                print(f"Occluded - {is_occluded}")
                if (not verbose):
                    return
            else:
                print("Not Occluded")
        t4 = time.perf_counter()

        # Frame Processing
        img.recognize_pieces()

        # Orientation
        if self.orientation is None:
            self.calc_orientation()
        t5 = time.perf_counter()

        if verbose:
            fen_img = img.get_fen_img()
            fen_latest_img = ChessUtils.fen_to_png(
                self.latest_bound_fen, ".", file_name="", is_write=False)
            img_np = ((img.img).permute(1, 2, 0).cpu().numpy()
                      * 255).astype(np.uint8)
            img_np = np.ascontiguousarray(img_np)
            corners = img.board_detection.clone().detach().to(torch.int32)
            for _ in range(len(corners)):
                cv2.circle(img_np, (corners[_, 0].item(), corners[_, 1].item()),
                           radius=5, color=(33, 158, 188), thickness=-1)
            img_np[:200, -200:] = fen_img
            img_np[-200:, -200:] = fen_latest_img
            cv2.putText(img_np, "Awake" if is_wakeup else "Not Awake", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if is_wakeup else (255, 0, 0), 2)
            cv2.putText(img_np, "Occluded" if is_occluded else "Not Occluded", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0) if is_occluded else (0, 255, 0), 2)
            img_out = Image.fromarray(img_np).convert("RGB")
            if (self.game_out_path is not None):
                img_out.save(os.path.join(
                    self.game_out_path, f"img_{self.t}.jpg"))

        # Context Awareness
        if (is_wakeup) and (not is_occluded):
            self.context_model.set_probs(
                self.context_model.model.top_t()+1, self.prep_probs(img.piece_matrix), self.t)
        t6 = time.perf_counter()

        self.avg_times["board_detection"] += t2-t1
        self.avg_times["wakeup"] += t3-t2
        self.avg_times["occlusion"] += t4-t3
        self.avg_times["piece_recognition"] += t5-t4
        self.avg_times["HMM"] += t6-t5

        return True

    def bind(self):
        if self.context_model.model.top_t() != self.context_model.model.top_bind_t():
            self.context_model.bind()

    def get_history(self, include_non_bound: bool = False):
        return self.context_model.get_history(include_non_bound)

    def get_latest_fens(self):
        hist = self.get_history(self.context_continous)
        fens = []
        for i in range(hist.shape[0]):
            fens.append(
                ChessUtils.ChessTensorUtils.tensorToFEN_MAX(hist[[i], ::-1]))

        fens = [
            f"{fen}" for fen in fens if not (fen in self.broadcasted_fens)]
        self.broadcasted_fens += fens
        # for i in range(len(fens)):
        #   fens[i] += f"{self.last_num+i}"
        self.last_num += len(fens)
        print("FENS:", fens)

        if (self.fen_update is not None) and (len(fens) >= 1):
            self.latest_bound_fen = fens[-1]
            self.fen_update(fens[-1])

        # self.broadcasted_fens += fens
        if (self.game_out_path is not None) and len(fens) > 0:
            with open(os.path.join(self.game_out_path, "game_fens.csv"), "a") as f:
                f.write("\n" + "\n".join(fens))
