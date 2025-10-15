from ultralytics import YOLO
import numpy as np
import torch
import cv2

from PIL import Image

import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = YOLO(os.path.join(os.path.dirname(
#     os.path.abspath(__file__)), "best_piece_yolo.pt")).to(device)
model = YOLO(os.path.join(os.environ['WEIGHTS'], "piece_yolo.pt")).to(device)


def detect_pieces_img(img: torch.Tensor):
    # tmp = torch.tensor(np.ascontiguousarray(img.cpu().numpy()))
    results = model(img.unsqueeze(0), verbose=False)
    # results[0].show()

    return conv_boxes(results[0].boxes), results[0]


def conv_boxes(boxes):
    # class_names = {0: 'black-bishop', 1: 'black-king', 2: 'black-knight', 3: 'black-pawn', 4: 'black-queen', 5: 'black-rook',
    #                6: 'white-bishop', 7: 'white-king', 8: 'white-knight', 9: 'white-pawn', 10: 'white-queen', 11: 'white-rook'}
    class_names = model.names
    class_ids = boxes.cls.cpu().numpy().astype(int)
    coords = boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2) for each detection

    # Combine class name and coordinates
    detections = []
    for cls_id, box in zip(class_ids, coords):
        class_name = class_names[cls_id]
        x1, y1, x2, y2 = box
        detections.append({
            "class": class_name,
            "class_id": cls_id,
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2)
        })

    return detections


def get_anchor(dets: torch.Tensor) -> torch.Tensor:
    return torch.stack([(dets[:, 0] + dets[:, 2])/2, dets[:, 3] + (dets[:, 1]-dets[:, 3]) * .2], dim=1)


def apply_transform(points: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    N = points.shape[0]
    hom = torch.cat(
        # (N,3)
        [points, torch.ones(N, 1, device=points.device, dtype=points.dtype)], dim=1)

    M = M.to(torch.float32)
    hom = hom.to(torch.float32)
    proj = (M @ hom.t()).t()   # (N,3)
    w = proj[:, [2]]
    # avoid dividing by zero: add small eps if desired
    return proj[:, :2] / (w + 1e-12)


def align_boxes_to_board(boxes, M):
    sqs = []
    board = torch.ones(1, 8, 8, dtype=torch.int8) * 12

    det_classes = [det["class"] for det in boxes]
    boxes_xyxy = torch.tensor(
        [[det["x1"], det["y1"], det["x2"], det["y2"]] for det in boxes])
    anchors: torch.Tensor = get_anchor(boxes_xyxy)
    warpped_anchors: torch.Tensor = apply_transform(anchors, M)
    sqs = (warpped_anchors // 32).to(torch.int32)
    sqs = [(sq, det_cls) for sq, det_cls in zip(sqs.tolist(), det_classes)]

    channels = ['white-pawn', 'white-knight', 'white-bishop', 'white-rook', 'white-queen', 'white-king',
                'black-pawn', 'black-knight', 'black-bishop', 'black-rook', 'black-queen', 'black-king']
    for sq, piece in sqs:
        if (sq[0] >= 0 and sq[0] < 8 and sq[1] >= 0 and sq[1] < 8):
            board[0, sq[1], sq[0]] = channels.index(piece)

    return sqs, board


class PieceDetector:
    def __init__(self):
        self.img = None
        self.corners = None

        self.warpped_img = None
        self.M = None

    def set_img(self, img: torch.Tensor, corners: torch.Tensor):
        self.img = img  # 3, H, W
        self.corners = corners  # 4, 2

    def _warp(self, padding=(0, 0)):
        img_size = (256, 256)
        quad = self.corners.numpy().astype(np.float32)
        dst = np.array([
            [0, 0],
            [img_size[0], 0],
            [img_size[0], img_size[1]],
            [0, img_size[1]]
        ], dtype=np.float32)
        dst[:, 0] += padding[0]
        dst[:, 1] += padding[1]

        M = cv2.getPerspectiveTransform(quad, dst)
        M = torch.tensor(M, dtype=torch.float32)

        return M

    def preprocess(self):
        M = self._warp(padding=(0, 0))
        self.M = M

    def predict(self):
        # run yolo piece detection and set piece_bboxs
        self.piece_bboxs, self.pieces_yolo = detect_pieces_img(self.img)

        # align piece bboxs to squares using detected board
        sqs, self.piece_matrix = align_boxes_to_board(
            self.piece_bboxs, self.M)

        return self.piece_matrix
