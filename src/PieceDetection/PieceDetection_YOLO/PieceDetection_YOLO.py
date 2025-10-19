from ultralytics import YOLO
import numpy as np
import torch
from torchvision import transforms
import cv2

from PIL import Image

import onnx
import onnxruntime as ort

from typing import Literal

import os

if "device" in os.environ:
    device = os.environ["device"]
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"


class PieceDetectorYOLOOnnx():
    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
        )

    def __call__(self, *args):
        return self.forward(*args)

    def forward(self, x: torch.Tensor):
        x = x.cpu().numpy()[:, ::-1, :, :]

        outputs = self.session.run(None, {"images": x})

        res: np.ndarray = outputs[0][0]

        confidence_threshold = 0.5
        nms_threshold = 0.45

        # OpenCV's NMS
        indices = cv2.dnn.NMSBoxes(
            # Convert to list of [x, y, w, h] or keep [x1, y1, x2, y2]
            bboxes=res[:4].T.tolist(),
            scores=res[4:].max(axis=0).tolist(),
            score_threshold=confidence_threshold,
            nms_threshold=nms_threshold
        )

        # Get filtered results
        if len(indices) > 0:
            res = res[:, indices.flatten()]

        res: torch.Tensor = torch.tensor(res)
        res = torch.cat([res[:4], res[4:].argmax(dim=0).unsqueeze(
            0), res[4:].max(dim=0).values.unsqueeze(0)], dim=0)

        return res


model = None
# model = YOLO(os.path.join(os.environ['WEIGHTS'], "piece_yolo.pt")).to(device)


def detect_pieces_img(img: torch.Tensor):
    if isinstance(model, PieceDetectorYOLOOnnx):
        results = model(img.unsqueeze(0))
        return conv_boxes_onnx(results), None
    else:
        results = model(img.unsqueeze(0), verbose=False)
        # results[0].show()
        return conv_boxes(results[0].boxes), results[0]


def conv_boxes_onnx(boxes):
    class_names = {0: 'white-king',
                   1: 'black-king',
                   2: 'white-queen',
                   3: 'black-queen',
                   4: 'white-rook',
                   5: 'black-rook',
                   6: 'white-bishop',
                   7: 'black-bishop',
                   8: 'white-knight',
                   9: 'black-knight',
                   10: 'white-pawn',
                   11: 'black-pawn'}
    # class_names = model.names
    class_ids = boxes[4].numpy().astype(int)
    confs = boxes[5]
    # (x1, y1, x2, y2) for each detection
    cx = boxes[0]
    cy = boxes[1]
    w = boxes[2]
    h = boxes[3]

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    coords = torch.stack([x1, y1, x2, y2], dim=1).numpy()

    # Combine class name and coordinates
    detections = []
    for cls_id, conf, box in zip(class_ids, confs, coords):
        class_name = class_names[cls_id]
        x1, y1, x2, y2 = box
        detections.append({
            "class": class_name,
            "class_id": cls_id,
            "conf": conf,
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2)
        })

    return detections


def conv_boxes(boxes):
    # class_names = {0: 'black-bishop', 1: 'black-king', 2: 'black-knight', 3: 'black-pawn', 4: 'black-queen', 5: 'black-rook',
    #                6: 'white-bishop', 7: 'white-king', 8: 'white-knight', 9: 'white-pawn', 10: 'white-queen', 11: 'white-rook'}
    class_names = model.names
    class_ids = boxes.cls.cpu().numpy().astype(int)
    coords = boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2) for each detection
    confs = boxes.conf

    # Combine class name and coordinates
    detections = []
    for cls_id, conf, box in zip(class_ids, confs, coords):
        class_name = class_names[cls_id]
        x1, y1, x2, y2 = box
        detections.append({
            "class": class_name,
            "class_id": cls_id,
            "conf": conf,
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
    board = torch.zeros(1, 8, 8, 2, dtype=torch.float32)
    board[:, :, :, 0] = 12

    det_classes = [det["class"] for det in boxes]
    det_confs = [det["conf"] for det in boxes]
    boxes_xyxy = torch.tensor(
        [[det["x1"], det["y1"], det["x2"], det["y2"]] for det in boxes])
    anchors: torch.Tensor = get_anchor(boxes_xyxy)
    warpped_anchors: torch.Tensor = apply_transform(anchors, M)
    sqs = (warpped_anchors // 32).to(torch.int32)
    sqs = [(sq, det_cls, det_conf)
           for sq, det_cls, det_conf in zip(sqs.tolist(), det_classes, det_confs)]

    channels = ['white-pawn', 'white-knight', 'white-bishop', 'white-rook', 'white-queen', 'white-king',
                'black-pawn', 'black-knight', 'black-bishop', 'black-rook', 'black-queen', 'black-king']
    for sq, piece, conf in sqs:
        if (sq[0] >= 0 and sq[0] < 8 and sq[1] >= 0 and sq[1] < 8):
            if board[0, sq[1], sq[0], 1] < conf:
                board[0, sq[1], sq[0], 0] = channels.index(piece)
                board[0, sq[1], sq[0], 1] = conf

    return sqs, board[:, :, :, 0].to(torch.int)


class PieceDetector:
    def __init__(self, model_type: Literal["torch", "onnx", "onnx_dynamic", "onnx_static"] = "torch"):
        self.img = None
        self.corners = None

        self.warpped_img = None
        self.M = None

        self.load_model(model_type)

    def load_model(self, model_type: Literal["torch", "onnx", "onnx_dynamic", "onnx_static"] = "torch"):
        global model
        match model_type:
            case "torch":
                model = YOLO(os.path.join(
                    os.environ['WEIGHTS'], "piece_yolo.pt")).to(device)

            case "onnx":
                model = PieceDetectorYOLOOnnx(os.path.join(
                    os.environ["WEIGHTS"], "piece_yolo.onnx"))

            case "onnx_dynamic":
                model = PieceDetectorYOLOOnnx(os.path.join(
                    os.environ["WEIGHTS"], "piece_yolo_quantized_dynamic.onnx"))

            case "onnx_static":
                model = PieceDetectorYOLOOnnx(os.path.join(
                    os.environ["WEIGHTS"], "piece_yolo_quantized_static.onnx"))

    def set_img(self, img: torch.Tensor, corners: torch.Tensor):
        self.img = Image.fromarray(
            (img.clone().detach().permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)).resize((640, 480))
        self.img = transforms.ToTensor()(self.img).flip((0,))  # 3, H, W
        self.corners = corners.clone().detach()  # 4, 2

        self.corners[:, 0] *= 640/640
        self.corners[:, 1] *= 480/640

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
