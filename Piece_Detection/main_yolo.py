from ultralytics import YOLO
import numpy as np
import torch
import cv2

from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("./Piece_Detection/best6(11x).pt").to(device)


def detect_pieces_img(img):
    results = model(np.ascontiguousarray(img.cpu().numpy()))
    # results[0].show()

    return conv_boxes(results[0].boxes), results[0]


def conv_boxes(boxes):
    class_names = {0: 'black-bishop', 1: 'black-king', 2: 'black-knight', 3: 'black-pawn', 4: 'black-queen', 5: 'black-rook',
                   6: 'white-bishop', 7: 'white-king', 8: 'white-knight', 9: 'white-pawn', 10: 'white-queen', 11: 'white-rook'}
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


def get_anchor(det):
    return torch.tensor([(det["x1"] + det["x2"])/2, det["y2"] + (det["y1"]-det["y2"]) * .2])


def apply_transform(point, M):
    warpped = cv2.perspectiveTransform(
        point.unsqueeze(0).unsqueeze(0).cpu().numpy(), M.cpu().numpy())
    return torch.tensor(warpped).squeeze()


def align_boxes_to_board(boxes, board):
    M = board[0].inverse()

    sqs = []
    board = torch.ones(8, 8, dtype=torch.int8) * 12

    for det in boxes:
        anchor = get_anchor(det)
        warpped_anchor = apply_transform(anchor, M)

        sq = (int(warpped_anchor[0].item() // 32),
              int(warpped_anchor[1].item() // 32))
        sqs.append((sq, det["class"]))

    channels = ['white-pawn', 'white-knight', 'white-bishop', 'white-rook', 'white-queen', 'white-king',
                'black-pawn', 'black-knight', 'black-bishop', 'black-rook', 'black-queen', 'black-king']
    for sq, piece in sqs:
        if (sq[0] >= 0 and sq[0] < 8 and sq[1] >= 0 and sq[1] < 8):
            board[7-sq[0], sq[1]] = channels.index(piece)

    return sqs, board
