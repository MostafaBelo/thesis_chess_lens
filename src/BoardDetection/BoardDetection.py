# from BoardDetection.BoardDetector_YOLO_Saddle import Bounded_Saddle_Yolo
from BoardDetection.BoardDetector_Saddle import BoardDetection_saddle

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2

import torch
from torchvision import transforms

# import time

# bd = BoardDetector_YOLO()
# bd = BoardExtractor()
bd = None
transform = transforms.Compose([
    transforms.ToTensor()
])


class InvalidImage(Exception):
    pass


to_np_500 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((500, 500)),
    lambda x: np.array(x)
])


class BoardExtractor:
    def __init__(self):
        global bd
        self.img = None
        self.img_gray = None

        # bd = Bounded_Saddle_Yolo.BoardExtractor()
        bd = BoardDetection_saddle

    def _order_points_rotation_proof(self, pts):
        pts = np.array(pts, dtype="float32")
        center = np.mean(pts, axis=0)

        # Compute angle for each point relative to center
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])

        # Sort points by angle (counterclockwise)
        ordered = pts[np.argsort(angles)]

        # After sorting by angle, determine which is top-left
        # Top-left = smallest x+y among points
        topmost_index = np.argmin((ordered**2).sum(axis=1))
        ordered = np.roll(ordered, -topmost_index, axis=0)

        return ordered

    def extract_board(self, img: torch.Tensor, verbose=False):
        # return bd.extract_board(verbose)
        img = to_np_500(img)
        # img = (img.clone().detach().permute(
        # 1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        # img = np.array(Image.fromarray(img).resize((500, 500)))
        corners = bd.detect(img)
        corners = self._order_points_rotation_proof(corners)
        corners = torch.tensor(corners, dtype=torch.float32)
        # print(corners)
        return corners, 1

    def warp(self, img: torch.Tensor, quad, padding=(0, 0)):
        return bd.warp(img, quad)


board_extractor = BoardExtractor()
