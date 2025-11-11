from BoardDetection.BoardDetector_YOLO_Saddle import Bounded_Saddle_Yolo
# import BoardExtractor as SaddleBoardExtractor

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


# class BoardExtractor:
#     def __init__(self):
#         self.img = None
#         self.img_gray = None

#     def set_img(self, img: str | np.ndarray | torch.Tensor):
#         if type(img) == str:
#             img = transform(Image.open(img).convert("RGB").resize((640, 640)))
#         elif type(img) == np.ndarray:
#             if len(img.shape) != 3:
#                 raise InvalidImage("Invalid Image")
#             if img.shape[0] == 3:
#                 self.img = torch.tensor(img).permute(1, 2, 0)
#                 if self.img.max() > 1.5:
#                     self.img = self.img.to(torch.float32) / 255
#             elif img.shape[2] == 3:
#                 self.img = torch.tensor(img)
#                 if self.img.max() > 1.5:
#                     self.img = self.img.to(torch.float32) / 255
#             else:
#                 raise InvalidImage("Invalid Image")
#         elif type(img) == torch.Tensor:
#             if len(img.shape) != 3:
#                 raise InvalidImage("Invalid Image")
#             if img.shape[0] == 3:
#                 self.img = img.permute(1, 2, 0)
#                 if self.img.max() > 1.5:
#                     self.img = self.img.to(torch.float32) / 255
#             elif img.shape[2] == 3:
#                 self.img = img
#                 if self.img.max() > 1.5:
#                     self.img = self.img.to(torch.float32) / 255
#             else:
#                 raise InvalidImage("Invalid Image")
#         else:
#             raise InvalidImage("Invalid Image")

#         self.img_gray = self.img.mean(dim=2)

#     def _detect_board_img(self):
#         img = (self.img * 255).to(torch.uint8)
#         bd.set_img(img)
#         mask, conf = bd.predict()
#         return mask, conf

#     def _remove_small_components(self, mask):
#         # Ensure binary 0/1 mask
#         mask_bin = (mask > 0).astype(np.uint8)

#         # Find connected components
#         num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
#             mask_bin, connectivity=8)

#         if num_labels <= 1:
#             return mask_bin  # No components

#         # The background is label 0, so start from 1
#         largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

#         # Keep only the largest
#         largest_component = (labels == largest_label).astype(np.uint8)

#         return largest_component

#     def _getSaddle(self, img: torch.Tensor):
#         img = img.numpy().astype(np.float64)
#         gx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
#         gy = cv2.Sobel(img, cv2.CV_64F, 0, 1)
#         gxx = cv2.Sobel(gx, cv2.CV_64F, 1, 0)
#         gyy = cv2.Sobel(gy, cv2.CV_64F, 0, 1)
#         gxy = cv2.Sobel(gx, cv2.CV_64F, 0, 1)

#         S = gxx*gyy - gxy**2
#         return S

#     def _nonmax_suppression(self, S, thresh=0.1, window_size=5):
#         # Take absolute value for saddle strength
#         S_abs = np.abs(S)

#         # Normalize to [0,1]
#         S_norm = S_abs / S_abs.max()

#         # Threshold
#         mask = S_norm > thresh

#         # Dilation to find local maxima
#         dilated = cv2.dilate(S_norm, np.ones(
#             (window_size, window_size), np.uint8))
#         local_max = (S_norm == dilated)

#         # Keep only local maxima above threshold
#         final_mask = mask & local_max

#         # Get coordinates
#         pts = np.column_stack(np.nonzero(final_mask))[:, [1, 0]]
#         return pts, final_mask

#     def _best_fit_quad(self, pts):
#         # Step 1: convex hull
#         hull = cv2.convexHull(pts)

#         # Step 2: try different epsilons to get 4 vertices
#         best_quad = None
#         best_score = -1

#         for eps_factor in np.linspace(0.001, 0.05, 50):
#             approx = cv2.approxPolyDP(
#                 hull, eps_factor * cv2.arcLength(hull, True), True)
#             if len(approx) == 4:
#                 quad = approx.reshape(4, 2)

#                 # Step 3: create mask and count coverage
#                 # adjust size if needed
#                 mask = np.zeros((640, 640), dtype=np.uint8)
#                 scaled_quad = np.int32(quad)  # Ensure int
#                 cv2.fillPoly(mask, [scaled_quad], 255)

#                 # Scale/shift points into mask coordinates
#                 min_xy = pts.min(axis=0)
#                 shifted_pts = (pts - min_xy).astype(int)
#                 shifted_quad = (quad - min_xy).astype(int)
#                 mask = np.zeros(
#                     (shifted_pts[:, 1].max()+5, shifted_pts[:, 0].max()+5), dtype=np.uint8)
#                 cv2.fillPoly(mask, [shifted_quad], 255)

#                 coverage = np.sum(
#                     mask[shifted_pts[:, 1], shifted_pts[:, 0]] > 0)

#                 # Score = coverage + area importance
#                 area = cv2.contourArea(shifted_quad)
#                 score = coverage + 0.001 * area

#                 if score > best_score:
#                     best_score = score
#                     best_quad = quad

#         return best_quad

#     def _order_points_rotation_proof(self, pts):
#         pts = np.array(pts, dtype="float32")
#         center = np.mean(pts, axis=0)

#         # Compute angle for each point relative to center
#         angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])

#         # Sort points by angle (counterclockwise)
#         ordered = pts[np.argsort(angles)]

#         # After sorting by angle, determine which is top-left
#         # Top-left = smallest x+y among points
#         topmost_index = np.argmin(ordered.sum(axis=1))
#         ordered = np.roll(ordered, -topmost_index, axis=0)

#         return ordered

#     def _iou(self, quad_pts, yolo_mask):
#         # quad_pts: np.array shape (4,2), dtype float32
#         h, w = yolo_mask.shape[:2]

#         # Create mask from quad
#         quad_mask = np.zeros((h, w), dtype=np.uint8)
#         cv2.fillPoly(quad_mask, [quad_pts.astype(np.int32)], 1)

#         # Intersection and union
#         intersection = np.logical_and(quad_mask, yolo_mask)
#         union = np.logical_or(quad_mask, yolo_mask)

#         return intersection.sum() / union.sum()

#     def extract_board(self):
#         # start = time.time()

#         mask, conf = self._detect_board_img()
#         mask = self._remove_small_components(mask)

#         if len(self.img_gray.shape) != 2 or len(mask.shape) != 2 or self.img_gray.shape[0] != mask.shape[0] or self.img_gray.shape[1] != mask.shape[1]:
#             raise InvalidImage("Invalid Image")

#         S = self._getSaddle((self.img_gray * mask))
#         S = -S
#         S[S < 0] = 0
#         pts, _ = self._nonmax_suppression(S, .01, 20)

#         quad = self._best_fit_quad(pts)
#         quad_ordered = self._order_points_rotation_proof(quad)

#         # end = time.time()
#         # print(f"Total Time: {end-start:.4f}")

#         iou = self._iou(quad_ordered, mask)
#         # print(conf, iou)
#         conf *= iou

#         return quad_ordered, conf

#     def warp(self, quad, padding=(0,0)):
#         img_size = (256, 256)
#         quad = quad.astype(np.float32)
#         dst = np.array([
#             [0, 0],
#             [img_size[0], 0],
#             [img_size[0], img_size[1]],
#             [0, img_size[1]]
#         ], dtype=np.float32)
#         dst[:, 0] += padding[0]
#         dst[:, 1] += padding[1]

#         M = cv2.getPerspectiveTransform(quad, dst)
#         warpped = cv2.warpPerspective(self.img.numpy(), M, (img_size[0]+2*padding[0], img_size[1]+2*padding[1]))

#         return warpped, M

class BoardExtractor:
    def __init__(self):
        global bd
        self.img = None
        self.img_gray = None

        bd = Bounded_Saddle_Yolo.BoardExtractor()

    def set_img(self, img: torch.Tensor):
        self.img = img

    def extract_board(self, verbose=False):
        bd.set_img(self.img)
        return bd.extract_board(verbose)

    def warp(self, quad, padding=(0, 0)):
        return bd.warp(quad)


board_extractor = BoardExtractor()
