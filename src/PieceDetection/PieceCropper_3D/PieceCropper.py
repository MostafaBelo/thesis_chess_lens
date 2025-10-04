from matplotlib import pyplot as plt
import numpy as np
import torch
import cv2


class CameraMapper:
    cam_default = np.array([0, 0, .5])

    @staticmethod
    def get_M(normal_points: np.ndarray, warp_points: np.ndarray):
        M = cv2.getPerspectiveTransform(normal_points, warp_points)
        return M

    @staticmethod
    def from_normal_to_warp(points: np.ndarray, M: np.ndarray):
        N = points.shape[0]
        points_ = np.concat([points, np.ones((N, 1))], axis=1)
        warpped_points = points_ @ M.T
        warpped_points = (warpped_points / warpped_points[:, [2]])[:, :2]
        return warpped_points

    @staticmethod
    def from_warp_to_normal(points: np.ndarray, M: np.ndarray):
        N = points.shape[0]
        M_inv = np.linalg.inv(M)
        points_ = np.concat([points, np.ones((N, 1))], axis=1)
        normal_points = points_ @ M_inv.T
        normal_points = (normal_points / normal_points[:, [2]])[:, :2]
        return normal_points

    @staticmethod
    def get_K_R(current_size: tuple[float, float], cam_point: np.ndarray = cam_default, f: float = 2739.79, original_size: tuple[float, float] = (3024, 4032)):
        H, W = current_size
        H_o, W_o = original_size

        s = 0
        cx = W/2
        cy = H/2

        fx = f * W/W_o
        fy = f * H/H_o

        K = np.array([
            [fx, s, cx],
            [0, fy, cy],
            [0,  0,  1]
        ])  # K: intrinsics of 3d -> 2d

        r_val = 1/np.sqrt(2)
        R_ = np.array([
            [1, 0, 0],
            [0, -r_val, -r_val],
            [0, r_val, -r_val],
        ])
        t = (-R_) @ cam_point.reshape((3, 1))

        R = np.concat([R_, t], axis=1)
        # R: extrinsics of World -> Camera
        R = np.concat([R, np.array([[0, 0, 0, 1]])], axis=0)

        return K, R

    @staticmethod
    def from_3d_to_2d(points: np.ndarray, K: np.ndarray, R: np.ndarray):
        N = points.shape[0]
        points_ = np.concat([points, np.ones((N, 1))], axis=1)
        res_points = points_ @ R.T
        res_points = (res_points / res_points[:, [3]])[:, :3]
        res_points = res_points @ K.T
        res_points = (res_points/res_points[:, [2]])[:, :2]
        return res_points

    @staticmethod
    def from_2d_to_3d(points: np.ndarray, K: np.ndarray, R: np.ndarray):
        N = points.shape[0]
        points_ = np.concat([points, np.ones((N, 1))], axis=1)
        res_points = points_ @ np.linalg.inv(K).T

        points_ = np.concat([res_points, np.ones((N, 1))], axis=1)
        res_points = points_ @ np.linalg.inv(R).T
        res_points = (res_points/res_points[:, [3]])[:, :3]

        return res_points

    @staticmethod
    def trace_ray(directions: np.ndarray, Z_0: np.ndarray, cam_point: np.ndarray = cam_default):
        N = directions.shape[0]
        if directions.shape[1] != 3 or len(directions.shape) != 2 \
                or cam_point.shape[0] != 3 or len(cam_point.shape) != 1 \
                or Z_0.shape[0] != N or len(Z_0.shape) != 1:
            raise Exception("Invalid input")

        t: np.ndarray = ((Z_0-cam_point[2]) /
                         (directions[:, 2] - cam_point[2]))
        p = cam_point + t.reshape(N, 1) * \
            (directions - cam_point.reshape(1, 3))
        return p


class PieceCropper:
    def __init__(self):
        self.img = None
        self.numpy_img = None
        self.corners = None

    def set_img(self, img: torch.Tensor, corners: torch.Tensor):
        self.img = img
        self.numpy_img = (img*255).permute(1, 2, 0).numpy().astype(np.uint8)
        self.corners = corners

        dst_pts = np.array([
            [0, 0],
            [256, 0],
            [256, 256],
            [0, 256]
        ], dtype=np.float32)
        self.M = CameraMapper.get_M(
            corners.numpy().astype(np.float32), dst_pts)

    def process_img(self, original_img_size=None):
        xs = np.linspace(0, 256, 9)
        ys = np.linspace(0, 256, 9)
        X, Y = np.meshgrid(xs, ys)
        grid = np.stack([X, Y], axis=2)

        if original_img_size is None:
            K, R = CameraMapper.get_K_R(self.img.shape[1:])
        else:
            K, R = CameraMapper.get_K_R(
                self.img.shape[1:], original_size=original_img_size)

        grid_2d = CameraMapper.from_warp_to_normal(grid.reshape(-1, 2), self.M)
        grid_3d = CameraMapper.trace_ray(
            CameraMapper.from_2d_to_3d(grid_2d, K, R), np.zeros((81)))
        grid_3d[:, 2] += .15
        grid_top = CameraMapper.from_3d_to_2d(grid_3d, K, R).reshape(9, 9, 2)
        grid_bottom = grid_2d.reshape(9, 9, 2)

        H, W = self.img.shape[1:]
        grid_top[:, :, 0] = np.clip(grid_top[:, :, 0], 0, W-1)
        grid_top[:, :, 1] = np.clip(grid_top[:, :, 1], 0, H-1)

        grid_bottom[:, :, 0] = np.clip(grid_bottom[:, :, 0], 0, W-1)
        grid_bottom[:, :, 1] = np.clip(grid_bottom[:, :, 1], 0, H-1)

        sq_size = (64, 64)

        res = torch.zeros(8, 8, sq_size[0], sq_size[1], 3)

        for r in range(8):
            for c in range(8):
                mask = np.zeros(self.img.shape[1:], dtype=np.uint8)
                points = np.concat(
                    [grid_bottom[r:r+2, c:c+2], grid_top[r:r+2, c:c+2]], axis=0).astype(np.int32).reshape(-1, 1, 2)
                hull = cv2.convexHull(points)
                # cv2.fillPoly(mask, hull, 255)
                cv2.fillConvexPoly(mask, hull, 255)

                x, y, w, h = cv2.boundingRect(points)
                roi = self.numpy_img[y:y+h, x:x+w]
                roi_mask = mask[y:y+h, x:x+w]

                # roi_masked = cv2.bitwise_and(roi, roi, mask=roi_mask)
                roi_masked = np.zeros_like(roi)
                roi_masked = np.where(roi_mask[..., None], roi, roi_masked)

                resized = cv2.resize(roi_masked, sq_size,
                                     interpolation=cv2.INTER_AREA)

                res[r, c] = torch.tensor(resized)

        res = res.permute(0, 1, 4, 2, 3).to(torch.float32) / 255
        return res


piece_cropper = PieceCropper()
