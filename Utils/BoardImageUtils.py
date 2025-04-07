import numpy as np
import torch
import cv2


def get_mask(corners: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    mask = np.zeros(size)
    mask = cv2.fillPoly(mask, [corners.numpy().astype(np.int32)], 1)
    return torch.tensor(mask, dtype=torch.float32)


def get_acc(corners_prime: torch.Tensor, corners: torch.Tensor, img_shape: tuple[int, int]) -> float:
    # IOU
    mask_prime = get_mask(corners_prime, img_shape)
    mask = get_mask(corners, img_shape)
    return get_acc_masks(mask, mask_prime)


def get_acc_masks(mask1: torch.Tensor, mask2: torch.Tensor) -> float:
    # IOU
    intersection = torch.sum(mask1 * mask2)
    union = torch.sum((mask1 + mask2) > 0)
    return (intersection / union).item()
