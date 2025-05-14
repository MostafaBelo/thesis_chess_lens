import BoundedCornerExtractor
import importlib
import torch

from matplotlib import pyplot as plt

from BoardDetector_YOLO import BoardDetector_YOLO

bd = BoardDetector_YOLO()

img = torch.tensor(plt.imread("../data/image_0.jpg"))
# img = torch.tensor(plt.imread("../data/data_manual/1741715439429.jpg"))

bd.set_img(img)
mask, conf = bd.predict()

importlib.reload(BoundedCornerExtractor)

img_gray = img.numpy()

CE = BoundedCornerExtractor.BoundedCornerExtractor()

CE.setImg(img_gray, mask)
CE.apply()
# M, mask_fit, error, iou_error = CE.fitBoard()

M, corners, mask_fit, error = CE.fitBoard()

print(error, corners)
