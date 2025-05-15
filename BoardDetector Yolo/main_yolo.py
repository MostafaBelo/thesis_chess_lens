import BoundedCornerExtractor
import importlib
import torch

from matplotlib import pyplot as plt

from BoardDetector_YOLO import BoardDetector_YOLO

import sys
file_path = "../data/image_0.jpg"
device = "cuda" if torch.cuda.is_available() else "cpu"
if len(sys.argv) >= 2:
    file_path = sys.argv[1]

    if len(sys.argv) >= 3:
        device = sys.argv[2]

bd = BoardDetector_YOLO()
bd.model.to(device)
print(bd.model.device)

img = torch.tensor(plt.imread(file_path))
# img = torch.tensor(plt.imread("../data/data_manual/1741715439429.jpg"))

import time

start_time = time.time()

# -----------

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

# -----------

end_time = time.time()
print(f"Time taken: {end_time - start_time}S")