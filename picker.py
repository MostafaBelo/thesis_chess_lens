import numpy as np
from PIL import Image
import cv2

from CornerExtractor import CornerExtractor
CE = CornerExtractor(ClusterDeltas=(100, np.pi/180 * 10))

image_file_path = "data/image3.jpg"
img = np.array(Image.open(image_file_path))

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
CE.setImg(img_gray)
CE.apply()

# CE.interactivePlot()
CE.clustered_interactivePlot()
