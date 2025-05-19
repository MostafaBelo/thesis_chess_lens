from ultralytics import YOLO
import torch

from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt


model_path = './BoardDetector_Yolo/bd_yolo.pt'
default_conf = 0.7


class BoardDetector_YOLO:
    def __init__(self, model_path=model_path, conf=default_conf):
        self.model = None
        self.clear()

        self.load_model(model_path)

    def clear(self):
        self.img = None
        self.data = None

    def check_model(self):
        if (self.model is None):
            self.load_model()

    def check_img(self):
        if (self.img is None):
            raise ValueError(
                "Image not set. Please set an image using set_img() method.")

    def check_data(self):
        if (self.data is None):
            self.process()

    def load_model(self, model_path=model_path):
        self.model = YOLO(model_path)

    def set_img(self, img):
        self.img = img
        img = self.preprocess()

    def preprocess(self):
        self.check_img()
        img = self.img

        img = Image.fromarray(img.numpy()).resize((640, 640))
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)

        self.img = img

    def process(self):
        self.check_model()
        self.check_img()

        img = self.img
        r = self.model(img, conf=0.7)

        mask = r[0].masks.data.squeeze().cpu()
        if (len(mask.shape) >= 3 and mask.shape[0] > 1):
            mask = mask[[0]]

        conf = r[0].boxes[0].conf.item()

        self.data = (mask, conf)

    def postprocess(self):
        self.check_data()
        mask = self.data[0]

        # apply opening to remove small masks
        open_size = 10  # 6
        dilate_size = 5
        morph_open_mask = np.ones((open_size, open_size), np.uint8)
        morph_dilate_mask = np.ones((dilate_size, dilate_size), np.uint8)
        eroded = cv2.erode(mask.numpy(), morph_open_mask, iterations=1)
        dilated = cv2.dilate(eroded, morph_open_mask, iterations=1)
        dilated = cv2.dilate(dilated, morph_dilate_mask, iterations=1)

        self.data = (dilated, self.data[1])

    def predict(self):
        self.process()
        self.postprocess()

        return self.data

    def plot_mask(self):
        self.check_img()
        self.check_data()

        img = self.img
        mask = self.data[0]

        plt.imshow(img.squeeze().permute(1, 2, 0))
        plt.imshow(mask, alpha=mask*.5, cmap='Reds')
