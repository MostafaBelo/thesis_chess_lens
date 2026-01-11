import os
import cv2
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import timm

from dotenv import load_dotenv
load_dotenv()

if "device" in os.environ:
    device = os.environ["device"]
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"


def correct_image(image):
    b, g, r = cv2.split(image.astype(np.float32))
    avg_b, avg_g, avg_r = np.mean(b), np.mean(g), np.mean(r)
    avg_gray = (avg_b + avg_g + avg_r) / 3

    b = b * (avg_gray / avg_b)
    g = g * (avg_gray / avg_g)
    r = r * (avg_gray / avg_r)

    return cv2.merge([b, g, r]).clip(0, 255).astype(np.uint8)


class TinyOcclusionCNN(nn.Module):
    # def __init__(self):
    #     super().__init__()
    #     self.net = nn.Sequential(
    #         nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    #         nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    #         nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
    #         nn.AdaptiveAvgPool2d(1),
    #         nn.Flatten(),
    #         nn.Linear(64, 2)
    #     )

    def __init__(self):
        super().__init__()
        self.net = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=False,
            num_classes=1
        )

        self.net.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.net.classifier.in_features, 1)
        )

    def forward(self, x):
        return self.net(x)


model = None


class OcclusionDetectorCNN:
    def __init__(self):
        self.img = None

        self.load_model()

    def load_model(self):
        global model
        model = TinyOcclusionCNN()
        model.load_state_dict(torch.load(
            os.path.join(os.environ['WEIGHTS'], "occlusion_cnn.pth"), map_location=device))
        model = model.to(device)

    def set_img(self, img: torch.Tensor):
        self.img = img

    def is_occluded(self):
        img = (self.img.squeeze().permute(1, 2, 0).flip(dims=(2,)).cpu().numpy()
               * 255).astype(np.uint8)
        img = correct_image(img)
        img = torch.tensor(
            img, dtype=torch.float32, device=device)/255
        img = img.permute(2, 0, 1).flip(dims=(0,)).unsqueeze(0)
        with torch.no_grad():
            out: torch.Tensor = model(img)
        pred = out.softmax(dim=1)

        return (pred > .5)[:, 0].item(), pred[:, 0].item()
