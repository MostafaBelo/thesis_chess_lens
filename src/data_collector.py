import os
from dotenv import load_dotenv
load_dotenv()  # noqa

import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

from ImageProvider.ImageProvider import ImageProvider

import time
import sys
import os  # noqa

data_dir = ""
os.makedirs(data_dir, exist_ok=True)

img_count = 0

camera = ImageProvider(interval=0.5)

try:
    while True:
        # input(f"Image {img_count}:")
        time.sleep(.5)

        img = camera.take_image()
        img = Image.fromarray(img).resize((640, 480))
        img.save(f"{data_dir}/img_{img_count}.jpg")

        img_count += 1

        print("Saved Successfully")
except Exception as e:
    print(f"Exited due to error - {e}")
finally:
    camera.quit()
