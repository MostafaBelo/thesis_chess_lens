from picamera2 import Picamera2, Preview
import os
from dotenv import load_dotenv
load_dotenv()  # noqa

from torchvision import transforms
from PIL import Image

import time
import sys
sys.path.append("/usr/lib/python3/dist-packages")

data_dir = ""

img_count = 0

picam2 = Picamera2()
camera_config = picam2.create_still_configuration()
picam2.configure(camera_config)
picam2.start_preview(Preview.NULL)
picam2.start()


def take_image() -> None:
    img = picam2.capture_array()
    img = Image.fromarray(img).resize((640, 480))
    img.save(f"{data_dir}/img_{img_count}.jpg")


try:
    while True:
        input(f"Image {img_count}:")

        take_image()
        img_count += 1

        print("Saved Successfully")
except Exception as e:
    print(f"Exited due to error - {e}")
finally:
    picam2.stop()
