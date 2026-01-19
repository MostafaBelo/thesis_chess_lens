import os
from dotenv import load_dotenv
load_dotenv()  # noqa

import numpy as np
import cv2
from torchvision import transforms
from PIL import Image

import time
import sys
import os  # noqa

data_dir = ""
os.makedirs(data_dir, exist_ok=True)

img_count = 0

camera = "pi"  # pi / cv2

postprocess = None
postprocess_params = {}

if camera in ["pi", "pi130"]:
    sys.path.append("/usr/lib/python3/dist-packages")
    from picamera2 import Picamera2, Preview
    picam2 = Picamera2()
    camera_config = picam2.create_still_configuration()
    picam2.configure(camera_config)
    # picam2.set_controls({
    #     "AwbEnable": True,
    #     "AwbMode": 4
    # })
    picam2.start_preview(Preview.NULL)
    picam2.start()
    time.sleep(2)

    if camera == "pi130":
        calib = np.load(os.path.join(
            os.environ["WEIGHTS"], 'fisheye_calibration.npz'))
        K = calib['K']
        D = calib['D']
        img_size = tuple(calib['img_size'])

        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, img_size, cv2.CV_16SC2)

        def process(frame):
            undistorted = cv2.remap(frame, map1, map2,
                                    interpolation=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT)
            return undistorted

        postprocess = process

elif camera == "cv2":
    import cv2
    cap = cv2.VideoCapture(0)


def take_image():
    if camera in ["pi", "pi130"]:
        img = picam2.capture_array()
    elif camera == "cv2":
        ret, img = cap.read()  # Read frame continuously for live preview
        if not ret:
            cap.release()
            raise Exception("❌ Failed to capture image")
        img = img[:, :, ::-1]
    if postprocess is not None:
        img = postprocess(img)
    img = Image.fromarray(img).resize((640, 480))
    img.save(f"{data_dir}/img_{img_count}.jpg")


try:
    while True:
        # input(f"Image {img_count}:")
        time.sleep(.5)

        take_image()
        img_count += 1

        print("Saved Successfully")
except Exception as e:
    print(f"Exited due to error - {e}")
finally:
    if camera == "pi":
        picam2.stop()
    elif camera == "cv2":
        cap.release()
        cv2.destroyAllWindows()
