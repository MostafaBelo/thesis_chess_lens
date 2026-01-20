import os
from dotenv import load_dotenv
load_dotenv()  # noqa

import numpy as np
import cv2

import time
import sys

from typing import Literal


class ImageProvider:
    def __init__(self, camera: Literal["pi", "pi130", "cv2", "files"] | None = None):
        if camera is None:
            self.camera = "cv2" if (
                "CAMERA" not in os.environ) else os.environ["CAMERA"]
        else:
            self.camera = camera
        self.postprocess = None

        if self.camera in ["pi", "pi130"]:
            sys.path.append("/usr/lib/python3/dist-packages")
            from picamera2 import Picamera2, Preview
            self.picam2 = Picamera2()
            camera_config = self.picam2.create_still_configuration()
            self.picam2.configure(camera_config)
            self.picam2.start_preview(Preview.NULL)
            self.picam2.start()
            time.sleep(2)

            if self.camera == "pi130":
                calib = np.load(os.path.join(
                    os.environ["WEIGHTS"], 'fisheye_calibration.npz'))
                K = calib['K']
                D = calib['D']
                img_size = tuple(calib['img_size'])

                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, D, np.eye(3), K, img_size, cv2.CV_16SC2)

                def process(frame) -> np.ndarray:
                    undistorted = cv2.remap(frame, map1, map2,
                                            interpolation=cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_CONSTANT)
                    return undistorted

                self.postprocess = process

        elif self.camera == "cv2":
            self.cap = cv2.VideoCapture(0)

    def take_image(self):
        if self.camera in ["pi", "pi130"]:
            img = self.picam2.capture_array()
        elif self.camera == "cv2":
            ret, img = self.cap.read()  # Read frame continuously for live preview
            if not ret:
                self.cap.release()
                raise Exception("❌ Failed to capture image")
            img = img[:, :, ::-1]
        if self.postprocess is not None:
            img = self.postprocess(img)
        return img

    def quit(self):
        if self.camera == "pi":
            self.picam2.stop()
        elif self.camera == "cv2":
            self.cap.release()
            cv2.destroyAllWindows()
