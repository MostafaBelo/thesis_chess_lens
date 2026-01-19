import cv2
import numpy as np
import time
import sys
import glob

# Checkerboard dimensions (internal corners)
CHECKERBOARD = (8, 8)  # Adjust to your pattern
square_size = 25  # Size of squares in mm

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

# Capture images - take 20-30 images of checkerboard from different angles
print("Press SPACE to capture, Q to quit and calibrate")

camera = "pi"  # pi / cv2

if camera in "pi":
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
    return img


img_count = 0
try:
    while True:
        frame = take_image()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        ret_corners, corners = cv2.findChessboardCorners(
            gray, CHECKERBOARD, None)

        # display = frame.copy()
        # if ret_corners:
        #     cv2.drawChessboardCorners(
        #         display, CHECKERBOARD, corners, ret_corners)
        #     cv2.putText(display, "Pattern found! Press SPACE", (10, 30),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # cv2.putText(display, f"Captured: {img_count}/20", (10, 70),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # cv2.imshow('Calibration', display)

        if ret_corners:
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            img_count += 1
            print(f"Captured image {img_count}")

            if img_count >= 20:
                break
except Exception as e:
    print(f"Exited due to error - {e}")
finally:
    if camera == "pi":
        picam2.stop()
    elif camera == "cv2":
        cap.release()
        cv2.destroyAllWindows()

if img_count < 10:
    print("Need at least 10 images for calibration!")
    exit()

# Calibrate camera for fisheye
print("Calibrating fisheye camera...")
h, w = gray.shape[:2]

# Fisheye calibration flags
calibration_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                     cv2.fisheye.CALIB_CHECK_COND +
                     cv2.fisheye.CALIB_FIX_SKEW)

# Initialize camera matrix
K = np.zeros((3, 3))
D = np.zeros((4, 1))

# Calibrate
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]

ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints, imgpoints, gray.shape[::-1], K, D, rvecs, tvecs,
    calibration_flags, criteria)

print("\nCalibration complete!")
print("Camera matrix (K):")
print(K)
print("\nDistortion coefficients (D):")
print(D)

# Save calibration
np.savez('fisheye_calibration.npz', K=K, D=D, img_size=(w, h))
print("\nCalibration saved to fisheye_calibration.npz")
