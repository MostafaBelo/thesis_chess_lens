import cv2
import numpy as np
import glob

CHECKERBOARD = (8, 8)
square_size = 25

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

corners = []
current_img = None


def mouse_callback(event, x, y, flags, param):
    global corners, current_img
    if event == cv2.EVENT_LBUTTONDOWN:
        corners.append([x, y])
        cv2.circle(current_img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(current_img, str(len(corners)), (x+10, y+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imshow('Select Corners', current_img)


images = sorted(glob.glob(
    '/mnt/D/University/Thesis_Dataset/Temp/photos2/photos/acceptable/*.jpg'))
print(f"Found {len(images)} images")

cv2.namedWindow('Select Corners')
cv2.setMouseCallback('Select Corners', mouse_callback)

for idx, img_path in enumerate(images):
    print(f"\nImage {idx+1}/{len(images)}: {img_path}")
    print(f"Click {CHECKERBOARD[0] * CHECKERBOARD[1]} corners")
    print("Start from TOP-LEFT corner, go RIGHT, then next row")
    print("Press SPACE when done, R to restart, S to skip")

    frame = cv2.imread(img_path)
    corners = []
    current_img = frame.copy()

    cv2.imshow('Select Corners', current_img)

    while True:
        key = cv2.waitKey(1)

        if key == ord(' '):
            if len(corners) == CHECKERBOARD[0] * CHECKERBOARD[1]:
                corners_array = np.array(
                    corners, dtype=np.float32).reshape(-1, 1, 2)
                objpoints.append(objp)
                imgpoints.append(corners_array)
                print(f"✓ Corners accepted for image {idx+1}")
                break
            else:
                print(
                    f"✗ Need {CHECKERBOARD[0] * CHECKERBOARD[1]} corners, got {len(corners)}")

        elif key == ord('r'):
            corners = []
            current_img = frame.copy()
            cv2.imshow('Select Corners', current_img)
            print("Reset - click corners again")

        elif key == ord('s'):
            print("Skipped")
            break

        elif key == ord('q'):
            break

    if key == ord('q'):
        break

cv2.destroyAllWindows()

if len(objpoints) < 5:
    print("Need at least 5 images for calibration!")
    exit()

# Calibrate
print(f"\nCalibrating with {len(objpoints)} images...")
h, w = frame.shape[:2]

calibration_flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                     cv2.fisheye.CALIB_CHECK_COND +
                     cv2.fisheye.CALIB_FIX_SKEW)

K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in range(len(objpoints))]

ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints, imgpoints, (w, h), K, D, rvecs, tvecs,
    calibration_flags, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

print("\n✓ Calibration complete!")
print("Camera matrix (K):")
print(K)
print("\nDistortion coefficients (D):")
print(D)

np.savez('fisheye_calibration.npz', K=K, D=D, img_size=(w, h))
print("\nCalibration saved to fisheye_calibration.npz")
