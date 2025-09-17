import json
import glob
import cv2
import numpy as np
import os
from Dataset.DataSetLoaders.ChessDataset import ChessDataset

dataset = ChessDataset()


# Path to your images (change if needed)
image_files = [os.path.join(dataset.root_dirs[0], label["image_path"])
               for label in dataset.labels]
image_files = image_files[300:301]

# Corner labels in the order we want to collect
corner_labels = ["top-left", "top-right", "bottom-right", "bottom-left"]

# Dictionary to store results: {image_file: {"top-left": (x,y), ...}, ...}
all_corners = {f: {} for f in image_files}

# Mouse callback


def plot_circle(point, img: np.ndarray) -> np.ndarray:
    img_copy = img.copy()
    cv2.circle(img_copy, point, 5, (0, 0, 255), -1)
    return img_copy


def click_event(event, x, y, flags, params):
    global point_selected, img, img_copy
    img_copy = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        point_selected = (x, y)
        cv2.circle(img_copy, point_selected, 5, (0, 0, 255), -1)
        plot_circle(point_selected, img_copy)
        cv2.imshow("Image", img_copy)


def collect_corners():
    global point_selected, img, img_copy, all_corners

    for corner_name in corner_labels:
        print(f"\n=== Select {corner_name} corners for all images ===\n")
        for file in image_files:
            img = cv2.imread(file)
            for tmp_corner_name in all_corners[file].keys():
                img = plot_circle(all_corners[file][tmp_corner_name], img)
            img_copy = img.copy()
            point_selected = None

            cv2.imshow("Image", img_copy)
            cv2.setMouseCallback("Image", click_event)

            print(f"Click {corner_name} corner for {file}, then press 'n'.")

            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('n') and point_selected is not None:
                    all_corners[file][corner_name] = point_selected
                    break
                if key == ord('q'):
                    exit()

    cv2.destroyAllWindows()
    return all_corners


if __name__ == "__main__":
    results = collect_corners()

    # Save to JSON for reuse
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "chessboard_corners.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("\nSaved corners to chessboard_corners.json")
