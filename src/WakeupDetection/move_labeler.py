import cv2
import numpy as np
import json

from Dataset.DataSetLoaders import ChessDataset


def label_frames(dataset, output_file="labels.json"):
    """
    Label frames with move numbers.
    SPACE = same move, N = new move, Q = quit
    """
    # Load existing or start fresh
    try:
        with open(output_file, 'r') as f:
            labels = json.load(f)
    except:
        labels = []

    current_move = labels[-1] if labels else 0
    idx = len(labels)

    print("SPACE=same move, N=new move, Q=quit")

    print(f"Current Move: {current_move}")
    while idx < len(dataset):
        # Get image
        img, _ = dataset[idx]
        img = img.cpu().permute(1, 2, 0).numpy()[:, :, ::-1]
        img = (img*255).astype(np.uint8)

        # Show info
        # cv2.putText(img, f"Frame {idx} | Move {current_move}",
        #             (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Labeler", img)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            labels.append(current_move)
            idx += 1
        elif key == ord('n'):
            current_move += 1
            print(f"Current Move: {current_move}")
            labels.append(current_move)
            idx += 1

    # Save
    with open(output_file, 'w') as f:
        json.dump(labels, f)

    cv2.destroyAllWindows()
    print(f"Saved {len(labels)} labels")


# Usage:
ds = ChessDataset.GameDataset(
    config={
        "img_size": (640, 640),
        "include_only": "valid"
    }
)
label_frames(ds, "labels.json")
