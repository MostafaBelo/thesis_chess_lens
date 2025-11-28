import json
import re
import numpy as np

ANNOT_FILE = "annotations.json"
LABELS_IN = "/mnt/D/University/Thesis_Dataset/chessred2k/labels.txt"
LABELS_OUT = "/mnt/D/University/Thesis_Dataset/chessred2k/labels_updated.txt"

# Regex to identify the coordinate group (8 comma-separated numbers)
COORD_REGEX = re.compile(r"\(\s*\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+\s*\)")


def main():
    # Load annotations.json
    with open(ANNOT_FILE, "r") as f:
        annotations = json.load(f)

    edited = annotations.get("coords", {})
    faulty = set(annotations.get("faulty", []))

    with open(LABELS_IN, "r") as f:
        lines = f.readlines()

    updated_lines = []

    for idx, line in enumerate(lines):

        # If this line is marked faulty → skip
        if idx in faulty:
            print(f"Skipping faulty line {idx}")
            continue

        # If this line has edited coordinates → replace them
        if str(idx) in edited:
            new_coords = edited[str(idx)]
            # Flatten into "x1,y1,x2,y2,x3,y3,x4,y4"
            new_coords = np.array(new_coords)
            # new_coords[:, 0] *= 640/480
            # new_coords[:, 1] *= 480/640
            coord_text = "(" + ",".join(map(lambda x: str(x),
                                            new_coords.flatten().astype(int).tolist())) + ")"

            # Only replace the FIRST coordinate group (the 4-point group)
            line = COORD_REGEX.sub(coord_text, line, count=1)

        updated_lines.append(line)

    # Write output
    with open(LABELS_OUT, "w") as f:
        f.writelines(updated_lines)

    print("\nDone.")
    print(f"Updated file saved to {LABELS_OUT}")
    print(f"Faulty lines removed: {sorted(faulty)}")


if __name__ == "__main__":
    main()
