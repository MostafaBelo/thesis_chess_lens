import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

import pickle
from PIL import Image

images_path = "/home/mostafaelfaggal/ChessReD/Chess Recognition Dataset (ChessReD)_2_all/chessred"
# images_path = "/home/mostafaelfaggal/ChessReD/Chess Recognition Dataset (ChessReD)_2_all/chessred2k/images"

labels_file_path = "/home/mostafaelfaggal/ChessReD/Chess Recognition Dataset (ChessReD)_2_all/chessred_annotations.pkl"
# labels_file_path = "/home/mostafaelfaggal/ChessReD/Chess Recognition Dataset (ChessReD)_2_all/chessred2k_annotations.pkl"


def img_transform(img_size=(3024, 3024), pad_type="repeat", crop_anchor="center"):
    def transform_fn(x: torch.Tensor) -> torch.Tensor:
        if (img_size[0] < x.shape[0]):
            top = 0
            bottom = x.shape[0]

            match crop_anchor:
                case "center":
                    top = (x.shape[0] - img_size[0]) // 2
                    bottom = top + img_size[0]

                case _:
                    pass

            x = x[top:bottom, :, :]
        elif (img_size[0] > x.shape[0]):
            diff = img_size[0] - x.shape[0]
            top_diff = diff//2
            bottom_diff = diff - top_diff

            top_pad = torch.zeros(0, x.shape[1], x.shape[2])
            bottom_pad = torch.zeros(0, x.shape[1], x.shape[2])

            match pad_type:
                case "repeat":
                    top_pad = x[[0], :, :].repeat(1, top_diff, 1)
                    bottom_pad = x[[-1], :, :].repeat(1, bottom_diff, 1)

                case "zero":
                    top_pad = torch.zeros(top_diff, x.shape[1], x.shape[2])
                    bottom_pad = torch.zeros(
                        bottom_diff, x.shape[1], x.shape[2])

                case _:
                    pass

            x = torch.cat([top_pad, x, bottom_pad], dim=0)

        if (img_size[1] < x.shape[1]):
            left = 0
            right = x.shape[1]

            match crop_anchor:
                case "center":
                    left = (x.shape[1] - img_size[1]) // 2
                    right = left + img_size[1]

                case _:
                    pass

            x = x[:, left:right, :]
        elif (img_size[1] > x.shape[1]):
            diff = img_size[1] - x.shape[1]
            left_diff = diff//2
            right_diff = diff - left_diff

            left_pad = torch.zeros(x.shape[0], 0, x.shape[2])
            right_pad = torch.zeros(x.shape[0], 0, x.shape[2])

            match pad_type:
                case "repeat":
                    left_pad = x[:, [0], :].repeat(1, left_diff, 1)
                    right_pad = x[:, [-1], :].repeat(1, right_diff, 1)

                case "zero":
                    left_pad = torch.zeros(x.shape[0], left_diff, x.shape[2])
                    right_pad = torch.zeros(
                        x.shape[1], right_diff, x.shape[2])

                case _:
                    pass

            x = torch.cat([left_pad, x, right_pad], dim=1)

        return x

    return transform_fn


class ChessRecDataset(Dataset):  # chess recognition dataset
    def __init__(self, images_path=images_path, labels_file_path=labels_file_path, label_key="corners", transform_fn=img_transform()):
        self.images_path = images_path
        self.labels_file_path = labels_file_path
        self.label_key = label_key

        self.transform_fn = transform_fn

        self.data = []
        with open(self.labels_file_path, "rb") as f:
            self.labels: list = pickle.load(f)

        i = 0
        while i < len(self.labels):
            if not (self.label_key in self.labels[i]):
                self.labels.pop(i)
                i -= 1
            i += 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        info = self.labels[idx]
        img_path = f"{self.images_path}/{info["file_path"]}"

        img = self.transform(torch.tensor(np.array(Image.open(img_path))))

        return (img, info[self.label_key])

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.transform_fn is None):
            x = self.transform_fn(x)
        return x

    def getLoader(self, batch_size=8, shuffle=True, num_workers=4):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
