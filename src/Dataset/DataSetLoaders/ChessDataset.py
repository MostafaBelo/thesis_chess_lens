import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np

import pickle
from PIL import Image

from dotenv import load_dotenv
import os
import sys

import re

import random

from Utils.ChessUtils import ChessTensorUtils

# Load .env file
load_dotenv()

env_root_key = "CHESSDATASET_ROOT"
env_root_game_key = "CHESSGAMEDATASET_ROOT"


def convert_labels_txt_to_pkl(txt_data: str):
    txt_data = txt_data.strip().split("\n")
    pkl_data = []
    for line in txt_data:
        index = line.find(".#")
        id = line[:index]
        rest_of_line = line[index+2:]

        data = re.findall(r"\((.*?)\)", rest_of_line)
        if len(data) != 5:
            raise Exception(f"Invalid labels.txt record at: {line}")
        image_path, fen, orientation, corners, orig_img_size = data

        corners = list(map(lambda x: float(x), corners.split(",")))
        corners = torch.tensor(corners, dtype=torch.float32).reshape(4, 2)

        orig_img_size = list(map(lambda x: float(x), orig_img_size.split(",")))
        orig_img_size = torch.tensor(orig_img_size, dtype=torch.float32)

        board_tensor = ChessTensorUtils.onehot_to_int(
            ChessTensorUtils.FENtoTensor(fen)).squeeze().to(torch.uint8)

        pkl_data.append({
            "id": int(id),
            "image_path": image_path,
            "original_img_size": orig_img_size,
            "fen": fen,
            "orientation": orientation,
            "corners": corners,
            "board_tensor": board_tensor,
        })
    return pkl_data


def convert__pgn_labels_txt_to_pkl(txt_data: str):
    imgs, pgn = txt_data.strip().split("\n\n")
    img_objs = []

    for line in imgs.split("\n"):
        index = line.find(".#")
        id = line[:index]
        rest_of_line = line[index+2:]

        data = re.findall(r"\((.*?)\)", rest_of_line)
        if len(data) != 2:
            raise Exception(f"Invalid labels.txt record at: {line}")
        image_path, frame_validity = data
        img_objs.append({
            "id": int(id),
            "image_path": image_path,
            "frame_validity": frame_validity
        })

    pkl_data = {
        "imgs": img_objs,
        "pgn": pgn
    }

    return pkl_data


def _resize_corners(original_size, new_size):
    def main(label):
        corners: torch.Tensor = label["corners"].clone()
        corners[:, 0] *= new_size[1]/original_size[1]
        corners[:, 1] *= new_size[0]/original_size[0]

        # label["corners"] = corners
        new_label = {
            **label
        }
        new_label["corners"] = corners

        return new_label

    return main


def _rotate_board_tensor():
    def main(label):
        orientation = label["orientation"]
        board_tensor = label["board_tensor"].clone()

        if orientation == 'r':
            board_tensor = torch.rot90(board_tensor, k=1, dims=(0, 1))
        elif orientation == 'l':
            board_tensor = torch.rot90(board_tensor, k=-1, dims=(0, 1))
        elif orientation == "t":
            board_tensor = torch.rot90(board_tensor, k=2, dims=(0, 1))
        elif orientation == "b":
            pass

        new_label = {
            **label
        }
        new_label["board_tensor"] = board_tensor

        return new_label
    return main


class ChessDataset(Dataset):
    def __init__(self, root_dirs: None | str | list[str] = None, img_transforms=None, target_transforms=None, img_label_transforms=None, force_build_pkl=False, config={}):
        if "custom_dataset" in config:
            data = config["custom_dataset"]

            if "root_dirs" not in data:
                raise Exception("Invalid Custom Dataset")
            self.root_dirs = data["root_dirs"]

            if "transforms" not in data:
                raise Exception("Invalid Custom Dataset")
            self.transforms = data["transforms"]

            if "target_transforms" not in data:
                raise Exception("Invalid Custom Dataset")
            self.target_transforms = data["target_transforms"]

            if "img_label_transforms" not in data:
                raise Exception("Invalid Custom Dataset")
            self.img_label_transforms = data["img_label_transforms"]

            if "labels" not in data:
                raise Exception("Invalid Custom Dataset")
            self.labels = data["labels"]
            return

        if root_dirs is None:
            if not os.environ.get(env_root_key):
                raise Exception("CHESSDATASET_ROOT not found in .env")
            root_dirs = os.environ.get(env_root_key).split(";")
        elif type(root_dirs) == str:
            root_dirs = [root_dirs]
        elif type(root_dirs) != list:
            raise Exception("Invalid root_dirs")

        self.root_dirs = root_dirs
        for root_dir in self.root_dirs:
            if not os.path.isdir(root_dir):
                raise Exception(f"Invalid root_dir: {root_dir}")

        self.transforms = transforms.Compose([
            transforms.Resize(
                (480, 640) if ("img_size" not in config) else config["img_size"]),
            *([transforms.Grayscale()]
              if ("gray" in config and config["gray"]) else []),
            transforms.ToTensor(),
            *([lambda img: (img*255).to(torch.uint8)]
              if ("is_int" in config and config["is_int"]) else []),
            *([img_transforms] if img_transforms is not None else [])
        ])
        self.target_transforms = transforms.Compose([
            *([_resize_corners((480, 640), config["img_size"])]
              if ("img_size" in config) else []),
            _rotate_board_tensor(),
            *([target_transforms] if target_transforms is not None else [])
        ])
        self.img_label_transforms = transforms.Compose([
            *([img_label_transforms] if img_label_transforms is not None else [])
        ])

        self.labels = []

        for root_dir in self.root_dirs:
            if (not os.path.exists(os.path.join(root_dir, "labels.pkl"))) or (force_build_pkl):
                if not os.path.exists(os.path.join(root_dir, "labels.txt")):
                    raise Exception(
                        f"labels.txt not found for dir: {root_dir}")
                with open(os.path.join(root_dir, "labels.txt"), "r") as f:
                    txt_data = f.read()
                pkl_data = convert_labels_txt_to_pkl(txt_data)
                with open(os.path.join(root_dir, "labels.pkl"), "wb") as f:
                    pickle.dump(pkl_data, f)

            with open(os.path.join(root_dir, "labels.pkl"), "rb") as f:
                labels = pickle.load(f)

            for i in range(len(labels)):
                try:
                    labels[i]["id"] = f"{root_dir} - {labels[i]['id']}"
                    labels[i]["image_path"] = os.path.join(
                        root_dir, labels[i]["image_path"])
                except Exception as e:
                    print(labels[i])
                    raise e
            self.labels += labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]

        img_path = label["image_path"]
        fen = label["fen"]
        orientation = label["orientation"]
        corners = label["corners"]

        img = Image.open(img_path).convert("RGB")

        img = self.transforms(img)
        label = self.target_transforms(label)
        img, label = self.img_label_transforms((img, label))

        return img, label

    @staticmethod
    def getLoader(dataset: 'ChessDataset', batch_size=4, num_worders=4):
        if num_worders == 0:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,     # Set True if using GPU
                num_workers=0,      # Adjust depending on your CPU cores
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,     # Set True if using GPU
                num_workers=4,      # Adjust depending on your CPU cores
                prefetch_factor=2,
                persistent_workers=True,
            )

    @staticmethod
    def train_valid_test_split(dataset: 'ChessDataset', sizes=(.8, .1, .1), random_state=42):
        random.seed(random_state)
        tmp_labels = dataset.labels
        random.shuffle(tmp_labels)
        train_size = sizes[0]
        valid_size = sizes[1]

        train_idx = int(len(tmp_labels) * train_size)
        train_labels = tmp_labels[:train_idx]

        valid_idx = int(len(tmp_labels) * valid_size)
        valid_labels = tmp_labels[train_idx:train_idx+valid_idx]

        test_labels = tmp_labels[train_idx+valid_idx:]

        train_dataset = ChessDataset(
            config={
                "custom_dataset": {
                    "root_dirs": dataset.root_dirs,
                    "transforms": dataset.transforms,
                    "target_transforms": dataset.target_transforms,
                    "img_label_transforms": dataset.img_label_transforms,
                    "labels": train_labels
                }
            }
        )

        valid_dataset = ChessDataset(
            config={
                "custom_dataset": {
                    "root_dirs": dataset.root_dirs,
                    "transforms": dataset.transforms,
                    "target_transforms": dataset.target_transforms,
                    "img_label_transforms": dataset.img_label_transforms,
                    "labels": valid_labels
                }
            }
        )

        test_dataset = ChessDataset(
            config={
                "custom_dataset": {
                    "root_dirs": dataset.root_dirs,
                    "transforms": dataset.transforms,
                    "target_transforms": dataset.target_transforms,
                    "img_label_transforms": dataset.img_label_transforms,
                    "labels": test_labels
                }
            }
        )

        return train_dataset, valid_dataset, test_dataset


class GameDataset(Dataset):
    def __init__(self, root_dirs: None | str | list[str] = None, img_transforms=None, force_build_pkl=False, config={}):
        if "custom_dataset" in config:
            data = config["custom_dataset"]

            if "root_dirs" not in data:
                raise Exception("Invalid Custom Dataset")
            self.root_dirs = data["root_dirs"]

            if "transforms" not in data:
                raise Exception("Invalid Custom Dataset")
            self.transforms = data["transforms"]

            if "target_transforms" not in data:
                raise Exception("Invalid Custom Dataset")
            self.target_transforms = data["target_transforms"]

            if "labels" not in data:
                raise Exception("Invalid Custom Dataset")
            self.labels = data["labels"]
            return

        if root_dirs is None:
            if not os.environ.get(env_root_key):
                raise Exception("CHESSDATASET_ROOT not found in .env")
            root_dirs = os.environ.get(env_root_game_key).split(";")
        elif type(root_dirs) == str:
            root_dirs = [root_dirs]
        elif type(root_dirs) != list:
            raise Exception("Invalid root_dirs")

        self.root_dirs = root_dirs
        for root_dir in self.root_dirs:
            if not os.path.isdir(root_dir):
                raise Exception(f"Invalid root_dir: {root_dir}")

        self.transforms = transforms.Compose([
            transforms.Resize(
                (480, 640) if ("img_size" not in config) else config["img_size"]),
            *([transforms.Grayscale()]
              if ("gray" in config and config["gray"]) else []),
            transforms.ToTensor(),
            *([lambda img: (img*255).to(torch.uint8)]
              if ("is_int" in config and config["is_int"]) else []),
            *([img_transforms] if img_transforms is not None else [])
        ])

        self.include_only = ["valid", "occlusion"]
        if "include_only" in config:
            self.include_only = config["include_only"].split(";")
        self.labels = []

        for root_dir in self.root_dirs:
            if (not os.path.exists(os.path.join(root_dir, "labels.pkl"))) or (force_build_pkl):
                if not os.path.exists(os.path.join(root_dir, "labels.txt")):
                    raise Exception(
                        f"labels.txt not found for dir: {root_dir}")
                with open(os.path.join(root_dir, "labels.txt"), "r") as f:
                    txt_data = f.read()
                pkl_data = convert__pgn_labels_txt_to_pkl(txt_data)
                with open(os.path.join(root_dir, "labels.pkl"), "wb") as f:
                    pickle.dump(pkl_data, f)

            with open(os.path.join(root_dir, "labels.pkl"), "rb") as f:
                labels = pickle.load(f)

            labels["imgs"] = [
                lbl for lbl in labels["imgs"] if lbl["frame_validity"] in self.include_only
            ]
            for i in range(len(labels["imgs"])):
                try:
                    labels["imgs"][i]["image_path"] = os.path.join(
                        root_dir, labels["imgs"][i]["image_path"])
                except Exception as e:
                    print(labels["imgs"][i]["image_path"])
                    raise e
            self.labels = labels  # TODO:

    def __len__(self):
        return len(self.labels["imgs"])

    def __getitem__(self, index):
        label = self.labels["imgs"][index]
        img_path = label["image_path"]

        img = Image.open(img_path).convert("RGB")

        img = self.transforms(img)

        return img, label

    def get_pgn(self):
        return self.labels["pgn"]

    @staticmethod
    def getLoader(dataset: 'ChessDataset', batch_size=4, num_worders=4):
        if num_worders == 0:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,     # Set True if using GPU
                num_workers=0,      # Adjust depending on your CPU cores
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,     # Set True if using GPU
                num_workers=4,      # Adjust depending on your CPU cores
                prefetch_factor=2,
                persistent_workers=True,
            )

    @staticmethod
    def train_valid_test_split(dataset: 'ChessDataset', sizes=(.8, .1, .1), random_state=42):
        random.seed(random_state)
        tmp_labels = dataset.labels
        random.shuffle(tmp_labels)
        train_size = sizes[0]
        valid_size = sizes[1]

        train_idx = int(len(tmp_labels) * train_size)
        train_labels = tmp_labels[:train_idx]

        valid_idx = int(len(tmp_labels) * valid_size)
        valid_labels = tmp_labels[train_idx:train_idx+valid_idx]

        test_labels = tmp_labels[train_idx+valid_idx:]

        train_dataset = ChessDataset(
            config={
                "custom_dataset": {
                    "root_dirs": dataset.root_dirs,
                    "transforms": dataset.transforms,
                    "labels": train_labels
                }
            }
        )

        valid_dataset = ChessDataset(
            config={
                "custom_dataset": {
                    "root_dirs": dataset.root_dirs,
                    "transforms": dataset.transforms,
                    "labels": valid_labels
                }
            }
        )

        test_dataset = ChessDataset(
            config={
                "custom_dataset": {
                    "root_dirs": dataset.root_dirs,
                    "transforms": dataset.transforms,
                    "labels": test_labels
                }
            }
        )

        return train_dataset, valid_dataset, test_dataset
