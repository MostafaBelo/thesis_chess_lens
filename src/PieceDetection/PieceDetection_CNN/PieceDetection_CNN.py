import math

import torch
from torch import nn
from torchvision import models
import numpy as np
import cv2

import os

from PieceDetection.PieceCropper_3D import PieceCropper

device = "cuda" if torch.cuda.is_available() else "cpu"


class PieceDetectorCNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.latent_dim = 512

        self.resnet = models.resnet18(pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        # Remove the last FC layer and avgpool
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        for param in self.resnet[-2:].parameters():
            param.requires_grad = True

        dropporb = .2

        # self.conv1 = nn.Sequential(
        #     # 32, 32
        #     nn.Conv2d(3, 16, 3, padding=1),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm2d(16),
        #     nn.MaxPool2d(4),
        #     nn.Dropout(dropporb),

        #     # 8, 8
        #     nn.Conv2d(16, 64, 3, padding=1),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm2d(64),
        #     nn.MaxPool2d(4),
        #     nn.Dropout(dropporb),

        #     # 2,2
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.LeakyReLU(),
        #     nn.BatchNorm2d(128),
        #     nn.MaxPool2d(2),
        #     nn.Dropout(dropporb),

        #     # 1, 1
        # )

        self.conv1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, self.latent_dim),
            nn.LeakyReLU(),

            # nn.AdaptiveAvgPool2d((1,1))
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1),
            nn.LeakyReLU(),
            # nn.BatchNorm2d(self.latent_dim),
            # nn.AdaptiveAvgPool2d((8,8)),
            nn.Dropout(dropporb),

            nn.Conv2d(self.latent_dim, self.latent_dim, 3, padding=1),
            nn.LeakyReLU(),
            # nn.BatchNorm2d(self.latent_dim),
            # nn.AdaptiveAvgPool2d((8,8)),
            nn.Dropout(dropporb),

            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.classifier = nn.Sequential(
            # nn.Linear(128, 64),
            # nn.LeakyReLU(),
            # nn.Dropout(dropporb),
        )

        self.occupancy_classifier_heads = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropporb),
            nn.Linear(128, 1)
        )
        self.piece_color_classifier_heads = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropporb),
            nn.Linear(128, 1)
        )
        self.piece_type_classifier_heads = nn.Sequential(
            nn.Linear(self.latent_dim, 128),
            nn.LeakyReLU(),
            nn.Dropout(dropporb),
            nn.Linear(128, 6)
        )

        self.pos_emb = nn.Parameter(torch.zeros(self.latent_dim, 8, 8))
        self.sin_pos_emb = None  # [C, 8, 8]

    def sinusoidal_embeddings_2d(self, height=10, width=10):
        """
        Create 2D sinusoidal positional embeddings for an (H, W) grid.

        height, width: grid size
        d_model: total embedding dimension (should be even)

        Returns:
            pos_emb: [H, W, d_model] tensor
        """
        d_model = self.latent_dim
        if self.sin_pos_emb is not None and (tuple(self.sin_pos_emb.shape) == (d_model, height, width)):
            return self.sin_pos_emb

        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for 2D sin/cos.")

        # Position indices
        y_pos = torch.arange(
            height, dtype=torch.float32).unsqueeze(1)  # [H, 1]
        x_pos = torch.arange(
            width, dtype=torch.float32).unsqueeze(1)   # [W, 1]

        # Each axis gets d_model/2 dimensions, split into sin/cos → d_model/4 each
        d_model_half = d_model // 2
        div_term = torch.exp(torch.arange(
            0, d_model_half, 2).float() * (-math.log(10000.0) / d_model_half))

        # Row (y) embeddings
        pe_y = torch.zeros(height, d_model_half)
        pe_y[:, 0::2] = torch.sin(y_pos * div_term)
        pe_y[:, 1::2] = torch.cos(y_pos * div_term)

        # Col (x) embeddings
        pe_x = torch.zeros(width, d_model_half)
        pe_x[:, 0::2] = torch.sin(x_pos * div_term)
        pe_x[:, 1::2] = torch.cos(x_pos * div_term)

        # Combine row and col into [H, W, d_model]
        pe = torch.zeros(height, width, d_model)
        for i in range(height):
            for j in range(width):
                pe[i, j] = torch.cat([pe_y[i], pe_x[j]], dim=0)

        self.sin_pos_emb = pe.permute(2, 0, 1).to(device)
        return self.sin_pos_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B, 8,8,c,32,32
        # x = x.permute(0,3,1,2,4,5)

        sq_size = (x.shape[4], x.shape[5])
        B = x.shape[0]
        C = x.shape[3]
        if C != 3:
            raise Exception("Invalid Image")

        # board_split = x.reshape(B, C, 10, sq_size[0], 10, sq_size[1]).permute(0,1,2,4,3,5)
        # board_split = x

        # embeds = self.conv1(board_split.permute(0,2,3,1,4,5).reshape(-1,C,sq_size[0],sq_size[1])).squeeze(-1).squeeze(-1).reshape(B, 10,10, 128).permute(0,3,1,2)
        embeds = self.conv1(self.resnet(x.reshape(-1, C, sq_size[0], sq_size[1]))).squeeze(
            -1).squeeze(-1).reshape(B, 8, 8, self.latent_dim).permute(0, 3, 1, 2)

        sin_pos_emb = self.sinusoidal_embeddings_2d(8, 8)

        # positioned_embeds = embeds + self.pos_emb + sin_pos_emb
        # positioned_embeds = embeds + sin_pos_emb
        positioned_embeds = embeds
        # attentioned_embeds = embeds.reshape(B, 10,10, 128).permute(0,3,1,2)
        attentioned_embeds = self.conv2(positioned_embeds)
        # cropped_attentioned_embeds = attentioned_embeds[:,:,1:9,1:9]
        res = self.classifier(attentioned_embeds.permute(
            0, 2, 3, 1).reshape(-1, self.latent_dim))
        res_occupancy = self.occupancy_classifier_heads(res).reshape(B, 8, 8)
        res_piece_color = self.piece_color_classifier_heads(
            res).reshape(B, 8, 8)
        res_piece_type = self.piece_type_classifier_heads(
            res).reshape(B, 8, 8, 6)

        return {
            "occupancy": res_occupancy,
            "piece_color": res_piece_color,
            "piece_type": res_piece_type,
        }


piece_detection_model = PieceDetectorCNNModel()
# piece_detection_model.load_state_dict(torch.load(
#     os.path.join(os.path.dirname(__file__), "best_model.pt"), map_location=device))
piece_detection_model.load_state_dict(torch.load(
    os.path.join(os.environ['WEIGHTS'], "piece_cnn.pt"), map_location=device))
piece_detection_model = piece_detection_model.to(device)


class PieceDetector:
    def __init__(self):
        self.img = None
        self.corners = None

        self.split_img = None

        self.warpped_img = None
        self.M = None

    def set_img(self, img: torch.Tensor, corners: torch.Tensor):
        self.img = img  # 3, H, W
        self.corners = corners  # 4, 2

    def _warp(self, padding=(0, 0)):
        img_size = (256, 256)
        quad = self.corners.numpy().astype(np.float32)
        dst = np.array([
            [0, 0],
            [img_size[0], 0],
            [img_size[0], img_size[1]],
            [0, img_size[1]]
        ], dtype=np.float32)
        dst[:, 0] += padding[0]
        dst[:, 1] += padding[1]

        M = cv2.getPerspectiveTransform(quad, dst)
        src_img = self.img.permute(1, 2, 0).numpy()
        src_img = src_img if src_img.max() <= 1 else (src_img*255).astype(np.uint8)
        warpped = cv2.warpPerspective(
            src_img, M, (img_size[0]+2*padding[0], img_size[1]+2*padding[1]))

        warpped = torch.tensor(
            warpped/255, dtype=torch.float32).permute(2, 0, 1)
        M = torch.tensor(M)

        return warpped, M

    def preprocess(self):
        PieceCropper.piece_cropper.set_img(
            self.img, self.corners)
        self.board_split = PieceCropper.piece_cropper.process_img()

    def predict(self):
        piece_detection_model.eval()
        with torch.inference_mode():
            preds = piece_detection_model(
                self.board_split.unsqueeze(0).to(device))

        occupancy: torch.Tensor = preds["occupancy"]
        piece_color: torch.Tensor = preds["piece_color"]
        piece_type: torch.Tensor = preds["piece_type"]

        occupancy = occupancy.sigmoid()
        piece_color = piece_color.sigmoid()
        piece_type = piece_type.softmax(dim=3)

        self.occupancy = occupancy
        self.piece_color = piece_color
        self.piece_type = piece_type

        res = torch.zeros(1, 8, 8, 13)
        res[:, :, :, 12] = 1-occupancy
        res[:, :, :, :6] = (
            piece_type * piece_color.unsqueeze(-1)) * occupancy.unsqueeze(-1)
        res[:, :, :, 6:12] = (
            piece_type * (1-piece_color).unsqueeze(-1)) * occupancy.unsqueeze(-1)

        res = res.permute(0, 3, 1, 2)

        return res
