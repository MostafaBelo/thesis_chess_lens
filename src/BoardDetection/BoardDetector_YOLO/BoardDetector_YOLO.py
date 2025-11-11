from ultralytics import YOLO
import torch
import onnx
import onnxruntime as ort

from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt

import os

model_path = os.path.join(os.environ['WEIGHTS'], 'bd_yolo.pt')
model_onnx_path = os.path.join(os.environ['WEIGHTS'], 'bd_yolo.onnx')
default_conf = 0.7

if "device" in os.environ:
    device = os.environ["device"]
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"


class YOLO_ONNX_Segmentation:
    def __init__(self, model_path=model_onnx_path):
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path):
        self.model = ort.InferenceSession(model_path)

    def __call__(self, img):
        outputs = self.model.run(None, {"images": img})
        return self._postprocess_yolo_segmentation(outputs)

    def _postprocess_yolo_segmentation(
        self,
        outputs: tuple[np.ndarray],
        original_img_shape: tuple = (640, 640),
        conf_threshold: float = 0.75,
        iou_threshold: float = 0.45,
    ) -> list:
        """
        Processes the raw ONNX outputs of a YOLO segmentation model to generate masks.

        Args:
            output_detection: The detection output (1, 37, 8400).
            output_proto: The prototype mask output (1, 32, 160, 160).
            original_img_shape: (H, W) of the original image before model input resizing.
            conf_threshold: Minimum confidence score to keep a detection.
            iou_threshold: NMS IoU threshold.

        Returns:
            A list of dictionaries, each containing 'box' (xyxy) and 'mask' (binary mask).
        """
        output_detection, output_proto = outputs
        H, W = original_img_shape

        # --- 1. Filter and Prepare Detections (output_detection) ---

        # Transpose from (1, 37, 8400) to (8400, 37) and remove batch dimension
        predictions = np.squeeze(output_detection).T

        # 37 columns: [bbox(4), score(1), mask_coeffs(32)]

        # Since you have 1 class, the class score is implicitly included in the
        # score column (usually column 4, or index 4) in the simplified ONNX export.
        # The confidence score is typically the result of (Objectness * Class_Score).

        scores = predictions[:, 4]  # Objectness/Confidence Score

        # Filter by confidence threshold
        # valid_predictions = predictions
        # valid_scores = scores
        valid_predictions = predictions[scores > conf_threshold]
        valid_scores = scores[scores > conf_threshold]

        if len(valid_predictions) == 0:
            return []

        # Extract Bounding Boxes (xywh)
        boxes_xywh = valid_predictions[:, :4]

        # Convert boxes from xywh to xyxy (for NMS)
        boxes_xyxy = np.copy(boxes_xywh)
        boxes_xyxy[:, 0] -= boxes_xyxy[:, 2] / 2  # x1 = x - w/2
        boxes_xyxy[:, 1] -= boxes_xyxy[:, 3] / 2  # y1 = y - h/2
        boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes_xywh[:, 2]  # x2 = x1 + w
        boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes_xywh[:, 3]  # y2 = y1 + h

        # Rescale boxes to original image size (assuming 640x640 input to ONNX model)
        model_size = output_proto.shape[-1] * 4  # e.g., 160*4 = 640
        ratio_w, ratio_h = W / model_size, H / model_size
        boxes_xyxy[:, 0] *= ratio_w
        boxes_xyxy[:, 1] *= ratio_h
        boxes_xyxy[:, 2] *= ratio_w
        boxes_xyxy[:, 3] *= ratio_h

        # --- 2. Apply Non-Maximum Suppression (NMS) ---

        # The `cv2.dnn.NMSBoxes` function expects scores as float32
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            valid_scores.tolist(),
            conf_threshold,
            iou_threshold
        )

        if len(indices) == 0:
            return []

        # Filtered results
        indices = indices.flatten()
        final_boxes = boxes_xyxy[indices]
        # Mask coefficients (32 values)
        final_coeffs = valid_predictions[indices, 5:37]
        final_scores = valid_scores[indices]

        # --- 3. Generate Raw Masks (Matrix Multiplication) ---

        # Reshape prototypes from (1, 32, 160, 160) to (32, 160*160)
        proto = np.squeeze(output_proto)
        proto = proto.reshape(32, -1)  # Shape (32, 25600)

        # Matrix multiplication: (N_detections, 32) @ (32, 160*160) -> (N_detections, 160*160)
        raw_masks = final_coeffs @ proto

        # Apply Sigmoid
        raw_masks = 1 / (1 + np.exp(-raw_masks))

        # Reshape back to (N_detections, 160, 160)
        masks_160x160 = raw_masks.reshape(-1, *output_proto.shape[-2:])

        # --- 4. Upsample and Binarize ---

        final_results = []

        for i in range(len(final_boxes)):
            box_xyxy = final_boxes[i].astype(int)
            mask_160 = masks_160x160[i]
            score = final_scores[i].astype(float).item()

            # Upsample mask to original image size
            # Interpolation needs to be linear for probability maps
            mask_full_res = cv2.resize(
                mask_160, (W, H), interpolation=cv2.INTER_LINEAR)

            # Binarize mask
            binary_mask = (mask_full_res > 0.5).astype(np.uint8)

            # --- 5. Crop Mask (Optional but standard for instance segmentation) ---

            # Use the bounding box to crop the mask for the final output
            x1, y1, x2, y2 = box_xyxy

            # Create an empty mask for cropping
            cropped_mask = np.zeros((H, W), dtype=np.uint8)

            # The relevant part of the mask is within the detected box coordinates (xyxy)
            # Note: We must clip the box coordinates to image boundaries (0 to W/H-1)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)

            cropped_mask[y1:y2, x1:x2] = binary_mask[y1:y2, x1:x2]

            final_results.append({
                'box': box_xyxy,  # Bounding box [x1, y1, x2, y2]
                'mask': torch.tensor(cropped_mask),  # Binary mask (0 or 255)
                'score': score,
            })

        return final_results


class BoardDetector_YOLO:
    def __init__(self, model_path=model_path, conf=default_conf):
        self.model = None
        self.clear()

        self.load_model(model_path)

    def clear(self):
        self.img = None
        self.data = None

    def check_model(self):
        if (self.model is None):
            self.load_model()

    def check_img(self):
        if (self.img is None):
            raise ValueError(
                "Image not set. Please set an image using set_img() method.")

    def check_data(self):
        if (self.data is None):
            self.process()

    def load_model(self, model_path=model_path):
        # self.model = YOLO(model_path).to(device)

        self.model = YOLO_ONNX_Segmentation(model_onnx_path)

    def set_img(self, img):
        self.img = img
        img = self.preprocess()

    def preprocess(self):
        self.check_img()
        # img = self.img

        # img = Image.fromarray(img.numpy()).resize((640, 640))
        # img = np.array(img).astype(np.float32) / 255.0
        # img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)

        # self.img = img

    def process(self):
        self.check_model()
        self.check_img()

        # r = self.model(self.img.unsqueeze(0), conf=0.7, verbose=False)
        # mask = r[0].masks.data.squeeze().cpu()
        # if (len(mask.shape) >= 3):
        #     mask = mask[0]
        # conf = r[0].boxes[0].conf.item()

        r = self.model(self.img.unsqueeze(0).cpu().numpy())
        mask = r[0]["mask"]
        conf = r[0]["score"]

        self.data = (mask, conf)

    def postprocess(self):
        self.check_data()
        mask = self.data[0]

        # apply opening to remove small masks
        open_size = 10  # 6
        dilate_size = 5
        morph_open_mask = np.ones((open_size, open_size), np.uint8)
        morph_dilate_mask = np.ones((dilate_size, dilate_size), np.uint8)
        # print(mask.shape, mask.dtype, type(mask), mask.max(), mask.min())
        eroded = cv2.erode(mask.numpy(), morph_open_mask, iterations=1)
        dilated = cv2.dilate(eroded, morph_open_mask, iterations=1)
        dilated = cv2.dilate(dilated, morph_dilate_mask, iterations=1)

        self.data = (dilated, self.data[1])

    def predict(self):
        self.process()
        self.postprocess()

        return self.data

    def plot_mask(self):
        self.check_img()
        self.check_data()

        img = self.img
        mask = self.data[0]

        plt.imshow(img.squeeze().permute(1, 2, 0))
        plt.imshow(mask, alpha=mask*.5, cmap='Reds')
