import ChessLens
from PIL import Image
import os
from tqdm import tqdm

input_folder = "/mnt/D/University/Thesis_Dataset/Temp/photos2/photos"
output_valid_folder = "/mnt/D/University/Thesis_Dataset/Temp/photos2/game6/valid"
output_occluded_folder = "/mnt/D/University/Thesis_Dataset/Temp/photos2/game6/occluded"
img = ChessLens.ChessLensImage(None, "cnn_onnx_static")

os.makedirs(output_valid_folder, exist_ok=True)
os.makedirs(output_occluded_folder, exist_ok=True)

for file in tqdm(os.listdir(input_folder)):
    try:
        img.load_image(os.path.join(input_folder, file))
        img.detect_board()
        warped, M = img.warp()
        is_occluded = img.is_occluded()

        if is_occluded:
            Image.fromarray(warped).convert("RGB").save(
                os.path.join(output_occluded_folder, file))
        else:
            Image.fromarray(warped).convert("RGB").save(
                os.path.join(output_valid_folder, file))
    except:
        pass
