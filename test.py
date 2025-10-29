from dotenv import load_dotenv
load_dotenv()  # noqa

from BoardDetection import BoardDetection
from Dataset.DataSetLoaders.ChessDataset import ChessDataset
import ChessLens
from matplotlib import pyplot as plt

# dataset = ChessDataset(
#     config={
#         "img_size": (640, 640),
#     }
# )
# img, y = dataset[0]
img = ChessLens.ChessLensImage(
    "/mnt/D/University/Thesis_Dataset/Games/Training_Session/game_1549.jpg")
# BoardDetection.board_extractor.set_img(img)
# quad, conf = BoardDetection.board_extractor.extract_board()
quad, conf = img.detect_board()
print(quad, conf)
warpped_img, _ = ChessLens.BoardDetection.board_extractor.warp(quad.numpy())
plt.imsave("out.jpg", warpped_img)
