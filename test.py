from BoardDetection import BoardDetection
from Dataset.DataSetLoaders.ChessDataset import ChessDataset

dataset = ChessDataset(
    config={
        "img_size": (640, 640),
    }
)
img, y = dataset[0]
BoardDetection.board_extractor.set_img(img)
quad, conf = BoardDetection.board_extractor.extract_board()
print(quad, conf)
