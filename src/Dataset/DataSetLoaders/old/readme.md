for the dataset and dataloaders make sure to add the folder paths in the ChessRec_Dataset folder.
The path to images folder and the path to the annotations.pkl file are needed.

Available labels:

- image_id : int
- file_path : str
- game_id : int
- move_id : int
- corners : list(list) [[float]8]8
- fen : str
- board_tensor : tensor [[int]8]8 (ints)
- board_tensor_one_hot : tensor [[[float]13]8]8 (one_hot [0/1], but can be used in models as probabilities)
