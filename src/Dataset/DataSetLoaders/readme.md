for the dataset and dataloaders make sure to add the folder paths in the .env as CHESSDATASET_ROOT

Available labels:

- id : int
- image_path : str
- fen : str
- orientation : "l" | "r" | "t" | "b"
- corners : tensor [[float32]2]4
- board_tensor : tensor [[uint8]8]8 (ints)
