from HMM import ChessHMM
# from Utils import ChessUtils
import numpy as np
# import torch
import time

# g = ChessHMM.ChessGameState()

hmm = ChessHMM.ChessHMM(20)
# piece_matrix = ChessUtils.ChessTensorUtils.FENtoTensor(
# "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1").squeeze().permute(1, 2, 0).numpy()[::-1]
# piece_matrix = -np.log(piece_matrix + (1e-7))

start_time = time.perf_counter()
for i in range(1, 3):
    hmm.set_probs(i, np.random.randn(8, 8, 13))
    # hmm.set_probs(i, piece_matrix)
hmm.bind(2)
end_time = time.perf_counter()

print(hmm.get_history(False))

print(f"Time: {((end_time-start_time) / 120)*1e6} Us")
