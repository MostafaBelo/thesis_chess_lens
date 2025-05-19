import torch
import numpy as np

import chess
import chess.svg
import cairosvg


class ChessTensorUtils():
    def FENtoTensor(self, fen: str) -> torch.Tensor:
        channels = ['P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
        board = np.zeros((13, 8, 8), dtype=np.float32)
        board[12, :, :] = 1
        if (len(fen.split(' ')) > 1):
            fen = fen.split(' ')[0]
        i, j = 0, 0
        for c in fen:
            if c == '/':
                i += 1
                j = 0
            elif c.isdigit():
                j += int(c)
            else:
                try:
                    chan = channels.index(c)
                except ValueError:
                    chan = 12
                board[12, i, j] = 0
                board[chan, i, j] = 1
                j += 1
        return torch.tensor(board).unsqueeze(0)

    def onehot_to_int(self, onehot: torch.Tensor) -> torch.Tensor:
        return onehot.argmax(dim=1)

    def int_to_onehot(self, ints: torch.Tensor) -> torch.Tensor:  # TODO:
        onehot = torch.zeros(13, ints.shape[1], ints.shape[2])
        onehot.scatter_(0, ints.unsqueeze(2), 1)
        return onehot.unsqueeze(0)

    def tensorToFEN_MAX(self, board: torch.Tensor) -> str:
        channels = ['P', 'N', 'B', 'R', 'Q', 'K',
                    'p', 'n', 'b', 'r', 'q', 'k', '1']
        if (len(board.shape) >= 4):
            board = self.onehot_to_int(board)

        if (board.shape[0] != 1):
            res = []
            for i in range(board.shape[0]):
                res.append(self.tensorToFEN_MAX(board[i]))
        else:
            fen = []
            for i in range(8):
                empty = 0
                for j in range(8):
                    piece = channels[board[0, i, j].item()]
                    if piece == '1':
                        empty += 1
                    else:
                        if empty > 0:
                            fen.append(str(empty))
                            empty = 0
                        fen.append(piece)
                if empty > 0:
                    fen.append(str(empty))
                if i < 7:
                    fen.append('/')

            return ''.join(fen)

    def randOneHot(self, seed: int = -1) -> torch.Tensor:
        if (seed != -1):
            np.random.seed(seed)

        board = np.zeros((13, 8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                board[np.random.randint(0, 13), i, j] = 1
        return torch.tensor(board).unsqueeze(0)

    def randInts(self, seed: int = -1) -> torch.Tensor:
        return self.onehot_to_int(self.randOneHot(seed))

    def randFEN(self, seed: int = -1) -> str:
        return self.tensorToFEN_MAX(self.randOneHot(seed))


def fen_to_png(fen: str, folder_path: str, file_name: str):
    board = chess.Board(f"{fen} w KQkq - 1 1")
    boardsvg = chess.svg.board(coordinates=True, board=board, size=350, colors={
                               "square light": "#E6D0A7", "square dark": "#A67D5B"})
    svg_file_path = f"{folder_path}/positions.svg"
    f = open(svg_file_path, "w")
    f.write(boardsvg)
    f.close()
    png_file_path = f"{folder_path}/{file_name}"
    cairosvg.svg2png(url=svg_file_path, write_to=png_file_path, scale=7)
