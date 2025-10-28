import torch
import numpy as np

import chess
import chess.pgn
import chess.svg

import os

if "svg" in os.environ and os.environ["svg"] == "TRUE":
    import cairosvg


class ChessTensorUtils():

    @staticmethod
    def FENtoTensor(fen: str) -> torch.Tensor:
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

    @staticmethod
    def onehot_to_int(onehot: torch.Tensor) -> torch.Tensor:
        return onehot.argmax(dim=1)

    @staticmethod
    def int_to_onehot(ints: torch.Tensor) -> torch.Tensor:  # TODO:
        onehot = torch.zeros(13, ints.shape[1], ints.shape[2])
        onehot.scatter_(0, ints.unsqueeze(2), 1)
        return onehot.unsqueeze(0)

    @staticmethod
    def tensorToFEN_MAX(board: torch.Tensor) -> str:
        channels = ['P', 'N', 'B', 'R', 'Q', 'K',
                    'p', 'n', 'b', 'r', 'q', 'k', '1']
        if (len(board.shape) >= 4):
            board = ChessTensorUtils.onehot_to_int(board)

        if (board.shape[0] != 1):
            res = []
            for i in range(board.shape[0]):
                res.append(ChessTensorUtils.tensorToFEN_MAX(board[i]))
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

    @staticmethod
    def randOneHot(seed: int = -1) -> torch.Tensor:
        if (seed != -1):
            np.random.seed(seed)

        board = np.zeros((13, 8, 8), dtype=np.float32)
        for i in range(8):
            for j in range(8):
                board[np.random.randint(0, 13), i, j] = 1
        return torch.tensor(board).unsqueeze(0)

    @staticmethod
    def randInts(seed: int = -1) -> torch.Tensor:
        return ChessTensorUtils.onehot_to_int(ChessTensorUtils.randOneHot(seed))

    @staticmethod
    def randFEN(seed: int = -1) -> str:
        return ChessTensorUtils.tensorToFEN_MAX(ChessTensorUtils.randOneHot(seed))

    @staticmethod
    def fens_to_pgn(fen_list):
        if not fen_list:
            return ""
        result = [fen_list[0]]
        for item in fen_list[1:]:
            if item != result[-1]:
                result.append(item)

        # Create a new game
        game = chess.pgn.Game()

        # Start from the first FEN
        board = chess.Board(result[0] + " w KQkq - 0 1")
        node = game

        for next_fen in result[1:]:
            # Compute the move that leads from current to next position
            move = None
            for candidate in board.legal_moves:
                board.push(candidate)
                if board.fen().split(' ', 1)[0] == next_fen:
                    move = candidate
                    board.pop()
                    break
                board.pop()

            if move is None:
                # raise ValueError(
                print(
                    f"Could not find a valid move to reach next FEN! - Current FEN: {board.fen()} - Next FEN: {next_fen}")
                break

            # Push the move to the game
            node = node.add_variation(move)
            board.push(move)

        return game


def fen_to_png(fen: str, folder_path: str, file_name: str):
    board = chess.Board(f"{fen} w KQkq - 1 1")
    boardsvg = chess.svg.board(coordinates=True, board=board, size=350, colors={
                               "square light": "#E6D0A7", "square dark": "#A67D5B"})
    svg_file_path = f"{folder_path}/positions.svg"
    f = open(svg_file_path, "w")
    f.write(boardsvg)
    f.close()
    png_file_path = f"{folder_path}/{file_name}"
    if "svg" in os.environ and os.environ["svg"] == "TRUE":
        cairosvg.svg2png(url=svg_file_path, write_to=png_file_path, scale=7)
