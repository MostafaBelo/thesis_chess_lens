from HMM import ChessHMM as context_model


class ChessHMM:
    def __init__(self, bredth=30, bind_period=20):
        self.model = context_model.ChessHMM(bredth)
        self.bredth = bredth
        self.bind_period = bind_period

    def bind(self):
        self.model.bind(self.model.top_t())

    def set_probs(self, timestep, piece_matrix):
        self.model.set_probs(timestep, piece_matrix)

        if (((self.model.top_t() % self.bind_period) == 0) and (self.model.top_t() >= self.bind_period) and (self.model.top_t()-self.bind_period+1 > self.model.top_bind_t())):
            self.model.bind(self.model.top_t()-self.bind_period+1)

    def print(self, timestep: int) -> str:
        return self.model.print(timestep)

    def get_history(self, include_non_bound: bool = False):
        return self.model.get_history(include_non_bound)

    def get_pgn(self) -> str:
        self.model.get_pgn()
