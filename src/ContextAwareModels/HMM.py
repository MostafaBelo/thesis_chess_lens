from HMM import ChessHMM as context_model

import time


class ChessHMM:
    def __init__(self, bredth=30, delay=120, bind_period=20):
        self.model = context_model.ChessHMM(bredth)
        self.bredth = bredth
        self.bind_period = bind_period
        self.delay = delay

        self.timestamp_map = {}

    def bind(self):
        self.model.bind(self.model.top_t())

    def set_probs(self, timestep, piece_matrix, actual_frame_index):
        self.timestamp_map[timestep] = actual_frame_index

        self.model.set_probs(timestep, piece_matrix)

        if (((self.model.top_t() % self.bind_period) == 0) and (self.model.top_t() >= self.bind_period+self.delay) and (self.model.top_t()-self.delay-self.bind_period+1 > self.model.top_bind_t())):
            self.model.bind(self.model.top_t()-self.delay+1)

    def check_bind(self, frame_index):
        bind_at = -1

        for i in range(self.model.top_bind_t()+1, self.model.top_t()):
            if (frame_index - self.timestamp_map[i]) >= self.delay:
                bind_at = i

        if bind_at == -1:
            return

        try:
            self.model.bind(bind_at)
            return True
        except:
            return False

    def print(self, timestep: int) -> str:
        return self.model.print(timestep)

    def get_history(self, include_non_bound: bool = False):
        return self.model.get_history(include_non_bound)

    def get_pgn(self) -> str:
        self.model.get_pgn()
