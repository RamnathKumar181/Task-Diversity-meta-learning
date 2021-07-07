import torch


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        self.results[run] = result

    def reset(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def print_statistics(self, run=None):
        result = torch.tensor(self.results)
        best_result = torch.tensor(result)

        r = best_result[:, 0]
        print(f'{self.info}: {r.mean():.4f} ± {r.std():.4f}')
