import ray
import numpy as np


@ray.remote
class BytePS:
    def __init__(self, model, n_workers):
        self.n_workers = n_workers
        self.n_recv = {}
        self.ready_for_pull = {}
        self.buffer = {}

        for key, param in enumerate(model.parameters()):
            if param.requires_grad:
                self.n_recv[key] = 0
                self.ready_for_pull[key] = False
                self.buffer[key] = np.zeros(param.size())

    def push(self, key, grad):
        if self.n_recv[key] == 0:
            self.buffer[key].fill(0)
        self.buffer[key] += grad
        self.n_recv[key] += 1
        if self.n_recv[key] == self.n_workers:
            self.buffer[key] /= self.n_workers
            self.ready_for_pull[key] = True

    def pull(self, key):
        if self.ready_for_pull[key]:
            self.ready_for_pull[key] = False
            self.n_recv[key] = 0
            return self.buffer[key]
