import time
from threading import Lock

import numpy as np
import ray


@ray.remote
class BytePS:
    def __init__(self, model, n_workers):
        self.n_workers = n_workers
        self.n_recv = {}
        self.ready_for_pull = {}
        self.buffer = {}
        self.locks = {}

        for key, param in enumerate(model.parameters()):
            if param.requires_grad:
                self.n_recv[key] = 0
                self.ready_for_pull[key] = False
                self.buffer[key] = np.zeros(param.size())
                self.locks[key] = Lock()

    def push(self, key, grad):
        self.locks[key].acquire()
        if self.n_recv[key] == 0:
            self.buffer[key].fill(0)
        self.buffer[key] += grad
        self.n_recv[key] += 1
        if self.n_recv[key] == self.n_workers:
            self.buffer[key] /= self.n_workers
            self.ready_for_pull[key] = True
        self.locks[key].release()

    def pull(self, key):
        while not self.ready_for_pull[key]:
            time.sleep(0.001)
        self.ready_for_pull[key] = False
        self.n_recv[key] = 0
        return self.buffer[key]
