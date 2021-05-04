import time
from threading import Lock

import numpy as np
import ray


@ray.remote
class BytePS:
    def __init__(self, model, sharded_keys, n_workers):
        self.n_workers = n_workers
        self.n_recv = {}
        self.ready_for_pull = {}
        self.buffer = {}
        self.store = {}
        self.locks = {}

        for key, param in enumerate(model.parameters()):
            if param.requires_grad:
                if key in sharded_keys:
                    self.n_recv[key] = 0
                    self.ready_for_pull[key] = False
                    self.store[key] = np.zeros(param.size())
                    self.locks[key] = Lock()

    def push(self, key, grad):
        self.locks[key].acquire()
        if self.n_recv[key] == 0:
            # copy is needed here.
            # https://github.com/ray-project/ray/issues/369
            self.buffer[key] = grad.copy()
        else:
            self.buffer[key] += grad
        self.n_recv[key] += 1
        if self.n_recv[key] == self.n_workers:
            self.buffer[key] /= self.n_workers
            self.store[key][:] = self.buffer[key]
            self.ready_for_pull[key] = True
        self.locks[key].release()

    def pull(self, key):
        while not self.ready_for_pull[key]:
            time.sleep(0.001)
        self.locks[key].acquire()
        self.ready_for_pull[key] = False
        self.n_recv[key] = 0
        self.locks[key].release()
        return self.store[key]
