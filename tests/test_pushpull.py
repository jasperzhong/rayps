import unittest

import numpy as np
import ray
from torchvision.models import resnet50

import rayps
import rayps.ps


class TestPushPull(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ray.init()

    def test_single_worker_pushpull(self):
        model = resnet50()
        ps = rayps.ps.BytePS.options(max_concurrency=8).remote(model, 1)

        sent_grads = []
        futures = []
        for key, param in enumerate(model.parameters()):
            if param.requires_grad:
                grad = np.random.normal(size=param.size())
                sent_grads.append(grad)
                ps.push.remote(key, grad)
                futures.append(ps.pull.remote(key))

        recv_grads = ray.get(futures)

        for g1, g2 in zip(sent_grads, recv_grads):
            self.assertTrue(np.allclose(g1, g2))

    def test_multi_workers_pushpull(self):
        model = resnet50()
        num_workers = 32
        ps = rayps.ps.BytePS.options(
            max_concurrency=num_workers).remote(model, num_workers)

        sent_grads = []
        futures = []
        for key, param in enumerate(model.parameters()):
            if param.requires_grad:
                grads = [np.random.normal(size=param.size())
                         for _ in range(num_workers)]
                sent_grads.append(np.mean(grads, axis=0))
                [ps.push.remote(key, grad) for grad in grads]
                futures.append(ps.pull.remote(key))

        recv_grads = ray.get(futures)

        for g1, g2 in zip(sent_grads, recv_grads):
            self.assertTrue(np.allclose(g1, g2))

    def test_multi_iters_multi_workers_pushpull(self):
        model = resnet50()
        num_workers = 8
        ps = rayps.ps.BytePS.options(
            max_concurrency=num_workers).remote(model, num_workers)

        for _ in range(10):
            sent_grads = []
            futures = []
            for key, param in enumerate(model.parameters()):
                if param.requires_grad:
                    grads = [np.random.normal(size=param.size())
                             for _ in range(num_workers)]
                    sent_grads.append(np.mean(grads, axis=0))
                    [ps.push.remote(key, grad) for grad in grads]
                    futures.append(ps.pull.remote(key))

            recv_grads = ray.get(futures)

            for g1, g2 in zip(sent_grads, recv_grads):
                self.assertTrue(np.allclose(g1, g2))


if __name__ == "__main__":
    unittest.main()
