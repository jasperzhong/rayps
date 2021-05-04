import unittest

import numpy as np
import ray
from torchvision.models import resnet50

import rayps
import rayps.ps


class TestPushPull(unittest.TestCase):
    def setUp(self):
        ray.init()

    def test_pushpull(self):
        model = resnet50()
        ps = rayps.ps.BytePS.remote(model, 1)

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
            print(g2)
            self.assertTrue(np.allclose(g1, g2))


if __name__ == "__main__":
    unittest.main()
