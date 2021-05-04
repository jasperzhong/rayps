import ray


@ray.remote
class Worker:
    def __init__(self, model, loss_fn, optim):
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim

    def compute_gradient(self, data, label):
        output = self.model(data)
        loss = self.loss_fn(output, label)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
