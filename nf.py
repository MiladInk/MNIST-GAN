# Normalizing flow, First try
# I am trying to replicate this article https://blog.evjang.com/2018/01/nf1.html
import matplotlib.pyplot as plt
import torch as tc
import torch.distributions as dist
from torch import nn, optim


def plot_samples(x_samples):
    x1_samples, x2_samples = x_samples[:, 0], x_samples[:, 1]
    plt.plot(x1_samples, x2_samples, 'ro')
    plt.show()


def sample(show: bool = False):
    batch_size = 512
    x2_dist = dist.Normal(loc=0., scale=4.)
    x2_samples = x2_dist.sample(tc.Size([512]))
    x1_dist = dist.Normal(loc=.25 * tc.pow(x2_samples, 2), scale=tc.ones(batch_size, dtype=tc.float32))
    x1_samples = x1_dist.sample()
    x_samples = tc.stack([x1_samples, x2_samples], dim=1)
    if show:
        plot_samples(x_samples)
    return x_samples


# base distribution
base_dist = dist.MultivariateNormal(loc=tc.zeros([2], dtype=tc.float32), covariance_matrix=tc.eye(2))


class PReLU(nn.Module):

    def __init__(self, dim):
        super(PReLU, self).__init__()
        self.inner_alpha = nn.Parameter(tc.randn(1, dim), requires_grad=True)
        self.alpha = None

    def forward(self, x):
        self.alpha = tc.abs(self.inner_alpha) + 0.1
        y = tc.where(x >= 0, x, self.alpha * x)
        return y, self.log_abs_det_jacobian(x)

    def backward(self, y):
        self.alpha = tc.abs(self.inner_alpha) + 0.1
        x = tc.where(y >= 0, y, y / self.alpha)
        return x, -self.log_abs_det_jacobian(x)

    def log_abs_det_jacobian(self, x):
        jacobian_diag = tc.where(x >= 0, tc.zeros(1), tc.log(self.alpha))
        return tc.sum(jacobian_diag, dim=1)


class AffineConstantFlow(nn.Module):
    def __init__(self, dim):
        super(AffineConstantFlow, self).__init__()
        self.log_scale = nn.Parameter(tc.randn(1, dim), requires_grad=True)
        self.shift = nn.Parameter(tc.randn(1, dim), requires_grad=True)

    def forward(self, x):
        z = x * tc.exp(self.log_scale) + self.shift
        log_det = tc.sum(self.log_scale, dim=1)
        return z, log_det

    def backward(self, z):
        x = (z - self.shift) * tc.exp(-self.log_scale)
        log_det = tc.sum(-self.log_scale, dim=1)
        return x, log_det


class MatrixFlow(nn.Module):
    def __init__(self, dim):
        super(MatrixFlow, self).__init__()
        self.matrix = nn.Parameter(tc.randn(dim, dim), requires_grad=True)
        self.shift = nn.Parameter(tc.randn(1, dim), requires_grad=True)

    def forward(self, x):
        z = tc.mm(x, self.matrix) + self.shift
        log_det = tc.slogdet(self.matrix).logabsdet * tc.ones(x.shape[0])
        return z, log_det

    def backward(self, z):
        x = tc.mm((z - self.shift), tc.inverse(self.matrix))
        log_det = -tc.slogdet(self.matrix).logabsdet * tc.ones(z.shape[0])
        return x, log_det


class NormalizingFlow(nn.Module):
    def __init__(self, flows):
        super(NormalizingFlow, self).__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x):
        m, _ = x.shape
        log_det = tc.zeros(m)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def backward(self, z):
        m, _ = z.shape
        log_det = tc.zeros(m)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.backward(z)
            log_det += ld
            xs.append(z)
        return xs, log_det


class NormalizingFLowModel(nn.Module):
    def __init__(self, prior, flows):
        super(NormalizingFLowModel, self).__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)

    def forward(self, x):
        zs, log_det = self.flow.forward(x)
        # TODO what is this
        prior_log_prob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_log_prob, log_det

    def backward(self, z):
        xs, log_det = self.flows.backward(z)
        return xs, log_det

    def sample(self, num_samples):
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.backward(z)
        return xs


if __name__ == '__main__':
    sample(show=True)
    flows = []
    for i in range(20):
        flows.append(MatrixFlow(dim=2))
        flows.append(PReLU(dim=2))

    model = NormalizingFLowModel(base_dist, flows=flows)
    optimizer = optim.Adam(model.parameters(), lr=3e-3)

    model.train()

    for step in range(100000):
        x_samples = sample(show=False)
        model.zero_grad()
        zs, prior_log_prob, log_det = model(x_samples)
        log_prob = prior_log_prob + log_det
        loss = -tc.sum(log_prob)

        # tweak the model parameters
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(step, loss.item())
            plot_samples(model.sample(1000)[-1].detach().numpy())
