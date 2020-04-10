import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim
from torch.utils import data as t_data
from torchvision import datasets, transforms
from torchvision.utils import save_image

batch_size = 4
data_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_training_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=data_transforms)
train_loader = t_data.DataLoader(mnist_training_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', download=True, train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        # encoder
        self.encoder_layer1 = nn.Linear(784, 400)
        self.encoder_mu = nn.Linear(400, 20)
        self.encoder_logvar = nn.Linear(400, 20)
        # decoder
        self.decoder_layer1 = nn.Linear(20, 400)
        self.decoder_layer2 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.encoder_layer1(x))
        return self.encoder_mu(h1), self.encoder_logvar(h1)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x):
        h3 = F.relu(self.decoder_layer1(x))
        return torch.sigmoid(self.decoder_layer2(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


net = VAE()
optimizer = optim.Adam(net.parameters(), lr=1e-3)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(reconstructed_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(reconstructed_x, x.view(-1, 784), reduction='sum')

    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


epochs = 100
log_interval = 100


def train(epoch_cnt):
    net.train()  # TODO what is it
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch, mu, logvar = net(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_cnt, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset)))


def test(epoch_cnt):
    net.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            recon_batch, mu, logvar = net(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch_cnt) + '.png', nrow=n)

    test_loss /= len(train_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(epochs):
    train(epoch)
    test(epoch)
    with torch.no_grad():
        sample = torch.randn(64, 20)
        sample = net.decode(sample)
        save_image(sample.view(64, 1, 28, 28),
                   'results/sample_' + str(epoch) + '.png')
