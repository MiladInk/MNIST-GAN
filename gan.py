import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torch.utils import data as t_data
from torch.utils.data import DataLoader

batch_size = 4
data_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
mnist_training_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=data_transforms)
dataloader_mnist_train = t_data.DataLoader(mnist_training_set, batch_size=batch_size, shuffle=True)


def random_noise() -> torch.Tensor:
    return torch.rand(batch_size, 100)


# A Generator Class which takes random noise and transform it to another distribution
# Simply said, it is a way to make arbitrary distributions

class Generator(nn.Module):
    def __init__(self, inp, out):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(inp, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 1000),
            nn.ReLU(inplace=True),
            nn.Linear(1000, 800),
            nn.ReLU(inplace=True),
            nn.Linear(800, out)
        )

    def forward(self, x):
        x = self.net(x)
        return x


# A Discriminator network which has the task to differentiate between the images generated
# by the generator and the real images

class Discriminator(nn.Module):

    def __init__(self, inp, out):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(inp, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, out),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.net(x)
        return x


def plot_img(array, number=None):
    array = array.detach()
    array = array.reshape(28, 28)

    plt.imshow(array, cmap='binary')
    plt.xticks([])
    plt.yticks([])
    if number:
        plt.xlabel(number, fontsize='x-large')
    plt.show()


gen = Generator(100, 784)
dis = Discriminator(784, 1)
d_steps = 100
g_steps = 100
printing_steps = 20
epochs = 100
optimizer1 = torch.optim.SGD(dis.parameters(), lr=0.001, momentum=0.9)
optimizer2 = torch.optim.SGD(gen.parameters(), lr=0.001, momentum=0.9)


def sample_from_torch_data_loader(data_loader: DataLoader) -> torch.Tensor:
    for data, _ in data_loader:
        return data


for epoch in range(epochs):
    for d_step in range(d_steps):
        dis.zero_grad()

        # here we should train the discriminator on the real data
        real_input = sample_from_torch_data_loader(dataloader_mnist_train)
        real_input = real_input.reshape(batch_size, 784)
        judgement = dis(real_input)  # here is what discriminator thinks of the data
        truth = torch.ones(batch_size, 1)  # here is the truth, all the images are real of course
        loss = nn.BCELoss()(judgement, truth)
        loss.backward()

        # training discriminator by data produced by the generator
        fake_input = gen(random_noise()).detach()
        judgement = dis(fake_input)
        truth = torch.zeros(batch_size, 1)
        loss = nn.BCELoss()(judgement, truth)
        loss.backward()
        optimizer1.step()

    for g_step in range(g_steps):
        gen.zero_grad()

        generated = gen(random_noise())
        judgement = dis(generated)
        goal = torch.ones(batch_size, 1)
        # the goal is to make the discrimination being wrong, so the goal is to get all ones from it
        loss = nn.BCELoss()(judgement, goal)
        loss.backward()
        optimizer2.step()

    def demonstrate_generator():
        images = gen(random_noise())
        for img in images:
            plot_img(img)
            
    if epoch % printing_steps == 0:
        print('epoch iteration:', epoch, '\n\n')
        demonstrate_generator()





