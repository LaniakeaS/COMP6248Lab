import os.path

import torch
import torch.nn.functional as F
import torchbearer
import torchvision.transforms as transforms
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import numpy as np


seed = 7
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

# flatten 28*28 images to a 784 vector for each image
transform = transforms.Compose([
    transforms.ToTensor(),  # convert to tensor
    transforms.Lambda(lambda x: x.view(-1))  # flatten into vector
])

# load data
trainset = MNIST(".", train=True, download=True, transform=transform)
testset = MNIST(".", train=False, download=True, transform=transform)

# create data loaders
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=True)


# define baseline model
class BaselineModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(BaselineModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        if not self.training:
            out = F.softmax(out, dim=1)
        return out


hidden_sizes = [1, 10, 100, 1000, 10000, 100000, 200000, 300000]
epochs = 30
load = True
all_test_losses = []
all_test_accs = []
all_train_losses = []
all_train_accs = []
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
for hidden_size in hidden_sizes:
    if load and os.path.exists('data/' + str(hidden_size) + '.pkl'):
        past_results = torch.load('data/' + str(hidden_size) + '.pkl')
    else:
        model = BaselineModel(784, hidden_size, 10)
        loss_func = nn.CrossEntropyLoss()
        opt = optim.Adam(model.parameters())
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        trial = torchbearer.Trial(model, opt, loss_func, metrics=['loss', 'accuracy']).to(device)
        trial.with_generators(trainloader, test_generator=testloader)
        past_results = trial.run(epochs=epochs)
        results = trial.evaluate(data_key=torchbearer.TEST_DATA)
        torch.save(past_results, 'data/' + str(hidden_size) + '.pkl')
    test_losses = []
    test_accs = []
    train_losses = []
    train_accs = []
    for result in past_results:
        test_losses.append(result['loss'])
        test_accs.append(result['acc'])
        train_losses.append(result['running_loss'])
        train_accs.append(result['running_acc'])
    all_test_losses.append(test_losses)
    all_test_accs.append(test_accs)
    all_train_losses.append(train_losses)
    all_train_accs.append(train_accs)

choosen_size_range = [1000, 10000, 100000, 200000]
for i in range(len(hidden_sizes)):
    if hidden_sizes[i] in choosen_size_range:
        test_losses = all_test_losses[i]
        train_losses = all_train_losses[i]
        test_accs = all_test_accs[i]
        train_accs = all_train_accs[i]
        print(test_losses[-1], train_losses[-1], test_accs[-1], train_accs[-1])
        plt.figure()
        plt.title(str(hidden_sizes[i]) + ' hidden nodes')
        plt.plot(range(1, epochs + 1), test_losses, label='test loss')
        plt.plot(range(1, epochs + 1), train_losses, label='train loss')
        plt.legend()
        plt.savefig(str(hidden_sizes[i]) + 'loss.eps')
        plt.show()

        plt.figure()
        plt.title(str(hidden_sizes[i]) + ' hidden nodes')
        plt.plot(range(1, epochs + 1), test_accs, label='test acc')
        plt.plot(range(1, epochs + 1), train_accs, label='train acc')
        plt.legend()
        plt.savefig(str(hidden_sizes[i]) + 'acc.eps')
        plt.show()
