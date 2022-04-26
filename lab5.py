import os.path

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn
from torch import optim
import torchbearer
import numpy as np
import matplotlib.pyplot as plt


seed = 7
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)


class MyDataset(Dataset):
    def __init__(self, size=5000, dim=40, random_offset=0):
        super(MyDataset, self).__init__()
        self.size = size
        self.dim = dim
        self.random_offset = random_offset

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError('{} index out of range'.format(self.__class__.__name__))
        rng_state = torch.get_rng_state()
        torch.manual_seed(index + self.random_offset)
        while True:
            img = torch.zeros(self.dim, self.dim)
            dx = torch.randint(-10, 10, (1,), dtype=torch.float)
            dy = torch.randint(-10, 10, (1,), dtype=torch.float)
            c = torch.randint(-20, 20, (1,), dtype=torch.float)
            params = torch.cat((dy/dx, c))
            xy = torch.randint(0, img.shape[1], (20, 2), dtype=torch.float)
            xy[:, 1] = xy[:, 0] * params[0] + params[1]
            xy.round_()
            xy = xy[xy[:, 1] > 0]
            xy = xy[xy[:, 1] < self.dim]
            xy = xy[xy[:, 0] < self.dim]
            for i in range(xy.shape[0]):
                x, y = xy[i][0], self.dim - xy[i][1]
                img[int(y), int(x)] = 1
            if img.sum() > 2:
                break
        torch.set_rng_state(rng_state)
        return img.unsqueeze(0), params

    def __len__(self):
        return self.size


class SimpleCNNBaseline(nn.Module):
    def __init__(self, input_size, num_classes=2, hidden_size=128, kernel_size=3, padding=1):
        super(SimpleCNNBaseline, self).__init__()
        self.conv = nn.Conv2d(1, 48, (kernel_size, kernel_size), padding=padding)
        self.fc1 = nn.Linear(48 * (input_size + padding * 2 - kernel_size + 1)**2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


class SimpleCNNPooling(nn.Module):
    def __init__(self, num_classes=2, hidden_size=128, kernel_size=3, padding=1):
        super(SimpleCNNPooling, self).__init__()
        self.conv1 = nn.Conv2d(1, 48, (kernel_size, kernel_size), padding=padding)
        self.conv2 = nn.Conv2d(48, 48, (kernel_size, kernel_size), padding=padding)
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(48, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pooling(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


class CNNImprove(nn.Module):
    def __init__(self, num_classes=2, hidden_size=128, kernel_size=3, padding=1):
        super(CNNImprove, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, (kernel_size, kernel_size), padding=padding)
        self.conv2 = nn.Conv2d(48, 48, (kernel_size, kernel_size), padding=padding)
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Linear(48, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        idxx = torch.repeat_interleave(
            torch.arange(-20, 20, dtype=torch.float).unsqueeze(0) / 40.0,
            repeats=40, dim=0
        ).to(x.device)
        idxy = idxx.clone().t()
        idx = torch.stack([idxx, idxy]).unsqueeze(0)
        idx = torch.repeat_interleave(idx, repeats=x.shape[0], dim=0)
        x = torch.cat([x, idx], dim=1)
        out = self.conv1(x)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pooling(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out


train_data = MyDataset()
val_data = MyDataset(size=500, random_offset=33333)
test_data = MyDataset(size=500, random_offset=99999)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(val_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=True)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def run(model, model_parameters_path, load=True):
    opt = optim.Adam(model.parameters())
    loss_func = nn.MSELoss()
    if load and os.path.exists(model_parameters_path):
        model.load_state_dict(torch.load(model_parameters_path))
        past_results = torch.load(model_parameters_path + '_past.pkl')
    else:
        trial = torchbearer.Trial(model, opt, loss_func, metrics=['loss']).to(device)
        trial.with_generators(train_loader, val_generator=val_loader)
        past_results = trial.run(epochs=100)
        torch.save(past_results, model_parameters_path + '_past.pkl')
        torch.save(model.state_dict(), model_parameters_path)

    trial = torchbearer.Trial(model, opt, loss_func, metrics=['loss']).to(device)
    trial.with_generators(train_loader, test_generator=test_loader)
    results = trial.evaluate(data_key=torchbearer.TEST_DATA)
    print(results)
    return past_results


def plot(past, model):
    train_losses = []
    validation_losses = []
    for result in past:
        validation_losses.append(result['loss'])
        train_losses.append(result['running_loss'])
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='training loss')
    plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='validation loss')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title(model._get_name())
    plt.savefig(model._get_name() + '_loss.eps')
    plt.show()

    im, params = test_data[10]
    predict = model(im.unsqueeze(0).to(device)).to(device)[0].detach()
    plt.figure()
    plt.imshow(im.squeeze())
    params_draw(params, 'green')
    params_draw(predict, 'blue')
    plt.savefig(model._get_name() + '.eps')
    plt.show()


def params_draw(params, color):
    x0, x1 = 0, 40
    y0, y1 = 0, 40
    k, b = params
    k, b = -k, (-b + 40)
    px0, px1 = x0 - 5, x1 + 5
    x = torch.linspace(px0, px1, px1 - px0).to(device)
    y = k * x + b
    plt.plot(x.to('cpu'), y.to('cpu'), color=color)
    plt.xlim([x0, x1 - 1])
    plt.ylim([x0, y1 - 1])


model = SimpleCNNBaseline(train_data.dim)
plot(run(model, './SimpleCNNBaseline_weights'), model)

model = SimpleCNNPooling()
plot(run(model, './SimpleCNNPooling_weights'), model)

model = CNNImprove()
plot(run(model, './CNNImprove_weights'), model)
