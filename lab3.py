import torch
import pandas as pd
from torch.utils import data
import matplotlib.pyplot as plt


'''# Q1.1
def rastrigin(x, A):
    return A * 2 + (x @ x) - torch.sum(A * torch.cos(2 * torch.pi * x))


A = 1
epochs = 100
output = 0
errors = []
steps = []
labels = ['SGD', 'SGD + momentum', 'Adagrad', 'Adam']

x = torch.tensor([5, 5], requires_grad=True, dtype=torch.float64)
opt = torch.optim.SGD([x], lr=0.01)
stepError = []
for epoch in range(epochs):
    opt.zero_grad()
    output = rastrigin(x, A)
    stepError.append(output.detach())
    output.backward()
    opt.step()
steps.append(stepError)
errors.append(output.detach())
print('SGD result: ', x, '\nSGD error: ', output, '\n')

x = torch.tensor([5, 5], requires_grad=True, dtype=torch.float64)
opt = torch.optim.SGD([x], lr=0.01, momentum=0.9)
stepError = []
for epoch in range(epochs):
    opt.zero_grad()
    output = rastrigin(x, A)
    stepError.append(output.detach())
    output.backward()
    opt.step()
steps.append(stepError)
errors.append(output.detach())
print('SGD + momentum result: ', x, '\nSGD + momentum error: ', output, '\n')

x = torch.tensor([5, 5], requires_grad=True, dtype=torch.float64)
opt = torch.optim.Adagrad([x], lr=0.01)
stepError = []
for epoch in range(epochs):
    opt.zero_grad()
    output = rastrigin(x, A)
    stepError.append(output.detach())
    output.backward()
    opt.step()
steps.append(stepError)
errors.append(output.detach())
print('Adagrad result: ', x, '\nAdagrad error: ', output, '\n')

x = torch.tensor([5, 5], requires_grad=True, dtype=torch.float64)
opt = torch.optim.Adam([x], lr=0.01)
stepError = []
for epoch in range(epochs):
    opt.zero_grad()
    output = rastrigin(x, A)
    stepError.append(output.detach())
    output.backward()
    opt.step()
steps.append(stepError)
errors.append(output.detach())
print('Adam result: ', x, '\nAdam error: ', output, '\n')

plt.figure()
plt.bar(labels, errors)
plt.savefig('1.1.eps')
plt.show()

plt.figure()
for i in range(len(steps)):
    plt.subplot(2, 2, i + 1)
    plt.plot(range(1, epochs + 1), steps[i])
    plt.title(labels[i])
plt.savefig('1.1.1.eps')
plt.show()'''

# Q2.1
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases' + '/iris/iris.data', header=None)
df = df.sample(frac=1, random_state=0)
df = df[df[4].isin(['Iris-virginica', 'Iris-versicolor'])]
mapping = {k: v for v, k in enumerate(df[4].unique())}
df[5] = (2 * df[4].map(mapping)) - 1
alldata = torch.tensor(df.iloc[:, [0, 1, 2, 3]].values)
alldata = (alldata - alldata.mean(dim=0)) / alldata.var(dim=0)
targets_tr = torch.tensor(df.iloc[:75, 5].values, dtype=torch.double)
targets_va = torch.tensor(df.iloc[75:, 5].values, dtype=torch.double)
data_tr = alldata[:75]
data_va = alldata[75:]
N, features = data_tr.shape
dataset = data.TensorDataset(data_tr, targets_tr)
dataloader = data.DataLoader(dataset, batch_size=25, shuffle=True)


def hinge_loss(y_pred, y_true):
    loss = torch.ones_like(y_pred) - torch.mul(y_pred, y_true)
    loss = torch.max(torch.zeros_like(loss), loss)
    return torch.sum(loss) / loss.shape[0]


def svm(x, w, b):
    h = (w * x).sum(1) + b
    return h


w = torch.randn(features, requires_grad=True)
b = torch.randn(1, requires_grad=True)
print(w, b)

opt = torch.optim.SGD([w, b], lr=0.01, weight_decay=0.0001)
errors = []

for epoch in range(100):
    for (data, target) in dataloader:
        opt.zero_grad()
        # YOUR CODE HERE
        predict = svm(data, w, b)
        loss = hinge_loss(predict, target)
        errors.append(loss.detach())
        loss.backward()
        opt.step()

plt.plot(range(1, len(errors) + 1), errors)
plt.show()

bingo = 0
predicts = svm(data_va, w, b)
for i in range(predicts.shape[0]):
    predict = 1 if predicts[i] > 0 else -1
    if predict == targets_va[i]:
        bingo += 1
print(bingo / predicts.shape[0])
print(w, b, '\n')

w = torch.randn(features, requires_grad=True)
b = torch.randn(1, requires_grad=True)
print(w, b)

opt = torch.optim.Adam([w, b], lr=0.01, weight_decay=0.0001)
errors = []

for epoch in range(100):
    for (data, target) in dataloader:
        opt.zero_grad()
        # YOUR CODE HERE
        predict = svm(data, w, b)
        loss = hinge_loss(predict, target)
        errors.append(loss.detach())
        loss.backward()
        opt.step()

plt.plot(range(1, len(errors) + 1), errors)
plt.show()

bingo = 0
predicts = svm(data_va, w, b)
for i in range(predicts.shape[0]):
    predict = 1 if predicts[i] > 0 else -1
    if predict == targets_va[i]:
        bingo += 1
print(bingo / predicts.shape[0])
print(w, b, '\n')
