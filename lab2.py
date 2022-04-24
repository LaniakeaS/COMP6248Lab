import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


torch.set_default_dtype(torch.float64)


# Q1.1
def gd_factorise_ad(A_: torch.Tensor, rank: int, num_epochs=1000, lr=0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = A_.shape
    U_ = torch.rand(m, rank, requires_grad=True)
    V_ = torch.rand(n, rank, requires_grad=True)
    for epoch in range(num_epochs):
        A_sgd = U_ @ V_.t()
        loss = torch.nn.functional.mse_loss(A_sgd, A_, reduction="sum")
        loss.backward(torch.ones(loss.shape))
        with torch.no_grad():
            U_ -= lr * U_.grad
            V_ -= lr * V_.grad
        U_.grad.zero_()
        V_.grad.zero_()
        '''E = A_ - U_ @ V_.t()
        U_ += lr * E @ V_
        V_ += lr * E.t() @ U_'''
    return U_.detach(), V_.detach()


# Q1.2
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases' + '/iris/iris.data', header=None)
data = torch.tensor(df.iloc[:, [0, 1, 2, 3]].values)
data = data - data.mean(dim=0)

U_sgd, V_sgd = gd_factorise_ad(data, 2)
A_sgd = U_sgd @ V_sgd.t()
print('Error of A_sgd: ', torch.nn.functional.mse_loss(A_sgd, data, reduction='sum'), '\n')

U, S, V = torch.svd(data)
S[2:] = 0.0
S = torch.diag(S)
A = U @ S @ V.t()
print('Error of A: ', torch.nn.functional.mse_loss(A, data, reduction='sum'), '\n')

# Q1.3
project = V[:, :2]
data_pca = data @ project
plt.subplot(1, 2, 1)
plt.scatter(data_pca[:, 0], data_pca[:, 1], s=5)
plt.title('Scatter of Projected Data')
plt.subplot(1, 2, 2)
plt.scatter(U_sgd[:, 0], U_sgd[:, 1], s=5)
plt.title('Scatter of Reduced Matrix U')
plt.savefig('pic.eps')
plt.show()

# Q2.1
df = df.sample(frac=1)
mapping = {k: v for v, k in enumerate(df[4].unique())}
df[5] = df[4].map(mapping)
alldata = torch.tensor(df.iloc[:, [0, 1, 2, 3]].values)
alldata = (alldata - alldata.mean(dim=0)) / alldata.var(dim=0)
targets_tr = torch.tensor(df.iloc[:100, 5].values, dtype=torch.double)
targets_va = torch.tensor(df.iloc[100:, 5].values, dtype=torch.double)
data_tr = alldata[:100]
data_va = alldata[100:]
N, features = data_tr.shape

W1 = torch.randn(features)
W2 = torch.randn(1)
b1 = b2 = 0
alpha = 0.0001


def model(X):
    return torch.relu(X @ W1 + b1) * W2 + b2


epochs = 1000
errors_tr = np.zeros(epochs)
errors_va = np.zeros(epochs)
for epoch in range(epochs):
    errors_tr[epoch] = torch.nn.functional.mse_loss(model(data_tr), targets_tr)
    errors_va[epoch] = torch.nn.functional.mse_loss(model(data_va), targets_va)
    for n in range(N):
        grad_b2 = model(data_tr[n]) - targets_tr[n]
        relu = torch.relu(data_tr[n] @ W1 + b1)
        b2 -= alpha * grad_b2
        W2 -= alpha * relu * grad_b2
        if not (relu == 0).all():
            b1 -= alpha * W2 * grad_b2
            W1 -= alpha * W2 * grad_b2 * data_tr[n]
plt.plot(np.arange(epochs), errors_tr)
plt.plot(np.arange(epochs), errors_va)
plt.legend(labels=['train', 'test'])
plt.savefig('2.2.3.eps')
plt.show()
print(model(data_va), '\n', targets_va, '\n')
print(torch.nn.functional.mse_loss(model(data_va), targets_va), '\n')

predicts = model(data_va)
predicts = torch.round(predicts)
print(predicts)
bingo = 0
for i in range(len(predicts)):
    if predicts[i] == targets_va[i]:
        bingo += 1
print(bingo / targets_va.shape[0])

predicts = model(data_tr)
predicts = torch.round(predicts)
print(predicts)
bingo = 0
for i in range(len(predicts)):
    if predicts[i] == targets_tr[i]:
        bingo += 1
print(bingo / targets_tr.shape[0])
