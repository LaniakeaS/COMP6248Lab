import torch
from typing import Tuple


torch.set_default_dtype(torch.float64)

# Q1.1
def sgd_factorise(A_: torch.Tensor, rank: int, num_epochs=1000, lr=0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = A_.shape
    U_ = torch.rand(m, rank)
    V_ = torch.rand(n, rank)
    for epoch in range(num_epochs):
        for r in range(m):
            for c in range(n):
                e = A_[r, c] - U_[r, :] @ V_[c, :]
                U_[r, :] += lr * e * V_[c, :]
                V_[c, :] += lr * e * U_[r, :]
    return U_, V_


# Q1.2
A = torch.tensor([[0.3374, 0.6005, 0.1735],
                  [3.3359, 0.0492, 1.8374],
                  [2.9407, 0.5301, 2.2620]])
U_sgd, V_sgd = sgd_factorise(A, 2)
A_sgd = U_sgd @ V_sgd.t()
print(U_sgd)
print(V_sgd)
print(A_sgd)
print('Error of A_sgd: ', torch.nn.functional.mse_loss(A_sgd, A, reduction='sum'), '\n')

# Q2.1
U_svd, S, V_svd = torch.svd(A)
S[-1] = 0.0
S = torch.diag(S)
A_svd = U_svd @ S @ V_svd.t()
print(U_svd)
print(S)
print(V_svd)
print(A_svd)
print('Error of A_svd: ', torch.nn.functional.mse_loss(A_svd, A, reduction='sum'), '\n')


# Q3.1
def sgd_factorise_masked(A_: torch.Tensor, M_: torch.Tensor, rank: int, num_epochs=1000, lr=0.01) -> Tuple[torch.Tensor, torch.Tensor]:
    m, n = A_.shape
    U_ = torch.rand(m, rank)
    V_ = torch.rand(n, rank)
    A_ *= M_
    for epoch in range(num_epochs):
        for r in range(m):
            for c in range(n):
                e = A_[r, c] - U_[r, :] @ V_[c, :]
                U_[r, :] += lr * e * V_[c, :]
                V_[c, :] += lr * e * U_[r, :]
    return U_, V_


# Q3.2
Mask = torch.Tensor([[1, 1, 1], [0, 1, 1], [1, 0, 1]])
U_sgd_mask, V_sgd_mask = sgd_factorise_masked(A, Mask, 2)
A_sgd_mask = U_sgd_mask @ V_sgd_mask.t()
print(A_sgd_mask)
print('Error of A_sgd_mask: ', torch.nn.functional.mse_loss(A_sgd_mask, A, reduction='sum'), '\n')
