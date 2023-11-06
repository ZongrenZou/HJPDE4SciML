"""Code for example 4: the first equation"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

import methods


###################### Load data ######################
data = sio.loadmat("./data/example_4_train.mat")
u = data["u"]
u_t = data["u_t"]
t = data["t"]


###################### Form basis ######################
def form_basis(x):
    x1, x2, x3 = np.split(x, 3, axis=-1)
    basis_0 = np.ones(shape=[x1.shape[0], 1])
    basis_1 = np.concatenate([x1, x2, x3], axis=-1)
    basis_2 = np.concatenate(
        [x1**2, x2**2, x3**2, x1 * x2, x2 * x3, x1 * x3], axis=-1
    )
    basis = np.concatenate([basis_0, basis_1, basis_2], axis=-1)
    return basis


phi = form_basis(u)


###################### the original PDHG ######################
print("The original PDHG...")
eps = 0.1
# initialization and hyperparameters
y = np.zeros(shape=[10, 1], dtype=np.float64)
c1 = np.zeros(shape=[10, 1], dtype=np.float64)
tau = 0.5
sigma = 0.5

A = tau * np.matmul(phi.T, phi) + np.eye(10, dtype=np.float64)
fs = []

# PDHG
for i in range(20000):
    # step 1
    _b = tau * np.matmul(phi.T, u_t[:, 0:1]) + c1 - tau * y
    # new_c = np.matmul(inv, _b)
    new_c = np.linalg.solve(A, _b)
    # step 2
    _c = 2 * new_c - c1
    # step 3
    _v = y + sigma * _c
    y = (
        eps * (_v > eps).astype(np.float64)
        + _v * (np.abs(_v) <= eps).astype(np.float64)
        - eps * (_v < -eps).astype(np.float64)
    )
    # update
    c1 = new_c
    # compute the objective
    u_t_pred = np.matmul(phi, c1)
    fs += [0.5 * np.sum((u_t_pred - u_t[:, 0:1]) ** 2) + eps * np.sum(np.abs(c1))]
c1_ref = c1
print("Identified coefficients:")
print(np.round(c1_ref.flatten(), 8))


###################### PDHG with Riccati ######################
print("PDHG with Riccati...")
eps = 1
tau = 0.5
P0 = np.eye(phi.shape[1]) / eps
q0 = np.zeros([phi.shape[1], 1])

h = 0.001
P2s = [P0]
q2s = [q0]
for i in range(u_t.shape[0]):
    P, q = methods.update_RK4(
        P2s[-1], q2s[-1], h, phi[i : i + 1, :], u_t[i : i + 1, 0:1], lamb=1, tau=tau
    )
    P2s += [P]
    q2s += [q]
P, q = P, q

eps = 0.1
# initialization and hyperparameters
y = np.zeros(shape=[10, 1], dtype=np.float64)
c1 = np.zeros(shape=[10, 1], dtype=np.float64)
tau = 0.5
sigma = 0.5

fs = []

# PDHG
for i in range(20000):
    # step 1
    new_c = np.matmul(P, c1 - tau * y) + q
    # step 2
    _c = 2 * new_c - c1
    # step 3
    _v = y + sigma * _c
    y = (
        eps * (_v > eps).astype(np.float64)
        + _v * (np.abs(_v) <= eps).astype(np.float64)
        - eps * (_v < -eps).astype(np.float64)
    )
    # update
    c1 = new_c
    # compute the objective
    u_t_pred = np.matmul(phi, c1)
    fs += [0.5 * np.sum((u_t_pred - u_t[:, 0:1]) ** 2) + eps * np.sum(np.abs(c1))]


###################### Results ######################
print("L1: ", np.linalg.norm(c1 - c1_ref, 1))
print("L1 relative: ", np.linalg.norm(c1 - c1_ref, 1) / np.linalg.norm(c1_ref, 1))
print("Identified coefficients:")
print(np.round(c1.flatten(), 8))
