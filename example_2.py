import numpy as np
import scipy.io as sio
import numba
import time
import argparse


import methods


parser = argparse.ArgumentParser()
parser.add_argument("--h", type=float, default=0.0001, help="step size")

args = parser.parse_args()


###################### Define some functions ###########################
def basis(x):
    M = np.linspace(1, 10, 10).reshape([1, -1])
    Z = 2 * np.pi * np.matmul(x, M)
    basis_1 = np.sin(Z)
    basis_2 = np.cos(Z)
    basis_3 = np.ones(shape=[x.shape[0], 1])
    _basis = np.concatenate([basis_1, basis_2, basis_3], axis=-1)
    return _basis


def basis_xx(x):
    M = np.linspace(1, 10, 10).reshape([1, -1])
    Z = 2 * np.pi * np.matmul(x, M)
    basis_1 = -((2 * np.pi * M) ** 2) * np.sin(Z)
    basis_2 = -((2 * np.pi * M) ** 2) * np.cos(Z)
    basis_3 = np.zeros(shape=[x.shape[0], 1])
    _basis_xx = np.concatenate(
        [basis_1, basis_2, basis_3],
        axis=-1,
    )
    return _basis_xx


def pde(x):
    phi = basis(x)
    phi_xx = basis_xx(x)
    return 0.01 * phi_xx - phi


###################### Make data ###########################
data = sio.loadmat("./data/example_2_train.mat")
x_test = data["x_test"]
u_test = data["u_test"]
f_test = data["f_test"]
x_train = data["x_train"]
f_train = data["f_train"]
new_x_train = data["new_x_train"]
new_f_train = data["new_f_train"]
all_x_train = data["all_x_train"]
all_f_train = data["all_f_train"]

###################### Make basis function ###########################
phi_f = pde(x_train)
phi_u = basis(np.array([0.0, 1.0]).reshape([-1, 1]))


###################### Regular training ###########################
eps = 1
P0 = np.eye(phi_f.shape[1]) / eps
q0 = np.zeros([phi_f.shape[1], 1])
update = numba.njit(methods.update) # very inefficient! update it later
# update = methods.update

print("Training...")
# feed PDE
h = args.h
P = P0
qs = [q0]
for i in range(x_train.shape[0]):
    P, q = update(P, qs[-1], h, phi_f[i : i + 1, :], f_train[i : i + 1, :])
    qs += [q]

# feed boundary condition
h = 0.0001
P, q = update(P, qs[-1], h, phi_u[0:1, :], np.zeros([1, 1]))
qs += [q]
P, q = update(P, qs[-1], h, phi_u[1:2, :], np.zeros([1, 1]))
qs += [q]

###################### Add six new data ###########################
phi_new = pde(new_x_train)

print("Calibrating by adding six new data...")
h = args.h
q_2 = qs[-1].copy()
for i in range(new_x_train.shape[0]):
    P, q_2 = update(P, q_2, h, phi_new[i : i + 1, :], new_f_train[i : i + 1, :])

###################### Enforce boundary condition ###########################
print("Calibrating by enforcing boundary condition...")

h = args.h
q_3 = q_2.copy()
P, q_3 = update(P, q_3, h, phi_u[0:1, :], np.zeros([1, 1]), lamb=9)
P, q_3 = update(P, q_3, h, phi_u[1:2, :], np.zeros([1, 1]), lamb=9)

###################### Predict and save ###########################
phi_u = basis(x_test)
phi_f = pde(x_test)
sio.savemat(
    "./outputs/example_2.mat",
    {
        "x_test": x_test,
        "f_test": f_test,
        "u_test": u_test,
        "x_train": x_train,
        "f_train": f_train,
        "new_x_train": new_x_train,
        "new_f_train": new_f_train,
        "phi_u": phi_u,
        "phi_f": phi_f,
        "q_1": qs[-1],
        "q_2": q_2,
        "q_3": q_3,
    },
)
