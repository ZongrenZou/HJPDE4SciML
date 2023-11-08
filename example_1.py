import numpy as np
import scipy.io as sio
import numba
import time
import argparse


import methods


parser = argparse.ArgumentParser()
parser.add_argument("--h", type=float, default=0.001, help="step size")

args = parser.parse_args()

###################### Make data ###########################
N = 1000
w = 10
T = 10
x_test = np.linspace(0, T, N + 1).reshape([-1, 1])
y_test = 1 * np.sin(w * x_test)

np.random.seed(9999)
noise = 1
x_train = 10 * np.random.uniform(size=[50000, 1])
y_train = np.sin(w * x_train) + 1 * np.random.normal(size=x_train.shape)


###################### Make basis function ###########################
_x_train = x_train
phi = np.concatenate(
    [
        np.ones_like(_x_train),
        _x_train,
        _x_train**2,
        _x_train**3,
        np.sin(_x_train),
        np.sin(5 * _x_train),
        np.sin(8 * _x_train),
        np.sin(9 * _x_train),
        np.sin(10 * _x_train),
        np.sin(12 * _x_train),
    ],
    axis=-1,
)
b = y_train

phi_test = np.concatenate(
    [
        np.ones_like(x_test),
        x_test,
        x_test**2,
        x_test**3,
        np.sin(x_test),
        np.sin(5 * x_test),
        np.sin(8 * x_test),
        np.sin(9 * x_test),
        np.sin(10 * x_test),
        np.sin(12 * x_test),
    ],
    axis=-1,
)


###################### Solve ###########################
eps = 100
P0 = np.eye(phi.shape[1]) / eps
q0 = np.zeros([phi.shape[1], 1])
# update = numba.njit(methods.update) # very inefficient! update it later.
update = methods.update

h = float(args.h)
P = P0
qs = [q0]
t0 = time.time()
for i in range(50000):
    P, q = update(P, qs[-1], h, phi[i : i + 1, :], b[i : i + 1, :], lamb=1)
    qs += [q]
    if i % 100 == 0:
        # validate every 100 data points
        y_pred = phi_test @ qs[-1]
        t1 = time.time()
        print(
            "Number of seen data points: ",
            i,
            "; error: ",
            np.linalg.norm(y_pred - y_test, 2) / np.linalg.norm(y_test, 2),
        )
        print("Elapsed: ", t1 - t0)
        t0 = time.time()
qs = np.stack(qs, axis=0)

###################### Predict and save ###########################
sio.savemat(
    "./outputs/example_1.mat",
    {
        "phi": phi,
        "x_test": x_test,
        "y_test": y_test,
        "x_train": x_train,
        "y_train": y_train,
        "qs": qs,
    },
)
