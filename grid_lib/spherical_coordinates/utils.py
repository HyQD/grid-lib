import numpy as np
from opt_einsum import contract


def lowdin_orthogonalize(A, weights):

    n_active = A.shape[0]
    n_r = A.shape[1]
    n_lm = A.shape[2]

    A_ = contract("a, paJ->paJ", np.sqrt(weights), A)
    A_ = A_.reshape((n_active, n_r * n_lm)).T
    S = A_.T.conj() @ A_

    Sigma, X = np.linalg.eigh(S)
    Sm12 = X @ np.diag(Sigma ** (-0.5)) @ X.conj().T
    A_new = (A_ @ Sm12).T

    A_new = A_new.reshape((n_active, n_r, n_lm))
    A_new = contract("paJ, a->paJ", A_new, 1 / np.sqrt(weights))
    return A_new


class Counter:
    # Used to count iterations until convergence in bicgstab
    def __init__(self):
        self.counter = 0

    def __call__(self, x):
        self.counter += 1


def quadrature(weights, f):
    return np.sum(weights * f)


def mask_function(r, r_max, r0, n=4):
    mask_r = np.zeros(len(r))

    ind1 = r < r0
    ind2 = r == r_max
    ind3 = np.invert(ind1 + ind2)

    mask_r[ind1] = 1
    mask_r[ind2] = 0
    mask_r[ind3] = np.cos(np.pi * (r[ind3] - r0) / (2 * (r_max - r0))) ** (
        1 / n
    )

    return mask_r
