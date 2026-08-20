import numpy as np


def scf_diagonalization(
    H, w12, n_docc, conv_tol_grad=1e-8, max_iters=100, verbose=True
):
    """
    Perform self-consistent field iterations

    Parameters
    ----------
    H : np.ndarray
        One-electron Hamiltonian
    w12 : np.ndarray
        Particle-particle interaction
    n_docc : int
        Number of doubly occupied orbitals
    conv_tol_grad : float
        Gradient convergence tolerance
    max_iters : int
        Maximum number of iterations
    verbose : bool
        Print SCF iterations information
    """

    n_dvr = H.shape[0]

    occ = slice(0, n_docc)

    I = np.eye(n_dvr)

    grad_norm = 100.0
    iters = 0

    eps, C = np.linalg.eigh(H)  # Initial guess hcore

    while grad_norm > conv_tol_grad and iters < max_iters:
        D = np.einsum("bj,gj->bg", C[:, occ].conj(), C[:, occ])
        V_dir = np.einsum(
            "bb, ab, ag->ag", D, w12, I, optimize=True
        )  # V_dir is ends up as a diagonal matrix
        V_ex = np.einsum(
            "dg, gd->gd", D, w12, optimize=True
        )  # V_ex is a dense matrix
        F = H + 2 * V_dir - V_ex

        eps, C = np.linalg.eigh(F)
        e_rhf = np.trace(np.dot(D, H))
        e_rhf += np.trace(np.dot(D, F))
        grad_norm = np.linalg.norm(np.dot(F, D) - np.dot(D, F))
        iters += 1
        if verbose:
            print(
                f"ERHF: {e_rhf:.8f}, Iters: {iters}, ||FD-DF||: {grad_norm:.2e}"
            )

    return eps[occ], e_rhf, C[:, occ]
