import numpy as np
from opt_einsum import contract


def H0_psi(psi, ddr, potential, centrifugal_force_lm, centrifugal_force_r):
    psi_new = contract("Ij, ij->Ii", psi, ddr)
    psi_new += contract("Ik, k->Ik", psi, potential)

    psi_temp = contract("I,Ii->Ii", centrifugal_force_lm, psi)
    psi_new += contract("i, Ii->Ii", centrifugal_force_r, psi_temp)

    return psi_new


def x_psi(psi, x_Omega, r):
    tmp_x = contract("IJ, Jk->Ik", x_Omega, psi)

    return contract("Ik, k->Ik", tmp_x, r)


def y_psi(psi, y_Omega, r):
    tmp_y = contract("IJ, Jk->Ik", y_Omega, psi)

    return contract("Ik, k->Ik", tmp_y, r)


def z_psi(psi, z_Omega, r):
    tmp_z = contract("IJ, Jk->Ik", z_Omega, psi)

    return contract("Ik, k->Ik", tmp_z, r)


def x_x_psi(psi, x_x_Omega, r):
    tmp_x_x = contract("IJ, Jk->Ik", x_x_Omega, psi)

    return contract("Ik, k->Ik", tmp_x_x, r**2)


def y_y_psi(psi, y_y_Omega, r):
    tmp_y_y = contract("IJ, Jk->Ik", y_y_Omega, psi)

    return contract("Ik, k->Ik", tmp_y_y, r**2)


def px_psi(psi, dpsi_dr, x_Omega, H_x_beta, r_inv):
    tmp_x = contract("IJ, Jk->Ik", x_Omega, dpsi_dr)
    psi_beta_x = contract("IJ, Jk->Ik", H_x_beta, psi)
    tmp_x += contract("Ik, k->Ik", psi_beta_x, r_inv)

    return -1j * tmp_x


def py_psi(psi, dpsi_dr, y_Omega, H_y_beta, r_inv):
    tmp_y = contract("IJ, Jk->Ik", y_Omega, dpsi_dr)
    psi_beta_y = contract("IJ, Jk->Ik", H_y_beta, psi)
    tmp_y += contract("Ik, k->Ik", psi_beta_y, r_inv)

    return -1j * tmp_y


def pz_psi(psi, dpsi_dr, z_Omega, H_z_beta, r_inv):
    tmp_z = contract("IJ, Jk->Ik", z_Omega, dpsi_dr)
    psi_beta_z = contract("IJ, Jk->Ik", H_z_beta, psi)
    tmp_z += contract("Ik, k->Ik", psi_beta_z, r_inv)

    return -1j * tmp_z


def y_px_psi(psi, dpsi_dr, y_x_Omega, y_px_beta, r):
    tmp_y_px = contract("IJ, Jk->Ik", y_x_Omega, dpsi_dr)
    out = contract("Ik, k->Ik", tmp_y_px, r)
    out += contract("IJ, Jk->Ik", y_px_beta, psi)

    return -1j * out
