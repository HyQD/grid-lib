import numpy as np


def soft_core_Coulomb(x, Z1=-1.0, Z2=-1.0, x_c=0.0, a=1.0):
    """
    Coulomb potential in 1D.

    Args:
        x (np.ndarray): The grid points.
        Z1 (float): Charge of the first particle.
        Z2 (float): Charge of the second particle.
        x_c (float): The nuclear position.
        a (float): The regularization parameter.
    """
    return Z1 * Z2 / np.sqrt((x - x_c) ** 2 + a**2)


class Molecule1D:
    def __init__(self, R=[0.0], Z=[1], a=1.0):
        """
        Molecular potential in 1D.

        Args:
            R (list): The nuclear positions.
            Z (list): The nuclear charges.
            a (float): The regularization parameter.
        """
        self.R_list = R
        self.Z_list = Z
        if a <= 0:
            raise ValueError("The regularization parameter must be positive.")
        self.a = a

    def __call__(self, x):
        if isinstance(x, float):
            potential = 0
        else:
            potential = np.zeros(len(x))
        for R, Z in zip(self.R_list, self.Z_list):
            potential += soft_core_Coulomb(x, Z1=-1.0, Z2=Z, x_c=R, a=self.a)
        return potential


def harmonic_oscillator(x, omega):
    return 0.5 * omega**2 * x**2
