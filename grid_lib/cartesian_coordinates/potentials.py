import numpy as np


def Coulomb(x, Z, x_c=0.0, alpha=1.0):
    """
    Coulomb potential in 1D.

    Args:
        x (np.ndarray): The grid points.
        Z (float): The nuclear charge.
        x_c (float): The nuclear position.
        a (float): The regularization parameter.
    """
    return -Z / np.sqrt((x - x_c) ** 2 + alpha)


class SoftCoreCoulomb:

    def __init__(self, Z1=-1.0, Z2=-1.0, x_c=0.0, a=1.0):
        self.Z1 = Z1
        self.Z2 = Z2
        self.x_c = x_c
        if a <= 0:
            raise ValueError("The regularization parameter must be positive.")
        self.a = a

    def __call__(self, x):
        return self.Z1 * self.Z2 / np.sqrt((x - self.x_c) ** 2 + self.a**2)

class Molecule1D:
    def __init__(self, R=[0.0], Z=[1], alpha=1.0):
        """
        Molecular potential in 1D.

        Args:
            R (list): The nuclear positions.
            Z (list): The nuclear charges.
            alpha (float): The regularization parameter.
        """
        self.R_list = R
        self.Z_list = Z
        if alpha <= 0:
            raise ValueError("The regularization parameter must be positive.")
        self.alpha = alpha

    def __call__(self, x):
        if isinstance(x, float):
            potential = 0
        else:
            potential = np.zeros(len(x))
        for R, Z in zip(self.R_list, self.Z_list):
            potential += Coulomb(x, Z=Z, x_c=R, alpha=self.alpha)
        return potential


class HarmonicOscillator:
    def __init__(self, omega=1.0, x_c=0.0):
        self.omega = omega
        self.x_c = x_c

    def __call__(self, x):
        return 0.5 * self.omega**2 * (x - self.x_c)**2

def harmonic_oscillator(x, omega, x_c=0.0):
    return 0.5 * omega**2 * (x - x_c)**2

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
