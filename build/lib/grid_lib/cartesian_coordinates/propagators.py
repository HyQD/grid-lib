import numpy as np
from grid_lib.cartesian_coordinates.rhs import Vdirect_phi, Vexchange_phi
from scipy.sparse.linalg import LinearOperator, bicgstab
from grid_lib.spherical_coordinates.utils import (
    Counter,
    quadrature,
)
from grid_lib.cartesian_coordinates.exceptions import ConvergenceError


class ForwardEuler:
    def __init__(self, rhs, dt):
        """
        Initializes the Forward Euler method for a given right-hand side function
        and time step.

        Parameters
        ----------
        rhs : callable
            The right-hand side function of the ODE.
        dt : float
            The time step of the integration.
        """
        self.rhs = rhs
        self.dt = dt

    def step(self, psi, t0, direct=None):
        """
        Integrates the ODE
            dot(psi) = rhs(psi, t)
        from t0 to t0 + dt using the Forward Euler method.
        """
        if direct is None:
            return psi + self.dt * self.rhs(psi, t0)
        else:
            return psi + self.dt * self.rhs(psi, t0, direct)


class RungeKutta2:
    def __init__(self, rhs, dt):
        """
        Initializes the Runge-Kutta 2 method for a given right-hand side function
        and time step.

        Parameters
        ----------
        rhs : callable
            The right-hand side function of the ODE.
        dt : float
            The time step of the integration.
        """
        self.rhs = rhs
        self.dt = dt

    def step(self, psi, t0):
        """
        Integrates the ODE
            dot(psi) = rhs(psi, t)
        from t0 to t0 + dt using the Runge-Kutta 2 method.
        """
        k1 = self.rhs(psi, t0)
        k2 = self.rhs(psi + self.dt * k1, t0 + self.dt)
        return psi + self.dt / 2 * (k1 + k2)


class RungeKutta3:
    def __init__(self, rhs, dt):
        """
        Initializes the Runge-Kutta 3 method for a given right-hand side function
        and time step.

        Parameters
        ----------
        rhs : callable
            The right-hand side function of the ODE.
        dt : float
            The time step of the integration.
        """
        self.rhs = rhs
        self.dt = dt

    def step(self, psi, t0):
        """
        Integrates the ODE
            dot(psi) = rhs(psi, t)
        from t0 to t0 + dt using the Runge-Kutta 3 method.
        """
        k1 = self.rhs(psi, t0)
        k2 = self.rhs(psi + 0.5 * self.dt * k1, t0 + 0.5 * self.dt)
        k3 = self.rhs(psi - self.dt * k1 + 2 * self.dt * k2, t0 + self.dt)
        return psi + self.dt / 6 * (k1 + 4 * k2 + k3)


class RungeKutta4:
    def __init__(self, rhs, dt):
        """
        Initializes the Runge-Kutta 4 method for a given right-hand side function
        and time step.

        Parameters
        ----------
        rhs : callable
            The right-hand side function of the ODE.
        dt : float
            The time step of the integration.
        """
        self.rhs = rhs
        self.dt = dt

    def step(self, psi, t0, direct=None):
        """
        Integrates the ODE
            dot(psi) = rhs(psi, t)
        from t0 to t0 + dt using the Runge-Kutta 4 method.
        """
        if direct is None:
            k1 = self.rhs(psi, t0)
            k2 = self.rhs(psi + 0.5 * self.dt * k1, t0 + 0.5 * self.dt)
            k3 = self.rhs(psi + 0.5 * self.dt * k2, t0 + 0.5 * self.dt)
            k4 = self.rhs(psi + self.dt * k3, t0 + self.dt)
        else:
            k1 = self.rhs(psi, t0, direct)
            k2 = self.rhs(psi + 0.5 * self.dt * k1, t0 + 0.5 * self.dt, direct)
            k3 = self.rhs(psi + 0.5 * self.dt * k2, t0 + 0.5 * self.dt, direct)
            k4 = self.rhs(psi + self.dt * k3, t0 + self.dt, direct)
        return psi + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class Rk4:
    def __init__(self, H0, w12, x, e_field, n_docc, dt, CMF=False):
        self.H0 = H0
        self.w12 = w12
        self.x = x
        self.n_dvr = len(x)
        self.n_docc = n_docc
        self.e_field = e_field
        self.dt = dt
        self.CMF = CMF

    def rhs(self, psi, phi, t):
        H0_psi = np.dot(self.H0, phi)
        et_psi = self.e_field(t) * np.einsum("i,ij->ij", self.x, phi)

        Vdir_psi = Vdirect_phi(psi, phi, self.w12)
        Vex_psi = Vexchange_phi(psi, phi, self.w12)

        return -1j * (H0_psi - et_psi + 2 * Vdir_psi - Vex_psi)

    def step(self, phi, t0):

        """
        Integrate the ODE
            i dot(phi)_i(t) = F(phi, t)*phi_i(t)
        from tn to tn + dt using the fourth-order Runge-Kutta method.
        """
        if self.CMF:
            psi = phi.copy()
            k1 = self.rhs(psi, phi, t0)
            k2 = self.rhs(psi, phi + self.dt / 2 * k1, t0 + self.dt / 2)
            k3 = self.rhs(psi, phi + self.dt / 2 * k2, t0 + self.dt / 2)
            k4 = self.rhs(psi, phi + self.dt * k3, t0 + self.dt)
        else:
            k1 = self.rhs(phi, phi, t0)
            k1_phi = phi + self.dt / 2 * k1
            k2 = self.rhs(k1_phi, k1_phi, t0 + self.dt / 2)
            k2_phi = phi + self.dt / 2 * k2
            k3 = self.rhs(k2_phi, k2_phi, t0 + self.dt / 2)
            k3_phi = phi + self.dt * k3
            k4 = self.rhs(k3_phi, k3_phi, t0 + self.dt)

        return phi + self.dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


class CrankNicolson:
    def __init__(self, H0, w12, x, e_field, n_docc, dt, rtol=1e-10):
        self.H0 = H0
        self.w12 = w12
        self.x = x
        self.n_dvr = len(x)
        self.n_docc = n_docc
        self.e_field = e_field
        self.dt = dt
        I = np.complex128(np.eye(self.n_dvr))
        self.M = np.linalg.inv(I + 1j * self.dt / 2 * self.H0)
        self.rtol = rtol

    def M2phi(self, phi):
        phi = phi.reshape((self.n_dvr, self.n_docc))
        return np.dot(self.M, phi).ravel()

    def Fphi(self, psi, phi, t):

        phi = phi.reshape((self.n_dvr, self.n_docc))
        H0_phi = np.dot(self.H0, phi)
        et_phi = self.e_field(t) * np.einsum("i,ij->ij", self.x, phi)

        Vdir_phi = Vdirect_phi(psi, phi, self.w12)
        Vex_phi = Vexchange_phi(psi, phi, self.w12)

        F_phi = H0_phi - et_phi + 2 * Vdir_phi - Vex_phi

        return F_phi

    def step(self, phi, t0):

        """
        Integrate the ODE
            i dot(phi)_i(t) = F(phi, t)*phi_i(t)
        from tn to tn + dt using the implicit midpoint rule.
        """
        tn = t0 + self.dt / 2
        psi = phi.copy()

        psi_tmp = (
            phi.ravel() - 1j * self.dt / 2 * self.Fphi(psi, phi, tn).ravel()
        )

        Ap_lambda = (
            lambda phi, tn=tn, psi=psi: phi.ravel()
            + 1j * self.dt / 2 * self.Fphi(psi, phi, tn).ravel()
        )
        Ap_linear = LinearOperator(
            dtype=np.complex128,
            shape=(self.n_docc * self.n_dvr, self.n_docc * self.n_dvr),
            matvec=Ap_lambda,
        )
        M_linear = LinearOperator(
            (self.n_docc * self.n_dvr, self.n_docc * self.n_dvr),
            matvec=self.M2phi,
        )

        local_counter = Counter()
        phi, info = bicgstab(
            Ap_linear,
            psi_tmp,
            M=M_linear,
            x0=phi.ravel(),
            tol=self.rtol,
            callback=local_counter,
        )
        phi = phi.reshape((self.n_dvr, self.n_docc))

        if info != 0:
            raise ConvergenceError("BICGSTAB did not converge")

        return phi


class CMF2:
    def __init__(self, H0, w12, x, e_field, n_docc, dt, rtol=1e-10):
        self.H0 = H0
        self.w12 = w12
        self.x = x
        self.n_dvr = len(x)
        self.n_docc = n_docc
        self.e_field = e_field
        self.dt = dt
        I = np.complex128(np.eye(self.n_dvr))
        self.M = np.linalg.inv(I + 1j * self.dt / 4 * self.H0)
        self.rtol = rtol

    def M2phi(self, phi):
        phi = phi.reshape((self.n_dvr, self.n_docc))
        return np.dot(self.M, phi).ravel()

    def Fphi(self, psi, phi, t):

        phi = phi.reshape((self.n_dvr, self.n_docc))
        H0_phi = np.dot(self.H0, phi)
        et_phi = self.e_field(t) * np.einsum("i,ij->ij", self.x, phi)

        Vdir_phi = Vdirect_phi(psi, phi, self.w12)
        Vex_phi = Vexchange_phi(psi, phi, self.w12)

        F_phi = H0_phi - et_phi + 2 * Vdir_phi - Vex_phi

        return F_phi

    def local_step(self, psi, phi, t0, tstop):

        t_mid = (t0 + tstop) / 2
        step_length = tstop - t0

        psi_tmp = (
            phi.ravel()
            - 1j * step_length / 2 * self.Fphi(psi, phi, t_mid).ravel()
        )

        Ap_lambda = (
            lambda phi, tn=t_mid, psi=psi: phi.ravel()
            + 1j * step_length / 2 * self.Fphi(psi, phi, tn).ravel()
        )
        Ap_linear = LinearOperator(
            dtype=np.complex128,
            shape=(self.n_docc * self.n_dvr, self.n_docc * self.n_dvr),
            matvec=Ap_lambda,
        )
        M_linear = LinearOperator(
            (self.n_docc * self.n_dvr, self.n_docc * self.n_dvr),
            matvec=self.M2phi,
        )

        local_counter = Counter()
        phi_new, info = bicgstab(
            Ap_linear,
            psi_tmp,
            M=M_linear,
            x0=phi.ravel(),
            tol=self.rtol,
            callback=local_counter,
        )
        phi_new = phi_new.reshape((self.n_dvr, self.n_docc))

        if info != 0:
            raise ConvergenceError("BICGSTAB did not converge")

        return phi_new

    def step(self, phi, t0):

        """
        Integrate the ODE
            i dot(phi)_i(t) = F(phi, t)*phi_i(t)
        from tn to tn + dt using the implicit midpoint rule.
        """

        psi = phi.copy()

        phi_tmp = self.local_step(psi, phi, t0, t0 + self.dt / 2)
        phi_half = self.local_step(phi_tmp, phi, t0, t0 + self.dt / 2)
        phi_new = self.local_step(
            phi_tmp, phi_half, t0 + self.dt / 2, t0 + self.dt
        )

        return phi_new
