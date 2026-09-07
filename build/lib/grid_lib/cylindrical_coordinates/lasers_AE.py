import numpy as np


class sine_square_A_velocity:
    def __init__(self, E0, omega, td, phase=0., t0=0.):
        self.E0 = E0
        self.A0 = E0/omega
        self.omega = omega
        self.td = td
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            (np.sin(np.pi * dt / self.td) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.td - dt, 1.0)
            * np.cos(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse


class sine_square_A_length:
    def __init__(self, E0, omega, td, phase=0., t0=0.):
        self.E0 = E0
        self.A0 = E0/omega
        self.omega = omega
        self.td = td
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            np.sin(np.pi*dt/self.td)*( self.omega*np.sin(np.pi*dt/self.td)*np.sin(self.omega*dt + self.phase)
            - (2*np.pi/self.td)*np.cos(np.pi*dt/self.td)*np.cos(self.omega*dt + self.phase) )
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.td - dt, 1.0)
            * self.A0
        )
        return pulse



class gaussian_A_velocity:
    def __init__(self, E0, omega, sigma, phase=0., t0=0.):
        self.E0 = E0
        self.A0 = E0/omega
        self.omega = omega
        self.phase = phase
        self.t0 = t0
        self.sigma2 = sigma**2

    @property
    def t0_at_center(self):
        return True

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            np.exp(-dt**2/(2*self.sigma2))
            * np.cos(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse

class gaussian_A_length:
    def __init__(self, E0, omega, sigma, phase=0., t0=0.):
        self.E0 = E0
        self.A0 = E0/omega
        self.omega = omega
        self.phase = phase
        self.t0 = t0
        self.sigma2 = sigma**2

    @property
    def t0_at_center(self):
        return True

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            np.exp(-dt**2/(2*self.sigma2))
            * ( (dt/self.sigma2)*np.cos(self.omega * dt + self._phase(dt))
              + self.omega*np.sin(self.omega * dt + self._phase(dt))  )
            * self.A0
        )
        return pulse