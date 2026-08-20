import numpy as np


class linear_laser:
    def __init__(self, E0, omega, n_ramp):
        self.E0 = E0
        self.omega = omega
        self.T0 = 2 * np.pi / omega
        self.n_ramp = n_ramp

    def __call__(self, t):
        T0 = self.n_ramp * self.T0
        if t <= T0:
            ft = t / T0
        else:
            ft = 1
        return ft * np.sin(self.omega * t) * self.E0


class sine_laser:
    def __init__(self, E0, omega, td, phase=0.0, start=0.0):
        self.E0 = E0
        self.omega = omega
        self.tprime = td
        self.phase = phase
        self.t0 = start

    def __call__(self, t):
        dt = t - self.t0
        return (
            self.E0
            * np.sin(self.omega * t + self.phase)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
        )


class discrete_delta_pulse:
    def __init__(self, field_strength, dt, t0=0.0, **kwargs):
        self.field_strength = field_strength
        self.dt = dt
        self.t0 = t0

    def __call__(self, t):
        dt = t - self.t0
        return (
            self.field_strength
            * np.heaviside(self.dt - dt, 0.0)
            * np.heaviside(self.dt + dt, 0.0)
        )


class sine_square_laser:
    def __init__(self, E0, omega, td, phase=0.0, start=0.0):
        self.F_str = E0
        self.omega = omega
        self.tprime = td
        self.phase = phase
        self.t0 = start

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.sin(self.omega * dt + self._phase(dt))
            * self.F_str
        )
        return pulse


class square_length_dipole:
    def __init__(
        self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs
    ):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
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
            np.sin(np.pi * dt / self.tprime)
            * (
                self.omega
                * np.sin(np.pi * dt / self.tprime)
                * np.sin(self.omega * dt + self.phase)
                - (2 * np.pi / self.tprime)
                * np.cos(np.pi * dt / self.tprime)
                * np.cos(self.omega * dt + self.phase)
            )
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * self.A0
        )
        return pulse


class square_velocity_dipole:
    def __init__(
        self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs
    ):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
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
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.cos(self.omega * dt + self._phase(dt))
            * self.A0
        )
        return pulse


class square_velocity_first:
    def __init__(
        self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs
    ):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def _phase(self, t):
        if callable(self.phase):
            return self.phase(t)
        else:
            return self.phase

    def __call__(self, t):
        dt = t - self.t0

        A1_t = (
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.cos(self.omega * dt + self._phase(dt))
            * self.A0
        )

        A2_t = (
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.sin(self.omega * dt + self._phase(dt))
            * self.A0
        )

        return A1_t, A2_t


class square_velocity_exp_p:
    def __init__(
        self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs
    ):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.exp(1j * (self.omega * dt + self.phase))
            * self.A0
        )
        return pulse


class square_velocity_exp_m:
    def __init__(
        self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs
    ):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            (np.sin(np.pi * dt / self.tprime) ** 2)
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.exp(-1j * (self.omega * dt + self.phase))
            * self.A0
        )
        return pulse


class square_velocity_exp2_p:
    def __init__(
        self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs
    ):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            (np.sin(np.pi * dt / self.tprime) ** 2) ** 2
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.exp(1j * 2 * (self.omega * dt + self.phase))
            * self.A0**2
        )
        return pulse


class square_velocity_exp2_m:
    def __init__(
        self, field_strength, omega, ncycles, phase=0.0, t0=0.0, **kwargs
    ):
        self.field_strength = field_strength
        self.A0 = field_strength / omega
        self.omega = omega
        self.tprime = 2 * ncycles * np.pi / omega
        self.phase = phase
        self.t0 = t0

    def __call__(self, t):
        dt = t - self.t0
        pulse = (
            (np.sin(np.pi * dt / self.tprime) ** 2) ** 2
            * np.heaviside(dt, 1.0)
            * np.heaviside(self.tprime - dt, 1.0)
            * np.exp(-1j * 2 * (self.omega * dt + self.phase))
            * self.A0**2
        )
        return pulse
