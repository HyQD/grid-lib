import numpy as np
from matplotlib import pyplot as plt


class SincDvr:
    def __init__(self, x0, xN, N):
        self.x0 = x0
        self.xN = xN
        self.N = N
        self.dx = (xN - x0) / (N - 1)
        self.x = np.linspace(x0, xN, N)
        self.weights = self.dx * np.ones(N)

        self.D1 = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    self.D1[i, j] = 0
                else:
                    self.D1[i, j] = (-1) ** (i - j) / (self.dx * (i - j))

        self.D2 = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    self.D2[i, j] = -np.pi**2 / (3 * self.dx**2)
                else:
                    self.D2[i, j] = (
                        -2 * (-1) ** (i - j) / (self.dx**2 * (i - j) ** 2)
                    )


# if __name__ == "__main__":
#     x0 = -10
#     xN = 10
#     N = 201
#     sinc_dvr = SincDvr(x0, xN, N)

#     T = -0.5 * sinc_dvr.D2
#     V_HO = np.diag(0.5 * sinc_dvr.x**2)
#     V_1D_Hydrogen = np.diag(-1 / np.sqrt(sinc_dvr.x**2 + 2.0))
#     H_HO = T + V_HO
#     H_1D_Hydrogen = T + V_1D_Hydrogen

#     eps_ho, C_ho = np.linalg.eigh(H_HO)
#     print("Eigenvalues:", eps_ho[0:5])
#     eps_hydrogen, C_hydrogen = np.linalg.eigh(H_1D_Hydrogen)
#     print("Eigenvalues:", eps_hydrogen[0:5])

#     test_function = np.exp(-sinc_dvr.x**2)
#     df_dx = sinc_dvr.D1 @ test_function
#     plt.figure()
#     plt.subplot(211)
#     plt.plot(sinc_dvr.x, df_dx, label="Numerical derivative")
#     plt.legend()
#     plt.subplot(212)
#     plt.plot(sinc_dvr.x, df_dx - (-2 * sinc_dvr.x * np.exp(-sinc_dvr.x**2)), label="Error")
#     plt.legend()
#     plt.show()
