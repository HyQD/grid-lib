import numpy as np


class RadialMatrixElements:
    def __init__(self, grid_object):
        if grid_object.__class__.__name__ == "GaussLegendreLobatto":
            self.compute_matrix_elements_gauss_legendre_lobatto(grid_object)

    def compute_matrix_elements_gauss_legendre_lobatto(self, gll):
        self.r = gll.r[1:-1]
        self.r_inv = 1 / self.r
        self.r_dot = gll.r_dot[1:-1]
        self.D1 = gll.D1[1:-1, 1:-1]
        self.D2 = gll.D2[1:-1, 1:-1]
        self.nr = len(self.r)
        self.weights = gll.weights[1:-1]
