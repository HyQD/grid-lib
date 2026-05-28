import numpy as np
import matplotlib.pyplot as plt

from grid_lib.pseudospectral_grids import Grid1D


def main() -> None:
    x_min = -5.0
    x_max = 5.0

    grids = [
        (
            "Sinc",
            Grid1D(
                (
                    "sinc",
                    {
                        "x0": x_min,
                        "xN": x_max,
                        "N": 121,
                    },
                )
            ),
        ),
        (
            "GLL",
            Grid1D(
                (
                    "gll",
                    {
                        "N": 80,
                        "x0": x_min,
                        "xN": x_max,
                        "symmetrize": False,
                        "remove_boundaries": False,
                    },
                )
            ),
        ),
        (
            "FEMDVR Uniform",
            Grid1D(
                (
                    "femdvr_uniform",
                    {
                        "x_min": x_min,
                        "x_max": x_max,
                        "n_elements": 8,
                        "points_per_element": 16,
                        "symmetrize": False,
                        "remove_boundaries": False,
                    },
                )
            ),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    for ax, (label, grid) in zip(axes, grids):
        x = grid.x
        y = np.exp(-(x**2))

        ax.plot(x, y, "o-", lw=1.2, ms=3, label=r"$e^{-x^2}$")
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_xlim(x_min, x_max)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle("f(x) = exp(-x^2) on 1D Grids over [-5, 5]")
    plt.show()


if __name__ == "__main__":
    main()
