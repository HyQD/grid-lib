import numpy as np
import matplotlib.pyplot as plt


from grid_lib.pseudospectral_grids import setup_femdvr_uniform, setup_grid


def get_grid_points(grid):
    if hasattr(grid, "x"):
        return np.asarray(grid.x)
    if hasattr(grid, "r"):
        return np.asarray(grid.r)
    raise AttributeError("Grid object has neither 'x' nor 'r' coordinates.")


def get_grid_weights(grid):
    for attr in ("weights", "w", "weight"):
        if hasattr(grid, attr):
            return np.asarray(getattr(grid, attr))
    raise AttributeError("Grid object has no recognized quadrature weights attribute.")


if __name__ == "__main__":
    
    # 1) Sinc DVR grid
    sinc = setup_grid(
        "sinc",
        {
            "x0": -8.0,
            "xN": 8.0,
            "N": 41,
        },
    )
    
    # 2) Gauss-Legendre-Lobatto grid using endpoint-based setup
    gll = setup_grid(
        "gll",
        {
            "N": 40,
            "x0": -8.0,
            "xN":  8.0,
            "symmetrize": False,
        },
    )
    
    # 3) FEM-DVR grid with custom element boundaries and point counts
    nodes = np.array([-8.0, -3.0, 0.0, 4.0, 8.0])
    n_points = np.array([8, 10, 12, 10])
    femdvr = setup_grid(
        "femdvr",
        {
            "nodes": nodes,
            "n_points": n_points,
            "symmetrize": False,
        },
    )
    
    # 4) Convenience helper for a uniform FEM-DVR grid
    femdvr_uniform = setup_femdvr_uniform(
        x_min=-8.0,
        x_max=8.0,
        n_elements=4,
        points_per_element=15,
        symmetrize=False,
    )
    
    grids = [
        ("Sinc DVR", sinc, np.asarray(sinc.x)),
        ("Gauss-Legendre-Lobatto", gll, np.asarray(gll.r)),
        ("FEM-DVR", femdvr, np.asarray(femdvr.r)),
        ("Uniform FEM-DVR", femdvr_uniform, np.asarray(femdvr_uniform.r)),
    ]

    print(f"Quadrature int exp(-x^2) dx = sqrt(pi) = {np.sqrt(np.pi):.8f}, x=-oo to x=oo")
    for label, grid, x in grids:
        w = get_grid_weights(grid)
        f = np.exp(-(x**2))
        integral_num = np.sum(w * f)
        integral_exact = np.sqrt(np.pi)
        abs_error = abs(integral_num - integral_exact)
        print(
            f"  {label}: I_num={integral_num:.8f}, "
            f"|err|={abs_error:.2g}"
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes = axes.ravel()

    for ax, (label, _, x) in zip(axes, grids):
        y = np.exp(-(x**2))
        ax.plot(x, y, "o-", lw=1.5, label="$e^{-x^2}$")
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle("Function $f(x)=e^{-x^2}$ Sampled on Different Grids")
    plt.show()
