from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

pio.renderers.default = "notebook"
layout = go.Layout(margin=go.layout.Margin(l=0, r=0, b=0, t=0))


def get_error(x1: np.ndarray, x2: np.ndarray, order = "fro", plot: bool = True) -> None:
    err = np.linalg.norm(x2 - x1, ord=order)
    print(f'\tError: {err:.6f}')
    if plot:
        plot_trajectory_matplotlib(xs=x1, baseline_xs=x2)
    return err


def plot_3d_trajectory_matplotlib(
    xs: np.ndarray,
    fig_size: tuple[float, float] = (8, 8),
    linewidth: float = 0.2,
    **kwargs,
) -> None:
    fig = plt.figure(figsize=fig_size)
    fig.tight_layout()
    ax = fig.add_subplot(projection="3d")
    ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], linewidth=linewidth, **kwargs)
    ax.scatter(xs[0, 0], xs[0, 1], xs[0, 2], s=2, c="r")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()
    plt.close()


def plot_trajectory_matplotlib(
    xs: np.ndarray,
    t: Optional[np.ndarray] = None,
    fig_size: tuple[float, float] = (10, 5),
    linewidth: float = 0.2,
    baseline_xs: Optional[np.ndarray] = None,
):
    if t is None:
        t = np.arange(len(xs))

    fig = plt.figure(figsize=fig_size)
    fig.tight_layout()
    for i in range(xs.shape[1]):
        ax = fig.add_subplot(xs.shape[1], 1, i + 1)
        ax.plot(t, xs[:, i], linewidth=linewidth, c="b")
        if baseline_xs is not None:
            ax.plot(t, baseline_xs[:, i], linewidth=linewidth, c="k")
        ax.set_xlabel("t")
        ax.set_ylabel(f"x_{i}")
    plt.show()
    plt.close()


def plot_3d_trajectory_plotly(xs: np.ndarray) -> None:
    fig = go.Figure(data=go.Scatter3d(
        x=xs[:, 0], y=xs[:, 1], z=xs[:, 2],
        line=dict(
            color='darkblue',
            width=2
        )
    ))
    fig.update_layout(
        width=800,
        height=700,
        autosize=False,
        scene=dict(
            camera=dict(
                up=dict(x=0, y=0, z=1),
                eye=dict(x=0, y=1.0707, z=1),
            ),
            aspectratio = dict(x=1, y=1, z=0.7),
            aspectmode = 'manual'
        ),
    )
    fig.show()