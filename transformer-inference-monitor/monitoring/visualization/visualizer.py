from typing import Callable, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

from midiai.inspector.monitoring.constants import PITCH_MAP
from midiai.inspector.monitoring.types import Figure, Indices
from midiai.inspector.monitoring.utils import ExcessiveDisplayChecker
from midiai.inspector.monitoring.utils import TextColors as TC
from midiai.inspector.monitoring.visualization.plotters import (
    AttentionMapPlotter,
    BarcodeStylePlotter,
    EmbeddingWeightsPlotter,
    HistogramPlotter,
    PitchMatrixPlotter,
    Plotter,
    TimeAxisPlotter,
    VocabAxisPlotter,
)
from midiai.inspector.monitoring.vocab_utils import FIELD_VOCAB_RANGE, PITCH
from midiai.inspector.monitoring.vocab_utils import TokenFields as TF

DEFAULT_VOCAB_AXIS_PLOT_SIZE = (7, 2)
DEFAULT_TIME_AXIS_PLOT_SIZE = (12, 2.5)
SINGLE_BAR_PLOT_SIZE = (6, 0.3)
SINGLE_PITCH_PLOT_SIZE = (6, 3)


def open_and_close_canvas(func: Callable) -> Callable:
    """matplotlib의 Figure 객체를 열고 닫는 데코레이터"""

    def wrapper(self, *args, **kwargs) -> None:
        fig = plt.figure()
        func(self, fig, *args, **kwargs)
        fig.tight_layout()
        plt.show()
        plt.close()

    return wrapper


class Visualizer:
    """
    모니터링할 피처들의 value를 받아, 개별 메서드에 지정된 대로 시각화를 해주는 객체.
    개별 메서드들은, 모니터링할 피처들마다 하나씩 할당하는 것이 원칙.
    하나의 figure 내에 그려질 subplot들을 그리는 작업은
    global attributes로 지정된 세부 Plotter들이 수행함.
    """

    excessive_display_checker = ExcessiveDisplayChecker()
    embedding_Weights_plotter = EmbeddingWeightsPlotter()
    vocab_axis_plotter = VocabAxisPlotter()
    time_axis_plotter = TimeAxisPlotter()
    barcode_style_plotter = BarcodeStylePlotter()
    pitch_matrix_plotter = PitchMatrixPlotter()
    attention_map_plotter = AttentionMapPlotter()
    histogram_plotter = HistogramPlotter()

    LOGITPROBS_VISUALIZING_CFG: dict[str, dict[str, Union[Plotter, int, str]]] = {
        TF.NOTE_POSITION: {
            "Plotter": barcode_style_plotter,
            "vertical_size": 1,
            "kwargs": {
                "cmap": "Greys",
                "subtitle": "Note Position",
            },
        },
        TF.VELOCITY: {
            "Plotter": barcode_style_plotter,
            "vertical_size": 1,
            "kwargs": {
                "cmap": "Reds",
                "subtitle": "Velocity",
            },
        },
        TF.PITCH: {
            "Plotter": pitch_matrix_plotter,
            "vertical_size": 2.5,
            "kwargs": {
                "subtitle": "Pitch",
            },
        },
        TF.NOTE_DURATION: {
            "Plotter": barcode_style_plotter,
            "vertical_size": 1,
            "kwargs": {
                "cmap": "Blues",
                "subtitle": "Note Duration",
            },
        },
    }

    @open_and_close_canvas
    def plot_word_embedding(
        self,
        fig: Figure,
        x: np.ndarray,
        field: Optional[str] = None,
        **plotter_kwargs,
    ) -> None:
        ax = fig.add_subplot()
        self.embedding_Weights_plotter.plot(ax, x, **plotter_kwargs)
        fig.suptitle("Embedding weights")

    @open_and_close_canvas
    def plot_logitprobs_single_field(
        self,
        fig: Figure,
        xs: np.ndarray,
        types: str,
        field: Optional[str] = None,
        horizontal_size: float = 10.0,
        title: Optional[str] = None,
        **plotter_kwargs,
    ) -> None:
        if len(xs.shape) == 1:
            xs = [xs]
        num_subplots = len(xs)
        if res := self.excessive_display_checker._check(num_subplots):  # noqa: F841
            return
        vertical_size = 0
        config = self.LOGITPROBS_VISUALIZING_CFG.get(field)
        for i, x in enumerate(xs, start=1):
            ax = fig.add_subplot(num_subplots, 1, i)
            if config is None:
                ax = self.vocab_axis_plotter.plot(ax, x, ylabel=types, **plotter_kwargs)
                horizontal_size = 10
                vertical_size += 2
            else:
                vocab_range = FIELD_VOCAB_RANGE.get(field)
                plotter, vert, kwargs = config.values()
                ax = plotter.plot(ax, x[vocab_range], **kwargs)
                horizontal_size = 7
                vertical_size += vert

        fig.set_size_inches(horizontal_size, vertical_size)
        if title is not None:
            fig.suptitle(title, fontsize=9)

    @open_and_close_canvas
    def plot_logitprobs_multiple_fields(
        self,
        fig: Figure,
        xs: np.ndarray,
        fields: list[Optional[str]] = None,
        title: Optional[str] = None,
        **plotter_kwargs,
    ):  # TODO: add_gridspec() 사용하여 subplot들의 vertical size 개선
        if len(xs.shape) == 1:
            xs = [xs]
        num_subplots = len(xs)

        assert num_subplots == len(fields), "len(xs) != len(fields)!"
        if res := self.excessive_display_checker._check(num_subplots):  # noqa: F841
            return
        vertical_size = 0
        for i, (x, field) in enumerate(zip(xs, fields), start=1):
            config = self.LOGITPROBS_VISUALIZING_CFG.get(field)
            ax = fig.add_subplot(num_subplots, 1, i)
            if config is None:
                ax = self.vocab_axis_plotter.plot(ax, x)
                horizontal_size = 10
                vertical_size += 3
            else:
                vocab_range = FIELD_VOCAB_RANGE.get(field)
                plotter, vert, kwargs = config.values()
                ax = plotter.plot(ax, x[vocab_range], **kwargs)
                horizontal_size = 7
                vertical_size += vert

        fig.set_size_inches(horizontal_size, vertical_size)

        if title is not None:
            fig.suptitle(title, fontsize=9)

    @open_and_close_canvas
    def plot_attention_map(
        self,
        fig: Figure,
        xs: np.ndarray,
        axis: Indices,
        title: Optional[str] = None,
        **plotter_kwargs,
    ) -> None:
        ax = fig.add_subplot()
        if len(xs.shape) == 1:  # line plot
            ax = self.time_axis_plotter.plot(
                ax,
                y=xs.squeeze(),
                x=axis,
                ylabel="Attention scores",
                xlabel="Step",
                **plotter_kwargs,
            )
            fig.set_size_inches(12, 4)
        else:  # matrix plot
            ax = self.attention_map_plotter.plot(ax, xs, **plotter_kwargs)
            fig.set_size_inches(8, 8)
        if title is not None:
            fig.suptitle(title, fontsize=9)

    @open_and_close_canvas
    def plot_time_series(
        self,
        fig: Figure,
        title: Optional[str] = None,
        **plotter_kwargs,
    ) -> None:
        ax = fig.add_subplot()
        ax = self.time_axis_plotter.plot(ax, **plotter_kwargs)
        fig.set_size_inches(12, 3.5)
        if title is not None:
            fig.suptitle(title)

    @open_and_close_canvas
    def plot_histogram(
        self,
        fig: Figure,
        y: np.ndarray,
        title: Optional[str] = None,
        **plotter_kwargs,
    ) -> None:
        ax = fig.add_subplot()
        if title == "Onset Histogram":
            labels = list(range(y.shape[0]))
        elif title == "Pitch Class Histogram":
            labels = list(PITCH_MAP.values())
        else:
            raise ValueError(f"Unknown 1-D Feature name: {TC.WARN}{title}{TC.ENDC}")
        ax = self.histogram_plotter.plot(ax, y, labels=labels, **plotter_kwargs)
        fig.set_size_inches(6, 2)
        if title is not None:
            fig.suptitle(title, fontsize=10)

    @open_and_close_canvas
    def plot_piano_roll(
        self,
        fig: Figure,
        piano_roll: np.ndarray,
        num_bars: int,
        cmap: str = "summer",
        **plotter_kwargs,
    ) -> None:
        ax = fig.add_subplot()
        # pitch_min = ...  # TODO: pitch_range 범위에 맞춰 ylim 할당
        # pitch_max = ...
        # pr = pr[:, pitch_range[0]: pitch_range[1] + 1]
        ax.imshow(piano_roll.T, aspect="auto", cmap=cmap, **plotter_kwargs)
        bar_pos = int(piano_roll.shape[0] / num_bars)
        for i in range(1, num_bars):
            ax.plot(
                [i * bar_pos] * PITCH.vocab_size,
                range(0, PITCH.vocab_size),
                "--",
                c="k",
                linewidth=0.7,
            )
        ax.set_xlabel("Ticks")
        ax.set_ylabel("Pitch")
        ax.set_ylim([0, PITCH.vocab_size - 1])
        ax.tick_params(axis="both", which="major", labelsize=8)
        fig.set_size_inches(num_bars, 3.5)
        # ax.set_ylim(0, pitch_range[1] - pitch_range[0])

    @open_and_close_canvas
    def plot_matrix_2D(
        self,
        fig: Figure,
        x: np.ndarray,
        title: str,
        cmap: str = "summer",
        **plotter_kwargs,
    ) -> None:
        ax = fig.add_subplot()
        ax.matshow(x, cmap=cmap, **plotter_kwargs)
        if title == "Grooving pattern similarity":  # TODO: plotter 따로 둘 것
            ax.set_title("Grooving pattern similarity")
            ax.set_xlabel("Bars")
            ax.set_ylabel("Bars")
        else:
            raise ValueError(f"Unknown 2-D feature type: {TC.WARN}{title}{TC.ENDC}")
        ax.tick_params(axis="both", which="major", labelsize=8)


VISUALIZER = Visualizer()
