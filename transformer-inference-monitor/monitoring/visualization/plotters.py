import abc
from typing import Any, Optional

import numpy as np

from midiai.inspector.monitoring.constants import NUM_OCTAVES, NUM_PITCHES, OCTAVES, PITCH_MAP
from midiai.inspector.monitoring.types import Indices, PlotAxis
from midiai.inspector.monitoring.vocab_utils import TokenFields as TF
from midiai.inspector.monitoring.vocab_utils import parse_token_field_by_id, vocab
from midiai.typevs import V


class Plotter(abc.ABC):
    """개별 subplot을 그리는 객체들의 베이스 클래스."""

    vocab: V = vocab

    def __init__(self, *args):
        ...

    @abc.abstractmethod
    def plot(self, ax: PlotAxis, x: np.ndarray, *args) -> PlotAxis:
        raise NotImplementedError


class EmbeddingWeightsPlotter(Plotter):
    """..."""

    def __init__(self):
        super().__init__()

    def plot(
        self,
        ax: PlotAxis,
        x: np.ndarray,
        subtitle: Optional[str] = None,
        **plotter_kwargs,
    ):
        ax.scatter(x[:, 0], x[:, 1], **plotter_kwargs)  # TODO: 세부 시각화
        if subtitle is not None:
            ax.set_title(subtitle)
        return ax


class TimeAxisPlotter(Plotter):
    """..."""

    field_color_map = {
        TF.PAD: "r",
        TF.BAR: "r",
        TF.NOTE_POSITION: "k",
        TF.VELOCITY: "k",
        TF.PITCH: "k",
        TF.NOTE_DURATION: "k",
        TF.CHORD_POSITION: "b",
        TF.CHORD: "b",
        TF.CHORD_BASS: "b",
        TF.CHORD_DURATION: "b",
        TF.EOS: "r",
        TF.OTHERS: "g",
    }

    def __init__(self):
        super().__init__()

    def plot(
        self,
        ax: PlotAxis,
        y: np.ndarray,
        x: Optional[list[int]] = None,
        ylabel: Optional[str] = None,
        xlabel: str = "Steps",
        linewidth: float = 0.4,
        linecolor: str = "k",
        scatter_size: float = 8.0,
        grid: bool = True,
        **plotter_kwargs,
    ) -> PlotAxis:
        ax.plot(y, linewidth=linewidth, c=linecolor, **plotter_kwargs)
        colors = (
            None
            if x is None
            else [self.field_color_map.get(parse_token_field_by_id(id)) for id in x]
        )
        if colors is not None:
            ax.scatter(range(len(y)), y, c=colors, s=scatter_size, **plotter_kwargs)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(grid, alpha=0.6)
        return ax


class VocabAxisPlotter(Plotter):
    """..."""

    def __init__(self):
        super().__init__()
        vocab = self.vocab

        sequence_word_offsets = []
        for v in vocab.sequence_word:
            if v.name != TF.BAR:
                sequence_word_offsets.append(v.value.offset)

        chord_word_offsets = []
        for v in vocab.chord_word:
            chord_word_offsets.append(v.value.offset)
        sequence_word_offsets.append(chord_word_offsets[0])

        other_offsets = []
        if meta_word_offset := self._get_first_offset("meta_word"):
            other_offsets.append(meta_word_offset)
        if guideline_word_offset := self._get_first_offset("guideline_word"):
            other_offsets.append(guideline_word_offset)
        if marker_sequence_word_offset := self._get_first_offset("marker_sequence_word"):
            other_offsets.append(marker_sequence_word_offset)
        other_offsets.append(self.vocab.vocab_size)
        chord_word_offsets.append(other_offsets[0])

        self.background_cfgs = []
        self._add_background_cfg(sequence_word_offsets, "green")
        self._add_background_cfg(chord_word_offsets, "blue")
        self._add_background_cfg(other_offsets, "gray")

    def _get_first_offset(self, attr_name: str) -> None:
        if hasattr(self.vocab, attr_name):
            attr_vocab = getattr(self.vocab, attr_name)
            for v in attr_vocab:
                return v.value.offset

    def _add_background_cfg(self, offsets: list[int], facecolor: str) -> None:
        for i in range(len(offsets) - 1):
            cfg = {
                "xmin": offsets[i],
                "xmax": offsets[i + 1],
                "facecolor": facecolor,
                "alpha": np.linspace(0.2, 0.5, len(offsets))[i],
            }
            self.background_cfgs.append(cfg)

    def _paint_background(self, ax: PlotAxis) -> PlotAxis:
        for cfg in self.background_cfgs:
            ax.axvspan(**cfg)
        return ax

    def plot(
        self,
        ax: PlotAxis,
        y: np.ndarray,
        x: Optional[Indices] = None,
        ylabel: Optional[str] = None,
        xlabel: Optional[str] = "Vocabulary",
        linewidth: float = 0.4,
        linecolor: str = "k",
        grid: bool = True,
        **plotter_kwargs,
    ) -> PlotAxis:
        if x is None:
            x = range(1, y.shape[0] + 1)
        ax.plot(x, y, linewidth=linewidth, c=linecolor, **plotter_kwargs)
        ax = self._paint_background(ax)
        ax.set_ylim([y.min(), y.max()])
        ax.set_xlim([0, y.shape[0]])
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.grid(grid, alpha=0.7)
        return ax


class BarcodeStylePlotter(Plotter):
    """
    1차원 어레이에 대해 바코드 스타일 시각화를 수행하는 Plotter.
    기본적으로 note_position, velocity, note_duration 로짓 및 확률 시각화에 사용.
    """

    vertical_size: int = 1

    def __init__(self):
        super().__init__()

    def plot(
        self,
        ax: PlotAxis,
        x: np.ndarray,
        cmap: str = "Greys",
        labelsize: int = 7,
        aspect: str = "auto",
        subtitle: Optional[str] = None,
        **plotter_kwargs,
    ) -> PlotAxis:
        ax.imshow(
            np.expand_dims(x, axis=0),
            cmap=cmap,
            aspect=aspect,
            **plotter_kwargs,
        )
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.tick_params(axis="x", which="major", labelsize=labelsize)
        if subtitle is not None:
            ax.set_title(subtitle, fontsize=8)
        return ax


class PitchMatrixPlotter(Plotter):
    """
    pitch 로짓 및 확률을 매트릭스 형태로 바꿔 시각화하는 Plotter.
    magnify: bool = True라면, 특정 threshold보다 높은 값들은 모두 +0.1을 하여 시각화에 도움을 줌
    (여기서는 0.0001(1e-4)로 설정됨).
    """

    vertical_size: int = 2.5
    NUM_GHOST_PITCHES: int = 4
    PLOT_LABEL_SIZE: int = 8

    def __init__(self):
        super().__init__()

    def _transform_pitch_to_matrix(self, value: np.ndarray) -> np.ndarray:
        """pitch 로짓 및 확률을 매트릭스 형태로 전환하는 루틴."""
        value_filtered_extended: list[int] = value.tolist()
        value_filtered_extended.extend(self.NUM_GHOST_PITCHES * [np.NaN])
        return np.flipud(np.array(value_filtered_extended).reshape((-1, NUM_PITCHES)))

    def plot(
        self,
        ax: PlotAxis,
        x: np.ndarray,
        subtitle: Optional[str] = None,
        cmap: str = "Greens",
        aspect: str = "auto",
        magnify: Optional[float] = None,
        **plotter_kwargs,
    ) -> PlotAxis:
        if magnify:  # magnify is not None or != 0
            x[x < magnify] == magnify

        pitch_matrix = self._transform_pitch_to_matrix(x)
        ax.imshow(pitch_matrix, cmap=cmap, aspect=aspect, **plotter_kwargs)
        ax.set_xticks(np.arange(0, NUM_PITCHES))
        ax.set_xticklabels([pitch for pitch in PITCH_MAP.values()])
        ax.set_yticks(np.arange(0, NUM_OCTAVES))
        ax.set_yticklabels(OCTAVES)
        ax.tick_params(axis="both", which="major", labelsize=self.PLOT_LABEL_SIZE)

        ax.set_xticks(np.arange(-0.5, NUM_PITCHES, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, NUM_OCTAVES, 1), minor=True)
        ax.tick_params(axis="both", which="minor", length=0)
        ax.grid(which="minor", color="k", linestyle="-", linewidth=0.1)

        # ax.set_xlabel("Notes", fontsize=7)
        ax.set_ylabel("Octaves", fontsize=7)
        if subtitle is not None:
            ax.set_title(subtitle, fontsize=8)
        return ax


class AttentionMapPlotter(Plotter):
    """..."""

    def __init__(self):
        super().__init__()

    def plot(
        self,
        ax: PlotAxis,
        x: np.ndarray,
        cmap: str = "plasma",
        aspect: Optional[str] = None,
        subtitle: str = "Self Attention Map",
        crop: Optional[float] = 0.2,
        **plotter_kwargs,
    ) -> PlotAxis:
        if crop:  # crop is not None
            x[x > crop] = crop
        ax.matshow(x, cmap=cmap, aspect=aspect, **plotter_kwargs)
        ax.set_xlabel("Key (sampled sequence)")
        ax.set_ylabel("Query (steps)")
        ax.set_title(subtitle)
        return ax


class HistogramPlotter(Plotter):
    """..."""

    def __init__(self):
        super().__init__()

    def plot(
        self,
        ax: PlotAxis,
        y: np.ndarray,
        labels: Optional[list[Any]] = None,
        color: str = "b",
        **plotter_kwargs,
    ) -> PlotAxis:
        ax.bar(x=labels, height=y, color=color, **plotter_kwargs)
        ax.set_ylabel("Frequency")
        ax.tick_params(axis="both", which="major", labelsize=8)
        return ax
