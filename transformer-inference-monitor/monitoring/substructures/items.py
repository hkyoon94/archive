from functools import cached_property
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor

from midiai.inspector.monitoring import vocab_utils
from midiai.inspector.monitoring.features import Logits, Probs, SelfAttentionMap
from midiai.inspector.monitoring.numeric_routines import (
    compute_chord_chroma,
    compute_tonal_centroid,
)
from midiai.inspector.monitoring.substructures.token import TokenHistory
from midiai.inspector.monitoring.types import Indices
from midiai.inspector.monitoring.utils import TextColors as TC
from midiai.inspector.monitoring.visualization.visualizer import VISUALIZER as Visualizer
from midiai.inspector.monitoring.vocab_utils import ItemFields as IF
from midiai.inspector.monitoring.vocab_utils import fill_spaces


class Item:
    """
    개별 TokenHistory가 모여 하나의 의미 단위를 이루는 객체들의 베이스 클래스.
    글로벌 str 속성인 field는 "bar", "chord", "note"등이 가능.
    """

    field: str

    def __init__(
        self,
        tokens: list[TokenHistory],
        count: Optional[int] = None,
    ):
        self.tokens = tokens
        self.count = count

        # original feature-value references
        self._logits: Tensor = self.tokens[0]._logits
        self._probs: Tensor = self.tokens[0]._probs
        self._self_attention_map: Tensor = self.tokens[0]._self_attention_map
        self._device: torch.device = self._logits.device

        self.ids: Indices = torch.tensor(self._get_token_attributes("id")).to(self._device)
        self.indices: Indices = torch.tensor(self._get_token_attributes("index")).to(self._device)

        # auxiliary causal shift index for Logits and Probs class
        self._indices = self.indices - 1

    def _get_token_attributes(self, attr: str) -> Union[Indices, list[int]]:
        attrs = []
        for token in self.tokens:
            value = getattr(token, attr)
            if value is not None:
                attrs.append(value)
        return attrs

    @cached_property
    def fields(self) -> list[str]:
        return self._get_token_attributes("field")

    # EXAMINING ORIGINAL FEATURES

    @cached_property
    def logits(self) -> Logits:
        return Logits(
            value=self._logits[self._indices],
            row_axis=self.ids,
            col_axis=vocab_utils.TOKEN_IDS,
        )

    @cached_property
    def probs(self) -> Probs:
        return Probs(
            value=self._probs[self._indices],
            row_axis=self.ids,
            col_axis=vocab_utils.TOKEN_IDS,
        )

    @cached_property
    def self_attention_map(self) -> SelfAttentionMap:
        return SelfAttentionMap(
            value=self._self_attention_map[:, :, self.indices],
            queries=self.ids,
            # keys=...
            # TODO: 이 상황에서 원래 sequence의 정보를 끌어오는 방법을 고민
            # self._index 처럼, self._sequence 참조 변수를 사용할지..?
        )

    def plot_logits(self, field_only: bool = True) -> None:
        """해당 Item이 생성되는 데 사용된 logit들을 플롯"""
        print(f"Selected item:| {str(self)}")
        if field_only:
            fields = [token.field for token in self.tokens]
        else:
            fields = None
        Visualizer.plot_logitprobs_multiple_fields(
            xs=self.logits.value.cpu().numpy(),
            fields=fields,
            title="Note Logits",
        )

    def plot_probs(self, field_only: bool = True) -> None:
        """해당 Item이 생성되는 데 사용된 probability들을 플롯"""
        print(f"Selected item:| {str(self)}")
        if field_only:
            fields = [token.field for token in self.tokens]
        else:
            fields = None
        Visualizer.plot_logitprobs_multiple_fields(
            xs=self.probs.value.cpu().numpy(),
            fields=fields,
            title="Note Probabilities",
        )


class Bar(Item):
    """Bar(마디) 객체. self.count는 몇 번째 마디인지를 나타냄."""

    field = IF.BAR

    def __init__(
        self,
        tokens: list[TokenHistory],
        count: Optional[int] = None,
    ):
        super().__init__(tokens, count)

    def __repr__(self) -> str:  # TODO: color format 적용
        return f"{TC.REDBG}{self.field.upper()}_{self.count}{TC.ENDC}: {self.ids}"


class Chord(Item):
    """
    하나의 코드를 나타내는 객체.
    chord_position, chord, chord_bass, chord_duration 토큰이 모여 구성됨.
    """

    field = IF.CHORD

    def __init__(
        self,
        tokens: list[TokenHistory],
        count: Optional[int] = None,
    ):
        super().__init__(tokens, count)
        self.position, self.chord, self.bass, self.tension, self.duration = self.tokens
        self.chord_name = self.chord._repr.split(" ")[1]
        self.bass_name = self.bass._repr.split(" ")[1]
        self.tension_name = self.tension._repr.split(" ")[1]
        self.tension_name = f"({self.tension_name})" if self.tension_name else ""
        self._name: str = f"{self.chord_name}{self.tension_name}/{self.bass_name}"

    def set_chord_name(self, name: str) -> None:
        self._name = name

    @cached_property
    def name(self) -> str:
        return self._name

    @cached_property
    def chroma(self) -> np.ndarray:
        return compute_chord_chroma(chord_id=self.chord.id)

    @cached_property
    def tonal_centroid(self) -> np.ndarray:
        return compute_tonal_centroid(self.chroma)

    def __repr__(self) -> str:
        field_repr = fill_spaces(f"{self.field.upper()}_{self.count}:")
        pos_repr = self.position._repr
        vel_repr = "|       -       "
        pch_repr = fill_spaces(f"{self._name}")
        dur_repr = self.duration._repr
        return (
            f"{TC.BLUE}{field_repr + pos_repr + vel_repr + pch_repr + dur_repr}"
            f": {self.ids}{TC.ENDC}"
        )


class Note(Item):
    """
    하나의 노트를 나타내는 객체.
    note_position, velocity, pitch, note_duration 토큰이 모여 구성됨.
    """

    field = IF.NOTE

    def __init__(
        self,
        tokens: list[TokenHistory],
        count: Optional[int] = None,
    ):
        super().__init__(tokens, count)
        self.field = IF.NOTE
        if len(tokens) == 4:
            self.position, self.velocity, self.pitch, self.duration = tokens
        else:
            self.position, self.pitch, self.duration = tokens
            self.velocity = None

    def __repr__(self) -> str:
        field_repr = fill_spaces(f"{self.field.upper()}_{self.count}:")
        pos_repr = self.position._repr
        vel_repr = self.velocity._repr if self.velocity is not None else ""
        pch_repr = self.pitch._repr
        dur_repr = self.duration._repr
        return f"{field_repr + pos_repr + vel_repr + pch_repr + dur_repr}: {self.ids}"


class Eos(Item):
    """sequence의 끝을 나타내는 EOS 객체"""

    field = IF.EOS

    def __init__(
        self,
        tokens: list[TokenHistory],
        count: Optional[int] = None,
    ):
        super().__init__(tokens, count)

    def __repr__(self) -> str:
        return f"{TC.REDBG}{self.field.upper()}{TC.ENDC}: {self.ids}"
