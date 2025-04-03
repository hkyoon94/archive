from functools import cached_property
from typing import Optional, Union

import numpy as np
import torch
from IPython.display import Audio
from torch import Tensor

from midiai.inspector.monitoring import vocab_utils
from midiai.inspector.monitoring.constants import NUM_PITCHES
from midiai.inspector.monitoring.features import Histogram, Logits, Probs, SelfAttentionMap
from midiai.inspector.monitoring.substructures.items import Bar, Chord, Item, Note
from midiai.inspector.monitoring.types import FeatureValues, Indices
from midiai.inspector.monitoring.vocab_utils import NOTE_POSITION, PITCH


class ItemUnion:
    """
    개별 TokenHistory가 모여 하나의 의미 단위를 이루는 객체들의 베이스 클래스.
    글로벌 str 속성인 field는 "bar", "chord" 등이 가능.
    self.lead에는 각 ItemUnion 인스턴스들의 근간이 되는 Bar 및 Chord 객체가 매핑됨.
    """

    field: type
    lead: Item

    def __init__(self, items: list[Item], count: Optional[int] = None):
        self.items = items
        self.count = count
        self.lead, self.notes = self._parse_items()

        # original feature-value references
        self._logits: Tensor = self.lead._logits
        self._probs: Tensor = self.lead._probs
        self._self_attention_map: Tensor = self.lead._self_attention_map
        self._device: torch.device = self.lead._device

        self.ids: Indices = torch.tensor(self._get_token_attributes("id")).to(self._device)
        self.indices: Indices = torch.tensor(self._get_token_attributes("index")).to(self._device)

        # auxiliary causal indices for original features
        self._indices = self.indices - 1

    def _parse_items(self) -> tuple[Item, list[Note]]:
        notes = []
        for item in self.items:
            if isinstance(item, self.field):
                lead = item
            elif isinstance(item, Note):
                notes.append(item)
        return lead, notes

    def _get_token_attributes(self, attr: str) -> Union[Indices, list[int]]:
        _attrs = []
        for item in self.items:
            attrs = item._get_token_attributes(attr)
            _attrs.extend(attrs)
        return _attrs

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
            queries=self.indices,
        )

    def plot_logits(self, field_only: bool = True) -> None:
        for item in self.items:
            print(f"Item {item.__repr__()} logits:")
            item.plot_logits(field_only=field_only)

    def plot_probs(self, field_only: bool = True) -> None:
        for item in self.items:
            print(f"Item {item.__repr__()} probabilities:")
            item.plot_probs(field_only=field_only)

    def __repr__(self) -> str:
        formatted = []
        for item in self.items:
            formatted.append(f"{item.__repr__()}")
        return "\n".join(formatted) + "\n"

    def render(self) -> Audio:
        ...


class BarUnion(ItemUnion):
    """한 마디(Bar) 내의 Item 객체들의 집합"""

    field = Bar
    lead: Bar

    def __init__(
        self,
        items: list[Item],
        count: Optional[int] = None,
    ):
        super().__init__(items, count)

    # EXTRACTING ITEMUNION-LEVEL FEATURES

    def onsets(
        self, resolution: int = 48, values_only: bool = False
    ) -> Union[FeatureValues, Histogram]:
        onset_bins = np.linspace(0, 1, resolution, endpoint=False)
        onsets_histogram = np.zeros(resolution)
        position_ids = np.array([note.position.id for note in self.notes])
        rel_positions = (position_ids - NOTE_POSITION.offset) / float(NOTE_POSITION.vocab_size)

        for rel_pos in rel_positions:
            onset = np.argmin(np.abs(onset_bins - rel_pos))
            onsets_histogram[onset] += 1

        if values_only:
            return onsets_histogram
        return Histogram(
            onsets_histogram,
            name="Onset Histogram",
        )


class HarmonicUnion(ItemUnion):
    """단일 코드가 지속되는 동안 나오는 Item 객체들의 집합"""

    field = Chord
    lead: Chord

    def __init__(self, items: list[Item], count: Optional[int] = None):
        super().__init__(items, count)

    def set_chord_name(self, name: str) -> None:
        self.lead._name = name

    @cached_property
    def chroma(self) -> np.ndarray:
        return self.lead.chroma

    # EXTRACTING ITEMUNION-LEVEL FEATURES

    def get_pitch_class_histogram(
        self, values_only: bool = False
    ) -> Union[FeatureValues, Histogram]:
        pitch_ids = [note.pitch.id for note in self.notes]

        pitch_class_histogram = np.zeros(NUM_PITCHES)
        for pitch_id in pitch_ids:
            pitch_class = (pitch_id - PITCH.offset) % NUM_PITCHES
            pitch_class_histogram[pitch_class] += 1

        if values_only:
            return pitch_class_histogram
        return Histogram(
            pitch_class_histogram,
            name="Pitch Class Histogram",
        )
