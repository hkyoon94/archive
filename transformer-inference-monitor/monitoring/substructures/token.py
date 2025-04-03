from functools import cached_property
from math import log
from typing import Optional

import torch
from torch import Tensor

from midiai.inspector.monitoring import vocab_utils
from midiai.inspector.monitoring.features import Logits, Probs, SelfAttentionMap
from midiai.inspector.monitoring.vocab_utils import (
    fill_spaces,
    parse_token_field_by_id,
    parse_token_representation_by_id,
)


class TokenHistory:
    """
    토큰 객체
    self.field에 토큰의 종류를 저장(SequentialDetector가 파싱에 사용)
    """

    field: str

    def __init__(
        self,
        id: int,
        index: Optional[int] = None,
        logits: Optional[Tensor] = None,
        probs: Optional[Tensor] = None,
        self_attention_map: Optional[Tensor] = None,
    ):
        self.id = id
        self.index = index

        # original feature-value references
        self._logits: Tensor = logits
        self._probs: Tensor = probs
        self._self_attention_map: Tensor = self_attention_map
        self._device: torch.device = self._logits.device

        # auxiliary causal shift index for Logits and Probs class
        if index > 0:
            self._index = index - 1
        else:  # first token of the sequence, no logits driven
            self._index = None

    @cached_property
    def field(self) -> str:
        return parse_token_field_by_id(self.id)

    @cached_property
    def logits(self) -> Logits:
        if self._index is None:
            return
        return Logits(
            value=self._logits[self._index],
            row_axis=torch.tensor([self.id]).to(self._device),
            col_axis=vocab_utils.TOKEN_IDS,
        )

    @cached_property
    def probs(self) -> Probs:
        if self._index is None:
            return
        return Probs(
            value=self._probs[self._index],
            row_axis=torch.tensor([self.id]).to(self._device),
            col_axis=vocab_utils.TOKEN_IDS,
        )

    @cached_property
    def self_attention_map(self) -> SelfAttentionMap:
        if self._index is None:
            return
        return SelfAttentionMap(
            value=self._self_attention_map[:, :, self._index],
            queries=torch.tensor([self.id]).to(self._device),
        )

    @cached_property
    def sampled_probability(self) -> float:
        """해당 토큰이 샘플링 되었던 확률값"""
        if self._index is not None:
            return self._probs[self._index, self.id].item()

    @cached_property
    def nll(self) -> Tensor:
        """해당 토큰이 샘플링 되었던 확률값의 negative log"""
        prob = self.sampled_probability
        if prob is None:
            return
        return -log(prob)

    @cached_property
    def _repr(self) -> str:
        return parse_token_representation_by_id(self.id)

    def __repr__(self) -> str:
        field_str = fill_spaces(f"{self.field}")
        return f"{field_str}: {self._repr} (id: {self.id})"
