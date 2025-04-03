from __future__ import annotations

import abc
from typing import Callable, Optional, Union

import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import Tensor

from midiai.inspector.monitoring.numeric_routines import (
    _slice_array_2D_with_axis_labels,
    gather_target,
    stepwise_entropy,
    target_nll,
)
from midiai.inspector.monitoring.types import FeatureValues, Indices
from midiai.inspector.monitoring.utils import ExcessiveDisplayChecker
from midiai.inspector.monitoring.utils import TextColors as TC
from midiai.inspector.monitoring.visualization.visualizer import VISUALIZER as Visualizer
from midiai.inspector.monitoring.vocab_utils import (
    fill_spaces,
    parse_token_field_by_id,
    parse_token_representation_by_id,
)

EXCESSIVE_DISP_CHECKER = ExcessiveDisplayChecker()


class Features(abc.ABC):
    """
    모니터링할 피처들의 베이스클래스.
    항상 FeatureValues 타입(ndarray, 텐서)의 numeric value를 self.value에 지니고 있어야 하며,
    모든 subclass들은 .plot() 메서드로 Visualizer와 연동되는 시각화 인터페이스를 만드는 것을 권장.
    """

    name: str

    def __init__(self, value: FeatureValues, *args, **kwargs):
        self.value: Union[np.ndarray, torch.Tensor] = value

    def plot(self, *args, **kwargs):
        ...

    def __repr__(self) -> str:
        return (
            f"Feature type '{self.name}', shape of value: " f"{TC.GREEN}{self.value.shape}{TC.ENDC}"
        )


# DERIVED FEATURES


class TimeSeries(Features):
    """일반적인 1차원 어레이 데이터 클래스."""

    def __init__(self, value: Tensor, axis: Tensor, name: str):
        super().__init__(value)
        self.axis = axis
        self.name = name

    def plot(self, title: Optional[str] = None, **plotter_kwargs) -> None:
        """
        kwargs:  # TODO: 세부 플로팅 루틴 보조인자인 kwargs 정리
            ...
        """
        Visualizer.plot_time_series(
            y=self.value.cpu().numpy(),
            x=self.axis.cpu().numpy(),
            ylabel=self.name,
            xlabel="Steps",
            title=title,
            **plotter_kwargs,
        )


class BarUnionFeatureMatrix(Features):  # TODO: 2D feature로..?
    """InterpretedSequence.BarUnions로부터 파생된 pairwise matrix feature."""

    def __init__(self, value: np.ndarray, name: str):
        super().__init__(value)
        self.name = name

    def plot(self, **plotter_kwargs) -> None:
        """
        kwargs:
            ...
        """
        Visualizer.plot_matrix_2D(
            x=self.value,
            title="Grooving pattern similarity",
            **plotter_kwargs,
        )


class Histogram(Features):
    """히스토그램 클래스."""

    def __init__(self, value: np.ndarray, name: str):
        super().__init__(value)
        self.name = name

    def plot(self, **plotter_kwargs) -> None:
        """
        kwargs:
            ...
        """
        Visualizer.plot_histogram(
            y=self.value,
            title=self.name,
            **plotter_kwargs,
        )


class PianoRoll(Features):
    """피아노롤 클래스."""

    name: str = "Piano Roll"

    def __init__(self, value: np.ndarray, num_bars: int):
        super().__init__(value)
        self.num_bars = num_bars

    def plot(self, **plotter_kwargs) -> None:
        """
        kwargs:
            ...
        """
        Visualizer.plot_piano_roll(
            self.value,
            self.num_bars,
            **plotter_kwargs,
        )


# ORIGINAL FEATURES


class EmbeddingWeights(Features):
    """모델의 워드 임베딩 weights"""

    name: str = "Embedding Parameters"
    reduce_dimension: int = 2

    def __init__(self, value: Tensor):
        super().__init__(value)

    def _tsne_reduce_weights(self, n_components) -> np.ndarray:
        return TSNE(n_components=n_components).fit_transform(self.value.cpu().numpy())

    def plot(self, **plotter_kwargs) -> None:
        """
        kwargs:
            ...
        """
        reduced_weights = self._tsne_reduce_weights(self.reduce_dimension)
        Visualizer.plot_word_embedding(
            reduced_weights,
            **plotter_kwargs,
        )


class Probs(Features):
    """
    토큰이 샘플링 되기 위한 확률들의 클래스: Logits로부터 파생될 수 있음.
    row_axis, col_axis는 각각 step축, vocab죽 기준의 토큰 레이블을 뜻함.
    * Ex)
        row_axis = [2, 451]
        col_axis = [0, 1, 2, ... 1002] (vocab 축)
    이면, 현재 (2 x 1003) 사이즈로 주어진 .value는:
    1. 첫번째 row의 vocab상 확률분포로 2가 샘플링 되었고,
    2. 두번째 row의 vocab상 확률분포로 451이 샘플링 되었다는 뜻.
    """

    name: str = "Probabilities"

    def __init__(
        self,
        value: Tensor,
        row_axis: Optional[Tensor] = None,
        col_axis: Optional[Tensor] = None,
    ):
        super().__init__(value)
        self.row_axis = row_axis
        self.col_axis = col_axis

    def slice(
        self,
        row: Optional[Union[Indices, str]] = None,
        col: Optional[Union[Indices, str]] = None,
        values_only: bool = False,
    ) -> Union[Probs, FeatureValues]:
        """
        slice_2D를 사용하여 row, col dimension 슬라이싱을 하여 리턴.
        Args:
            row (Optional[Union[Indices, str]]): row-축 (샘플링 스텝 축) 토큰 id 레이블.
            col (Optional[Union[Indices, str]]): column-축 (vocab id 축) 토큰 id 레이블.
            values_only (bool, optional): False일 경우, SelfAttentionMap 형태로 wrap하여
                리턴하는 것이 아닌, .values의 슬라이싱 결과만 리턴함.
        Returns:
            Union[FeatureValues, SelfAttentionMap]: 슬라이싱 결과.
        """
        values_slice, row_axis_sliced, col_axis_sliced = _slice_array_2D_with_axis_labels(
            arr=self.value,
            row_axis=self.row_axis,
            col_axis=self.col_axis,
            row_slice=row,
            col_slice=col,
        )
        if values_only:
            return values_slice
        return Probs(
            values_slice,
            row_axis=row_axis_sliced,
            col_axis=col_axis_sliced,
        )

    def negative_log(self) -> Tensor:
        """self.value의 negative log값을 리턴 (0일 경우 해당 인덱스는 inf가 됨)"""
        return -torch.log(self.value)

    def target_probabilities(self, values_only: bool = False) -> Union[FeatureValues, TimeSeries]:
        """
        self.value[i, self.row_axis[i]]를 모든 i에 대해 쭉 모은 것과 동일.
        즉, row_axis 방향으로, row_axis[i] 토큰이 뽑힌 확률값.
        """
        target_probs = gather_target(value=self.value, target=self.row_axis)
        if values_only:
            return target_probs
        return TimeSeries(
            value=target_probs,
            axis=self.row_axis,
            name="Target probability",
        )

    def target_nll(self, values_only: bool = False) -> Union[FeatureValues, TimeSeries]:
        """self.target_probabilities의 negative log값."""
        nll = target_nll(probs=self.value, target=self.row_axis)
        if values_only:
            return nll
        return TimeSeries(
            value=nll,
            axis=self.row_axis,
            name="Negative log prob (nll)",
        )

    def stepwise_entropy(self, values_only: bool = False) -> Union[FeatureValues, TimeSeries]:
        """self.value[i, :]의 확률값들의 엔트로피를 i에 대해 순차적으로 계산"""
        entropy = stepwise_entropy(probs=self.value)
        if values_only:
            return entropy
        return TimeSeries(
            value=entropy,
            axis=self.row_axis,
            name="Sampling Entropy",
        )

    def disp(self, threshold: float = 1e-4) -> None:
        """
        self.value에 row 별로 수록된 확률들을 col 내에서 높은 순서대로 나열.
        disp_threshold 보다 낮은 확률값은 표시되지 않음.
        row가 너무 많을 경우, 표시를 진행할 것인지를 물음.
        """
        if len(self.value.shape) > 1:
            res = EXCESSIVE_DISP_CHECKER._check(self.value.shape[0])
            if res:
                return
            all_probs = self.value
        else:
            all_probs = [self.value]

        for i, probs in enumerate(all_probs):
            print(f"{TC.BLUEBG}row {i}:{TC.ENDC}")
            probs = probs.cpu().numpy()
            probs_dict = {id: prob for id, prob in enumerate(probs)}
            probs_dict = sorted(
                probs_dict.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            p_sum = 0.0
            for i, (id, prob) in enumerate(probs_dict):
                if prob < threshold:
                    break
                field = fill_spaces(f"<{parse_token_field_by_id(id)}>")
                repr = parse_token_representation_by_id(id)
                p_sum += prob
                print(
                    f"{TC.BOLD}{TC.GREEN}{i}{TC.ENDC}\t: "
                    f"{field} {repr} (id: {id})\t| {prob:.4f}\t| p_sum: {p_sum:.6f}"
                )
            print("")

    def plot(
        self,
        field: Optional[str] = None,
        **plotter_kwargs,
    ) -> None:
        """
        kwargs:
            ...
        """
        Visualizer.plot_logitprobs_single_field(
            xs=self.value.cpu().numpy(), field=field, types="Probs", **plotter_kwargs
        )


class Logits(Features):
    """
    토큰 확률을 내는 로짓 클래스.
    compute_probs() 메서드로 Probs 클래스를 파생 가능.
    """

    name: str = "Logits"

    def __init__(
        self,
        value: Tensor,
        row_axis: Optional[Tensor] = None,
        col_axis: Optional[Tensor] = None,
    ):
        super().__init__(value)
        self.row_axis = row_axis
        self.col_axis = col_axis if col_axis is not None else range(self.value.shape[1])

    def slice(
        self,
        row: Optional[Union[Indices, str]] = None,
        col: Optional[Union[Indices, str]] = None,
        values_only: bool = False,
    ) -> Union[FeatureValues, Logits]:
        """
        self.values를 트정 조건으로 슬라이싱하여 볼 수 있게 해주는 메서드.
        위의 slice와 동일.
        """
        values_slice, row_axis_sliced, col_axis_sliced = _slice_array_2D_with_axis_labels(
            arr=self.value,
            row_axis=self.row_axis,
            col_axis=self.col_axis,
            row_slice=row,
            col_slice=col,
        )
        if values_only:
            return values_slice
        return Logits(
            values_slice,
            row_axis=row_axis_sliced,
            col_axis=col_axis_sliced,
        )

    def compute_probs(
        self,
        f: Callable,
        temperature: float,
        values_only: bool = False,
    ) -> Union[FeatureValues, Probs]:
        """self.logits에 prob_func 함수(softmax 종류)를 적용한 결과를 리턴"""
        probs_value: Tensor = f(self.value / temperature)
        if values_only:
            return probs_value
        return Probs(
            value=probs_value,
            row_axis=self.row_axis,
        )

    def disp(self, num: float = 20) -> None:
        """
        self.value의 각 행에 대해 수록된 값들을 열 내 높은 순서대로 num 개만큼 나열.
        """
        if len(self.value.shape) > 1:
            res = EXCESSIVE_DISP_CHECKER._check(self.value.shape[0])
            if res:
                return
            all_logits = self.value
        else:
            all_logits = [self.value]

        for i, logits in enumerate(all_logits):
            print(f"{TC.BLUEBG}row {i}:{TC.ENDC}")
            logits = logits.cpu().numpy()
            logits_dict = {id: logit for id, logit in enumerate(logits)}
            logits_dict = sorted(
                logits_dict.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for i, (id, logit) in enumerate(logits_dict):
                if i == num:
                    break
                field = fill_spaces(f"<{parse_token_field_by_id(id)}>")
                repr = parse_token_representation_by_id(id)
                print(
                    f"{TC.BOLD}{TC.GREEN}{i}{TC.ENDC}\t: "
                    f"{field} {repr} (id: {id})\t| {logit:.4f}\t"
                )

    def plot(self, field: Optional[str] = None, title: Optional[str] = None) -> None:
        """
        kwargs:
            ...
        """
        Visualizer.plot_logitprobs_single_field(
            xs=self.value.cpu().numpy(),
            field=field,
            types="Logits",
            title=title,
        )


class SelfAttentionMap(Features):
    """디코더 어텐션 스코어를 담는 클래스."""

    name: str = "Self Attention Map"

    def __init__(
        self,
        value: Tensor,
        queries: Optional[Tensor] = None,  # query-axis labels
        keys: Optional[Tensor] = None,  # key-axis labels
    ):
        super().__init__(value)
        self.queries = queries
        self.keys = keys

    def slice(
        self,
        layer: Optional[int] = None,
        head: Optional[int] = None,
        query: Optional[Union[Indices, str]] = None,
        key: Optional[Union[Indices, str]] = None,
        values_only: bool = False,
    ) -> Union[FeatureValues, SelfAttentionMap]:
        """
        layer, head dimension으로 slicing 혹은 summation을 진행한 후,
        slice_2D를 사용하여 query, key dimension 슬라이싱을 하여 리턴.
        Args:
            layer (Optional[int]): 어텐션 맵이 파생된 모델 레이어 번호.
                None일 경우 layer 축방향으로 summation.
            head (Optional[int]): 어텐션 맵이 파생된 layer의 모델 어텐션 헤드 번호.
                None 일 경우 head 축방향으로 summation.
            ids (Optional[Union[Indices, str]]): self-attn 진행한 토큰들의 id 레이블.
            values_only (bool, optional): False일 경우, SelfAttentionMap 형태로 wrap하여
                리턴하는 것이 아닌, .values의 슬라이싱 결과만 리턴함.
        Returns:
            Union[FeatureValues, SelfAttentionMap]: 슬라이싱 결과.
        """
        value = self.value
        if layer is None:  # sums layer dim
            value = value.sum(dim=0)
        else:  # extracts provided index
            value = value[layer]
        if head is None:  # sums multi-head dim
            value = value.sum(dim=0)
        else:  # extracts provided index
            value = value[head]

        values_slice, queries_sliced, keys_sliced = _slice_array_2D_with_axis_labels(
            arr=value,
            row_axis=self.queries,
            col_axis=self.keys,
            row_slice=query,
            col_slice=key,
        )
        if values_only:
            return value
        return SelfAttentionMap(
            value=values_slice,
            queries=queries_sliced,
            keys=keys_sliced,
        )

    def stepwise_entropy(self) -> float:
        ...  # TODO: 세부 구상

    def plot(self, **plotter_kwargs) -> None:
        """
        kwargs:
            ...
        """
        if len(self.value.shape) > 2:
            raise ValueError(
                f"Please use {TC.WARN}.slice(){TC.ENDC} to reduce dimension. "
                f"ex) attention_map{TC.WARN}.slice(){TC.ENDC}.plot()"
            )
        Visualizer.plot_attention_map(
            xs=self.value.cpu().numpy(),
            axis=self.keys,  # key dimension
            **plotter_kwargs,
        )
