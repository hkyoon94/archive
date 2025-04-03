from typing import Optional

import numpy as np
import torch
from numpy import cos, pi, sin
from numpy.linalg import norm
from torch import Tensor

from midiai.inspector.monitoring.constants import (
    CHORD_NOTE_MAP,
    CHORD_QUALITY_MAP,
    DEFAULT_SOFTMAX_TEMP,
    DEFAULT_TOP_K,
    NUM_CHORD_QUALITY,
    NUM_PITCHES,
)
from midiai.inspector.monitoring.types import FeatureValues, Indices
from midiai.inspector.monitoring.vocab_utils import CHORD, extract_field_index


def _slice_array_2D_with_axis_labels(
    arr: FeatureValues,
    row_axis: Tensor,
    col_axis: Tensor,
    row_slice: Optional[Indices] = None,
    col_slice: Optional[Indices] = None,
) -> tuple[FeatureValues, Indices, Indices]:
    """
    어레이의 값 및 축 라벨을 커스텀 슬라이싱하여 리턴하는 함수.
    Args:
        arr (FeatureValues): 슬라이싱 하고자 하는 어레이의 값.
        row_axis Tensor: 해당 어레이의 row 축 레이블.
        col_axis Tensor: 해당 어레이의 col 축 레이블.
        row_slice (Optional[Indices]): row 방향 슬라이스.
        col_slice (Optional[Indices]): col 방향 슬라이스.
    Returns:
        tuple[FeatureValues, Indices, Indices]: arr, row_axis, col_axis의 슬라이싱 결과
    """
    if row_slice is not None and isinstance(row_slice, str):  # if provided TF string
        row_slice = extract_field_index(ids=row_axis, field=row_slice)

    if col_slice is not None and isinstance(col_slice, str):  # if provided TF string
        col_slice = extract_field_index(ids=col_axis, field=col_slice)

    # TODO: 로직 개선 및 최적화
    if row_slice is not None and col_slice is not None:
        arr = arr[row_slice]
        try:
            arr = arr[:, col_slice]
        except IndexError:
            arr = arr[col_slice]
    elif row_slice is not None and col_slice is None:
        arr = arr[row_slice]
    elif row_slice is None and col_slice is not None:
        try:
            arr = arr[:, col_slice]
        except IndexError:
            arr = arr[col_slice]

    row_axis_sliced = row_axis[row_slice] if row_slice is not None else row_axis
    col_axis_sliced = col_axis[col_slice] if col_slice is not None else col_axis

    return arr, row_axis_sliced, col_axis_sliced


def temperatured_softmax(logits: Tensor, temperature: float = DEFAULT_SOFTMAX_TEMP) -> Tensor:
    """logits에 temperature를 나눠 softmax를 진행."""
    return torch.softmax(logits / temperature, dim=-1)


def top_k(probs: Tensor, k: int = DEFAULT_TOP_K) -> Tensor:
    """
    top_k 샘플링 기법.
    Args:
        probs (Tensor): (n x m) 확률값 텐서.
        k: top_k의 k.
    Returns:
        Tensor: dim=-1 당 top_k 적용된 후의 probs.
    """
    _, idx = torch.topk(probs, k=k, dim=-1)
    mask = torch.zeros_like(probs)
    for i, ind in enumerate(idx):
        mask[i].index_fill_(0, ind, 1)
    probs = mask * probs
    for i, prob in enumerate(probs):
        probs[i] = prob / prob.sum()

    return probs


def top_p(probs: Tensor, p: int = ...) -> Tensor:
    """
    top_p 샘플링 기법.
    Args:
        probs (Tensor): (n x m) 확률값 텐서.
        p (int): top_p의 p.
    Returns:
        Tensor: dim=-1 당 top_p 적용된 후의 probs.
    """
    ...
    return


def cosine_similarity(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    두 벡터 간의 코사인 유사도
    Args:
        x1 (np.ndarray): 벡터 1
        x2 (np.ndarray): 벡터 2
    Returns:
        float: 코사인 유사도
    """
    return np.dot(x1, x2) / (norm(x1) * norm(x2))


def gather_target(value: Tensor, target: Tensor) -> Tensor:
    """
    Args:
        value (Tensor): (n x m) 2D-어레이.
        target (Tensor): n 길이의 1D-어레이.
    Returns:
        Tensor: for i in range(n): value[i, target[i]] 를 모은 것과 동일.
    """
    target = target.unsqueeze(dim=-1)
    try:
        value = value.gather(dim=-1, index=target)
    except Exception:  # value tensor is 1-dimensional
        value = value[target[0]]
    return value.squeeze(dim=-1)


def target_nll(probs: Tensor, target: Tensor) -> Tensor:
    """
    probs 텐서에 대해 target index의 값들을 모은 후 각각의 negative log값을 리턴.
    Args:
        probs (Tensor): (n x m) 확률값 텐서.
        target (Tensor): (n) 길이의 1D-어레이.
    Returns:
        Tensor: (n) 길이의 1D-어레이. 이것의 mean은, loss와 동일함.
    """
    target_probs = gather_target(probs, target)
    nll = -torch.log(target_probs)
    return nll


def stepwise_entropy(probs: Tensor) -> Tensor:
    """
    probs 텐서의 각 row당 엔트로피를 리턴.
    Args:
        probs (Tensor): (n x m) 확률값 텐서.
    Returns:
        Tensor: (n) 길이의 엔트로피 어레이.
    """
    log_probs = torch.log(probs)
    log_probs[log_probs == -torch.inf] = 0.0  # 확률값이 0인 인덱스가 있을 경우 로그값을 0으로 대체
    try:
        entropy = torch.einsum(
            "ij, ij -> i",  # j dimension summation
            (probs, -log_probs),
        )
    except Exception:  # probs tensor is 1-dimensional
        entropy = torch.dot(probs, -log_probs).item()
    return entropy


def compute_chord_chroma(chord_id: int) -> np.ndarray:
    """
    코드 토큰을 받아 해당 코드의 chord_chorma 벡터를 반환
    (논문 "Harmonic Change Detection for musical chords segmentation" 참고)
    Args:
        chord_id (int): _description_
    Returns:
        np.ndarray: _description_
    """
    chord_id_minus_offset = chord_id - CHORD.offset
    chord_pitch, chord_quality = divmod(chord_id_minus_offset, NUM_CHORD_QUALITY)
    chord_quality = CHORD_QUALITY_MAP.get(chord_quality)
    chord_notes = np.array(CHORD_NOTE_MAP.get(chord_quality))
    chord_notes += chord_pitch
    _, note_indicator = divmod(chord_notes, NUM_PITCHES)
    chorma = np.zeros(NUM_PITCHES)

    for i in note_indicator:
        chorma[i] = 1
    return chorma


def compute_tonal_centroid(chroma: np.ndarray) -> np.ndarray:
    """
    코드의 chroma 벡터를 받아 해당 코드의 tonal centroid를 반환.
    (논문 "Harmonic Change Detection for musical chords segmentation" 참고)
    Args:
        chroma (np.ndarray): _description_
    Returns:
        np.ndarray: _description_
    """
    r_1, r_2, r_3 = 1, 1, 0.5
    Phi = np.zeros((6, 12))
    for p in range(Phi.shape[1]):
        Phi[0, p] = r_1 * sin(p * (7 / 6) * pi)
        Phi[1, p] = r_1 * cos(p * (7 / 6) * pi)
        Phi[2, p] = r_2 * sin(p * (3 / 2) * pi)
        Phi[3, p] = r_2 * cos(p * (3 / 2) * pi)
        Phi[4, p] = r_3 * sin(p * (2 / 3) * pi)
        Phi[5, p] = r_3 * cos(p * (2 / 3) * pi)

    raw_centroid = Phi @ chroma
    normalizer = np.linalg.norm(chroma, ord=2)
    return raw_centroid / normalizer


def compute_tonal_distance(chord_id_1: int, chord_id_2: int, ord: int = 2) -> float:
    """
    한 쌍의 tonal centroid 사이의 L-ord Euclidean 거리를 반환.
    Args:
        chord_id_1 (int): 코드 vocab 토큰 id 1
        chord_id_2 (int): 코드 vocab 토큰 id 2
        ord (int, optional): 크로마 사이의 거리를 계산할 Euclidean norm의 order
    """
    chroma_1 = compute_chord_chroma(chord_id_1)
    chroma_2 = compute_chord_chroma(chord_id_2)
    centroid_1 = compute_tonal_centroid(chroma_1)
    centroid_2 = compute_tonal_centroid(chroma_2)
    distance = np.linalg.norm(centroid_1 - centroid_2, ord=ord)

    return distance
