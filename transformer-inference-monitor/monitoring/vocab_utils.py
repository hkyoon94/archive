from enum import Enum

import torch

from midiai.inspector.monitoring.constants import (
    CHORD_QUALITY_MAP,
    CHORD_TENSION_MAP,
    NOTE_FIELDS,
    NUM_CHORD_QUALITY,
    NUM_PITCHES,
    PITCH_MAP,
    vocab,
)
from midiai.inspector.monitoring.types import Indices

"""
프로젝트 전역의 VOCAB은, 항상
    1. monitoring_config.yml에 사용자에 의해 지정된 후,
    2. constants.py 로부터 intialize 되어 여기로 import 된 뒤,
    3. 아래의 프로젝트 전역 변수들을 파생하여 정의함.
    # TODO: VOCAB과 같은 동적 전역변수 활용 테크닉 좀 더 알아보기..
"""


class TokenFields(str, Enum):
    """개별 토큰의 필드 범주."""

    PAD = "PAD"
    EOS = "EOS"
    BAR = "BAR"
    NOTE_POSITION = "NOTE_POSITION"
    VELOCITY = "VELOCITY"
    PITCH = "PITCH"
    NOTE_DURATION = "NOTE_DURATION"
    CHORD_POSITION = "CHORD_POSITION"
    CHORD = "CHORD"
    CHORD_BASS = "CHORD_BASS"
    CHORD_TENSION = "CHORD_TENSION"
    CHORD_DURATION = "CHORD_DURATION"
    NOTES = "NOTES"
    CHORDS = "CHORDS"
    OTHERS = "OTHERS"


class ItemFields(str, Enum):
    """토큰이 모인 Item 객체의 필드 범주."""

    EOS = "EOS"
    BAR = "BAR"
    NOTE = "NOTE"
    CHORD = "CHORD"
    INVALID = "INVALID"


class ItemUnionFields(str, Enum):
    """Item 객체가 모인 ItemUnion 객체의 필드 범주."""

    BAR = "BAR"
    HARMONIC = "HARMONIC"


# vocab value references
VOCAB_SIZE = vocab.vocab_size

NOTE_POSITION = vocab.sequence_word.NOTE_POSITION.value
if hasattr(vocab.sequence_word, TokenFields.VELOCITY.value):
    VELOCITY = vocab.sequence_word.VELOCITY.value
else:
    VELOCITY = None
PITCH = vocab.sequence_word.PITCH.value
NOTE_DURATION = vocab.sequence_word.NOTE_DURATION.value

CHORD_POSITION = vocab.chord_word.CHORD_POSITION.value
CHORD = vocab.chord_word.CHORD.value
if hasattr(vocab.chord_word, TokenFields.CHORD_BASS.value):
    CHORD_BASS = vocab.chord_word.CHORD_BASS.value
else:
    CHORD_BASS = None
if hasattr(vocab.chord_word, TokenFields.CHORD_TENSION.value):
    CHORD_TENSION = vocab.chord_word.CHORD_TENSION.value
else:
    CHORD_TENSION = None
CHORD_DURATION = vocab.chord_word.CHORD_DURATION.value


# vocab range references
PAD = vocab.special_token.PAD.value.offset
EOS = vocab.special_token.EOS.value.offset
BAR = vocab.sequence_word.BAR.value.offset

POS_RANGE = NOTE_POSITION.vocab_range
if VELOCITY is not None:
    VEL_RANGE = VELOCITY.vocab_range
else:
    VEL_RANGE = []
PITCH_RANGE = PITCH.vocab_range
DUR_RANGE = NOTE_DURATION.vocab_range

CHORD_POS_RANGE = CHORD_POSITION.vocab_range
CHORD_RANGE = CHORD.vocab_range
CHORD_BASS_RANGE = CHORD_BASS.vocab_range
CHORD_TENSION_RANGE = CHORD_TENSION.vocab_range
CHORD_DUR_RANGE = CHORD_DURATION.vocab_range


# 하나의 chord 단위를 이루는 components
# (NOTE_FIELDS와는 다르게 항상 vocab에서 파생)
CHORD_FIELDS = tuple([v.name for v in vocab.chord_word])

note_vocab_range = []
for field in NOTE_FIELDS:
    try:
        field_range = getattr(vocab.sequence_word, field).value.vocab_range
    except AttributeError:
        continue
    note_vocab_range += list(field_range)
note_vocab_range.insert(0, BAR)

chord_vocab_range = []
for field in CHORD_FIELDS:
    field_range = getattr(vocab.chord_word, field).value.vocab_range
    chord_vocab_range += list(field_range)


# 메타, 가이드라인, 피처 등 conditional token을 이루는 vocab range
total_vocab_range = set(range(VOCAB_SIZE))
conditional_vocab_range = list(
    total_vocab_range.difference(set(note_vocab_range))
    .difference(set(chord_vocab_range))
    .difference(set([PAD, BAR, EOS]))
)

TOKEN_IDS = torch.tensor(range(0, VOCAB_SIZE))

# Features를 플롯할 때 사용할 vocab range dict
FIELD_VOCAB_RANGE: dict[str, Indices] = {
    TokenFields.PAD: torch.tensor([PAD]),  # TODO: device가 정해진 경우 push하는 법..?
    TokenFields.EOS: torch.tensor([EOS]),
    TokenFields.BAR: torch.tensor([BAR]),
    TokenFields.NOTE_POSITION: torch.tensor(POS_RANGE),
    TokenFields.PITCH: torch.tensor(PITCH_RANGE),
    TokenFields.NOTE_DURATION: torch.tensor(DUR_RANGE),
    TokenFields.CHORD_POSITION: torch.tensor(CHORD_POS_RANGE),
    TokenFields.CHORD: torch.tensor(CHORD_RANGE),
    TokenFields.CHORD_BASS: torch.tensor(CHORD_BASS_RANGE),
    TokenFields.CHORD_TENSION: torch.tensor(CHORD_TENSION_RANGE),
    TokenFields.CHORD_DURATION: torch.tensor(CHORD_DUR_RANGE),
    TokenFields.NOTES: torch.tensor(note_vocab_range),
    TokenFields.CHORDS: torch.tensor(chord_vocab_range),
    TokenFields.OTHERS: torch.tensor(conditional_vocab_range),
}
if VELOCITY is not None:
    FIELD_VOCAB_RANGE[TokenFields.VELOCITY] = torch.tensor(VEL_RANGE)


def vocab_variables_to_device(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """cuda_id(device)를 받아, 위의 두개 상수들을 monkey patching하는 데에 쓰임."""
    _vocab_ranges = FIELD_VOCAB_RANGE.copy()
    for k, v in FIELD_VOCAB_RANGE.items():
        _vocab_ranges[k] = v.to(device)
    return TOKEN_IDS.to(device), _vocab_ranges


def extract_field_index(ids: Indices, field: str) -> torch.Tensor:
    """
    ids의 개별 토큰 id값들에 대해, 개별 인덱스가 field에 해당하는지를
    나타내는 True/False 어레이를 리턴. 슬라이싱 혹은 plotting 루틴에 사용됨.
    """
    vocab_range = FIELD_VOCAB_RANGE.get(field)
    return torch.isin(ids, vocab_range)


def parse_token_field_by_id(id: int) -> str:
    """
    토큰의 번호(id)를 받아 str 필드로 구분(ex: 2 -> "bar").
    """
    # special tokens
    if id == PAD:
        return TokenFields.PAD
    elif id == EOS:
        return TokenFields.EOS
    elif id == BAR:
        return TokenFields.BAR

    # note components
    elif id in POS_RANGE:
        return TokenFields.NOTE_POSITION
    elif id in VEL_RANGE:
        return TokenFields.VELOCITY
    elif id in PITCH_RANGE:
        return TokenFields.PITCH
    elif id in DUR_RANGE:
        return TokenFields.NOTE_DURATION

    # chord_components
    elif id in CHORD_POS_RANGE:
        return TokenFields.CHORD_POSITION
    elif id in CHORD_RANGE:
        return TokenFields.CHORD
    elif id in CHORD_BASS_RANGE:
        return TokenFields.CHORD_BASS
    elif id in CHORD_TENSION_RANGE:
        return TokenFields.CHORD_TENSION
    elif id in CHORD_DUR_RANGE:
        return TokenFields.CHORD_DURATION

    # others (meta, guideline, etc.)
    else:
        return TokenFields.OTHERS


INFO_STR_LEN: int = 16
SPACE: str = " "


def fill_spaces(s: str) -> str:
    """input str 변수 s의 길이를 INFO_STR_LEN 만큼 space로 채워 맞춤."""
    s += SPACE * (INFO_STR_LEN - len(s))
    return s


def parse_token_representation_by_id(id: int) -> str:
    """토큰의 번호(id)를 받아 속성을 요약하는 str을 반환."""
    # special tokens
    if id == PAD:
        repr = TokenFields.PAD
    elif id == EOS:
        repr = TokenFields.EOS
    elif id == BAR:
        repr = TokenFields.BAR

    # note components
    elif id in POS_RANGE:
        repr = f"| pos: {id - NOTE_POSITION.offset}/{NOTE_POSITION.vocab_size}"
    elif id in VEL_RANGE:
        repr = f"| vel: {id - VELOCITY.offset}/{VELOCITY.vocab_size}"
    elif id in PITCH_RANGE:
        pitch_octave, pitch_num = divmod(id - PITCH.offset, NUM_PITCHES)
        pitch_name = PITCH_MAP.get(pitch_num)
        repr = f"| {pitch_name}{pitch_octave - 2}"
    elif id in DUR_RANGE:
        repr = f"| dur: {id - NOTE_DURATION.offset}/{NOTE_DURATION.vocab_size}"

    # chord_components
    elif id in CHORD_POS_RANGE:
        repr = f"| pos: {id - CHORD_POSITION.offset}/{CHORD_POSITION.vocab_size}"
    elif id in CHORD_RANGE:
        chord_pitch, chord_quality = divmod(id - CHORD.offset, NUM_CHORD_QUALITY)
        repr = f"| {PITCH_MAP.get(chord_pitch) + CHORD_QUALITY_MAP.get(chord_quality)}"
    elif id in CHORD_BASS_RANGE:
        repr = f"| {id - CHORD_BASS.offset}"
    elif id in CHORD_TENSION_RANGE:
        repr = f"| {CHORD_TENSION_MAP.get(id - CHORD_TENSION.offset)}"
    elif id in CHORD_DUR_RANGE:
        repr = f"| dur: {id - CHORD_DURATION.offset}/{CHORD_DURATION.vocab_size}"

    # others (meta, guideline, etc.)
    else:
        repr = ""

    # appending space until INFO_STR_LEN
    repr = fill_spaces(repr)
    return repr
