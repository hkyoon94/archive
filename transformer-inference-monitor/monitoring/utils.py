import enum
import re
from fractions import Fraction
from time import time
from typing import Any, Callable, ClassVar, Union

from pydantic import BaseModel, field_validator, model_validator


class TextColors:
    ENDC = "\033[0m"  # basic
    RED = "\33[31m"  # red
    BLUE = "\033[94m"  # blue
    GREEN = "\033[92m"  # green
    REDBG = "\33[41m"  # red-background
    BLUEBG = "\33[44m"  # blue-background
    WARN = "\033[93m"  # used for warning msg
    BOLD = "\033[1m"  # bold
    UNDERLINE = "\033[4m"  # underline


class ExcessiveDisplayChecker:
    MAX_DISP_NUM = 20

    def _check(self, display_num: int) -> bool:
        """True를 리턴할 경우 시각화 및 display가 중단됨"""
        if display_num <= self.MAX_DISP_NUM:
            return False
        yn = input(
            f"WARNING | 시각화 요청 갯수가 너무 많습니다 "
            f"({self.MAX_DISP_NUM}개 이상: {display_num}개).\n"
            "계속 진행하시겠습니까? [Enter(y)/n]"
        )
        while True:
            if yn == "" or yn == "y":
                return False
            elif yn == "n":
                return True
            yn = input("잘못 입력하셨습니다. 계속 진행하시겠습니까? [Enter(y)/n]")


def set_max_disp_num(n: int) -> None:
    ExcessiveDisplayChecker.MAX_DISP_NUM = n


def timer(func: Callable) -> Callable:
    """타이머 데코레이터"""

    def wrapper(*args, **kwargs) -> Any:
        st = time()
        func_output = func(*args, **kwargs)
        ft = time()
        print(f"Task '{func.__name__}()' elapsed {ft - st:.6f} seconds.")
        return func_output

    return wrapper


def capitalize_func_name(func: Callable) -> str:
    return func.__name__.capitalize().replace("_", " ")


class ChordParsePattern(enum.Enum):
    ROOT_PATTERN = re.compile(r"[A-G](#|b|)")
    BASS_PATTERN = re.compile("/")
    TENSION_PATTERN = re.compile(r"\([^)]*\)")
    TENSION_LIST_PATTERN = re.compile("(#|b|)[0-9]+")
    ACCIDENTAL_PATTERN = re.compile(r"[#|b]")


class ChordName(BaseModel):
    name: str

    @property
    def root(self):
        root = ChordParsePattern.ROOT_PATTERN.value.match(self.name).group()
        return root

    @property
    def quality(self):
        with_bass = bool(ChordParsePattern.BASS_PATTERN.value.findall(self.name))
        quality = (
            ChordParsePattern.BASS_PATTERN.value.split(self.name)[0] if with_bass else self.name
        )
        quality = ChordParsePattern.TENSION_PATTERN.value.sub("", quality)
        quality = ChordParsePattern.ROOT_PATTERN.value.sub("", quality)
        return quality

    @property
    def tension(self):
        with_tension = bool(ChordParsePattern.TENSION_PATTERN.value.findall(self.name))
        tensions = (
            ChordParsePattern.TENSION_PATTERN.value.search(self.name).group()
            if with_tension
            else ""
        )
        tension_notes = [
            i.group() for i in ChordParsePattern.TENSION_LIST_PATTERN.value.finditer(tensions)
        ]
        return tension_notes

    @property
    def bass(self):
        with_bass = bool(ChordParsePattern.BASS_PATTERN.value.findall(self.name))
        bass = ChordParsePattern.BASS_PATTERN.value.split(self.name)[1] if with_bass else ""
        return bass

    def __repr__(self) -> str:
        return self.name


class ChordInfo(BaseModel):
    num_measures: int
    time_signature: str
    chord_progression: list[ChordName]
    default_num_beats: ClassVar[int] = 4
    chords_per_beat: ClassVar[int] = 2

    @property
    def _split_by_bar(self) -> list[list[ChordName]]:
        n = len(self.chord_progression)
        avg_per_measure = n // self.num_measures
        remainder = n % self.num_measures

        result = []
        start = 0
        for i in range(self.num_measures):
            end = start + avg_per_measure + (1 if i < remainder else 0)
            chords_in_measure = self.chord_progression[start:end]
            chord_names = [chord for chord in chords_in_measure]
            result.append(chord_names)
            start = end
        return result

    def _make_chord_indices_and_names(self) -> tuple[list[Fraction], list[ChordName]]:
        chords_per_bar = (
            self.chords_per_beat * Fraction(self.time_signature) * self.default_num_beats
        )

        idx_list = []
        name_list = []
        for bar_idx, bar in enumerate(self._split_by_bar):
            for c_idx, chord in enumerate(bar):
                if c_idx == 0 or chord != name_list[-1]:
                    idx_list.append(bar_idx + Fraction(c_idx, chords_per_bar))
                    name_list.append(chord)
        return idx_list, name_list

    @property
    def chord_indices(self) -> list[Fraction]:
        return self._make_chord_indices_and_names()[0]

    @property
    def chord_names(self) -> list[ChordName]:
        return self._make_chord_indices_and_names()[1]

    @property
    def chord_durations(self) -> list[Fraction]:
        duration_list = []
        for i in range(len(self.chord_indices) - 1):
            duration = self.chord_indices[i + 1] - self.chord_indices[i]
            duration_list.append(duration)
        duration_list.append(self.num_measures - self.chord_indices[-1])
        return duration_list

    @field_validator("num_measures", mode="before")
    def validate_num_measures(cls, v: Union[int, float]):
        if not isinstance(v, int):
            return int(v)
        return v

    @field_validator("chord_progression", mode="before")
    def validate_chord_progression(cls, v: Union[list[str], list[ChordName]]):
        if all(isinstance(chord, str) for chord in v):
            return [ChordName(name=chord) for chord in v]
        return v

    @model_validator(mode="after")
    def validate_chord_progression_length(self):
        chord_progression = self.chord_progression
        input_length = len(chord_progression)
        num_measures = self.num_measures
        time_signature = Fraction(self.time_signature)
        chords_per_bar = time_signature * self.default_num_beats * self.chords_per_beat
        valid_length = chords_per_bar * num_measures

        if input_length != valid_length:
            raise ValueError("chord progression length 오류")
        return self

    @classmethod
    def create(
        cls,
        num_measures: int,
        time_signature: str,
        chord_progression: list[str],
        **kwargs,
    ):
        return cls(
            chord_progression=[ChordName(name=chord) for chord in chord_progression],
            num_measures=num_measures,
            time_signature=time_signature,
        )

    def __str__(self) -> str:
        return f"{'_'.join([chord.name for chord in self.chord_progression])}"
