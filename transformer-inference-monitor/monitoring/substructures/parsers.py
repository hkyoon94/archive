from typing import Optional

from midiai.inspector.monitoring.substructures.items import Bar, Chord, Eos, Item, Note
from midiai.inspector.monitoring.substructures.itemunions import BarUnion, HarmonicUnion
from midiai.inspector.monitoring.substructures.token import TokenHistory
from midiai.inspector.monitoring.utils import ChordInfo
from midiai.inspector.monitoring.vocab_utils import CHORD_FIELDS, NOTE_FIELDS
from midiai.inspector.monitoring.vocab_utils import ItemFields as IF
from midiai.inspector.monitoring.vocab_utils import TokenFields as TF

# TODO: Sample 의존성 제거


class ItemCompletionDetector:
    """
    연속된 의미 단위 토큰의 순서 감지기:
    fields의 순서와 맞는 토큰들이 순서대로 오면 해당 토큰들을 self.components에 저장함
    만약 다음 토큰의 종류가 fields의 순서와 어긋난다면, detect() 메서드가 False를 리턴
    """

    def __init__(self, fields: list[str]):
        self.fields = fields
        self.num_fields = len(fields)
        self.count = -1
        self.renew()

    def renew(self) -> None:
        self.next_component_index = 0
        self.components = []

    def detect(self, token: TokenHistory) -> Optional[bool]:
        self.components.append(token)
        if token.field == self.fields[self.next_component_index]:
            self.next_component_index += 1  # got correct completion order
            if self.next_component_index == self.num_fields:  # completion achieved
                self.count += 1
                return True

            return None  # completion pending

        return False  # wrong completion order


class ItemParser:
    """개별 토큰들이 모인 의미 단위인 Item 객체를 모으는 파서."""

    def __init__(self):
        self.note_detector = ItemCompletionDetector(fields=NOTE_FIELDS)
        self.chord_detector = ItemCompletionDetector(fields=CHORD_FIELDS)

    def _detect_first_bar_position(self, tokens: list[TokenHistory]) -> list[TokenHistory]:
        """첫 BAR의 위치 이전의 conditional item은 걸러냄"""
        first_bar_pos = None
        for i, token in enumerate(tokens):
            if token.field == TF.BAR:
                first_bar_pos = i
                break
        if first_bar_pos is not None:
            return tokens[first_bar_pos:]
        return []

    def _collect_valid_tokens(self, tokens: list[TokenHistory]) -> tuple[str, Item]:
        """
        의미 단위(Note, Chord 등)로 묶일 수 있는 연속된 토큰 집합을 감지한 뒤,
        pre_items에 Bar, Note 또는 Chord 단위 형태로 파싱된 토큰 집합을 튜플 형태로 정리함
        """

        def detect_and_collect(
            detector: ItemCompletionDetector,
            token: TokenHistory,
            item_field: str,
        ) -> None:
            nonlocal invalid_group_count
            detection_result = detector.detect(token)
            if detection_result is True:
                pre_items.append((f"{item_field}_{detector.count}", detector.components))
            elif detection_result is False:
                invalid_group_count += 1
                pre_items.append((f"{IF.INVALID}_{invalid_group_count}", detector.components))
            else:  # detection result is None
                return
            detector.renew()

        pre_items, bar_count, invalid_group_count = [], -1, -1
        for token in tokens:
            if token.field == TF.BAR:
                bar_count += 1
                pre_items.append((f"{IF.BAR}_{bar_count}", [token]))
            elif token.field == TF.EOS:
                pre_items.append((f"{IF.EOS}_0", [token]))
            elif token.field in NOTE_FIELDS:
                detect_and_collect(
                    detector=self.note_detector,
                    token=token,
                    item_field=IF.NOTE,
                )
            elif token.field in CHORD_FIELDS:
                detect_and_collect(
                    detector=self.chord_detector,
                    token=token,
                    item_field=IF.CHORD,
                )

        return pre_items

    def parse_items(self, tokens: list[TokenHistory]) -> list[tuple[str, Item]]:
        """의미 단위(Note, Chord)로 모인 토큰 집합을 Item객체로 최종 전환"""
        tokens_after_fist_bar = self._detect_first_bar_position(tokens)
        pre_items = self._collect_valid_tokens(tokens_after_fist_bar)
        valid_items = []
        for item_field, item_tokens in pre_items:
            _, count = item_field.split("_")
            if IF.NOTE in item_field:
                valid_items.append(
                    Note(item_tokens, count=int(count)),
                )
            elif IF.CHORD in item_field:
                valid_items.append(
                    Chord(item_tokens, count=int(count)),
                )
            elif IF.BAR in item_field:
                valid_items.append(
                    Bar(item_tokens, count=int(count)),
                )
            elif IF.EOS in item_field:
                valid_items.append(
                    Eos(item_tokens),
                )

        return valid_items


class ItemUnionParser:
    """
    의미 단위의 토큰 집합인 Item 객체들을
    Bar 혹은 Chord를 기준으로 끊어 ItemUnion객체로 전환하는 파서.
    """

    @staticmethod
    def _chunk_by_lead_type(lead_type: type, items: list[Item]) -> list[list[Item]]:
        split_index = []
        for i, item in enumerate(items):
            if isinstance(item, lead_type):
                split_index.append(i)

        split_index.append(len(items))
        item_chunks = []
        for split_start, split_end in zip(split_index[:-1], split_index[1:]):
            item_chunks.append(items[split_start:split_end])

        return item_chunks

    def parse_bar_unions(self, items: list[Item]) -> list[BarUnion]:
        """BarUnion 파싱"""
        item_chunks = self._chunk_by_lead_type(Bar, items)
        unions: list[BarUnion] = []

        for item_chunk in item_chunks:
            unions.append(BarUnion(items=item_chunk))

        return unions

    def parse_harmonic_unions(
        self, items: list[Item], info: ChordInfo = None
    ) -> list[HarmonicUnion]:
        """Harmonic(코드)Union 파싱"""
        item_chunks = self._chunk_by_lead_type(Chord, items)
        unions: list[HarmonicUnion] = []

        for item_chunk in item_chunks:
            unions.append(HarmonicUnion(items=item_chunk))

        if info is not None:  # info가 주어진 경우, 각 union에 info에 의도된 코드명을 부여
            for union, chord_name in zip(unions, info.chord_names):
                union.set_chord_name(chord_name.__repr__())

        return unions
