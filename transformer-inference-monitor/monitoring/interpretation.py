from __future__ import annotations

from functools import cached_property, partial
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from IPython.display import Audio
from miditoolkit import MidiFile
from miditoolkit.pianoroll import parser
from pretty_midi import PrettyMIDI
from torch import Tensor

from midiai.inspector.monitoring import vocab_utils
from midiai.inspector.monitoring.constants import DEFAULT_SOFTMAX_TEMP, DEFAULT_TOP_K
from midiai.inspector.monitoring.features import (
    BarUnionFeatureMatrix,
    EmbeddingWeights,
    Logits,
    PianoRoll,
    Probs,
    SelfAttentionMap,
    TimeSeries,
)
from midiai.inspector.monitoring.numeric_routines import (
    cosine_similarity,
    temperatured_softmax,
    top_k,
)
from midiai.inspector.monitoring.substructures.items import Item
from midiai.inspector.monitoring.substructures.itemunions import BarUnion, HarmonicUnion
from midiai.inspector.monitoring.substructures.parsers import ItemParser, ItemUnionParser
from midiai.inspector.monitoring.substructures.token import TokenHistory
from midiai.inspector.monitoring.types import FeatureValues
from midiai.inspector.monitoring.utils import ChordInfo
from midiai.inspector.monitoring.utils import TextColors as TC
from midiai.inspector.monitoring.vocab_utils import BAR, PAD
from midiai.inspector.monitoring.vocab_utils import ItemUnionFields as UF
from midiai.inspector.monitoring.vocab_utils import (
    parse_token_representation_by_id,
    vocab,
    vocab_variables_to_device,
)
from midiai.inst_gen.decoder.decoder import decode_midi as decode_inst_midi
from midiai.inst_gen.generation.initializer import InstGenConfigInitializer, InstGenModelInitializer
from midiai.inst_gen.vocab import InstGenVocab
from midiai.models.transformer_xl import MemTransformerLM
from midiai.vocal_gen.decoder.decoder import decode_midi as decode_vocal_midi
from midiai.vocal_gen.generation.initializer import (
    VocalGenConfigInitializer,
    VocalGenModelInitializer,
)
from midiai.vocal_gen.vocab import VocalGenVocab


class InterpretedSequence:
    """
    Auto-regressive decoding으로 생성된 토큰 시퀀스를 분석하기 위한 툴.
    (기능은 midiai/inspector/monitoring/sequence_interpretation_manual.ipynb 참고)
    """

    def __init__(
        self,
        sequence: Union[list[int], np.ndarray],
        info: Optional[dict[str, Any]] = None,
        logits: Optional[Tensor] = None,
        self_attention_map: Optional[Tensor] = None,
        prob_func: Callable = partial(temperatured_softmax, temperature=DEFAULT_SOFTMAX_TEMP),
        sampling_func: Callable = partial(top_k, k=DEFAULT_TOP_K),
        interpreter: Optional[SequenceInterpreter] = None,
        lighter_mode: bool = False,  # for fast overall computation (do not parses TokenHistory)
    ):
        self.sequence = sequence
        self.info = info

        # original feature references
        self._logits: Tensor = logits
        self._self_attention_map: Tensor = self_attention_map
        self._probs: Tensor = None
        self._device = logits.device
        self._sequence: Tensor = torch.tensor(sequence).long().to(self._device)

        self.prob_func = prob_func
        self.sampling_func = sampling_func

        # TODO: 인터프리터 넣지 말고 step 샘플링용 컨트롤러를 따로 ..?
        self.interpreter = interpreter
        self.lighter_mode = lighter_mode

        self.tokens: list[TokenHistory] = []
        self.len_tokens: int = 0
        self._update_interpreted_info(logits, self_attention_map)  # TODO: 최적화 개선

    def _parse_tokens(self) -> None:
        new_tokens = []
        for token_index, token_id in enumerate(
            self.sequence[self.len_tokens :], start=self.len_tokens
        ):
            """최적화를 위해 개별 token 객체는 모두 self._<Features>의 '주소'만을 참조함"""
            new_token = TokenHistory(
                id=token_id,
                index=token_index,
                logits=self._logits,
                probs=self._probs,
                self_attention_map=self._self_attention_map,
            )
            new_tokens.append(new_token)
        self.tokens.extend(new_tokens)
        self.len_tokens = len(self.tokens)

    def _update_interpreted_info(self, logits: Tensor, self_attention_map: Tensor) -> None:
        self._logits = logits
        probs = self.prob_func(logits)
        self._probs = self.sampling_func(probs) if self.sampling_func is not None else probs
        self._self_attention_map = self_attention_map
        if not self.lighter_mode:
            self._parse_tokens()

    def set_prob_func(self, prob_func: Optional[Callable]) -> None:
        """
        logits -> probs를 산출하기 위해 적용해야 할 criterion을 설정.
        (probs = prob_func(logits))
        """
        self.prob_func = prob_func

    def set_sampling_func(self, sampling_func: Optional[Callable]) -> None:
        """
        probs -> sampling_probs를 산출하기 위해 적용해야 할 criterion을 설정.
        (sampling_probs = sampling_func(probs, sequence))  # TODO: sequence 전달 검토
        """
        self.sampling_func = sampling_func

    @cached_property
    def num_info_tokens(self) -> int:
        """conditional(prompt) 토큰들의 수"""
        for num_info_tokens, token_id in enumerate(self.sequence):
            if token_id == BAR:
                break
        return num_info_tokens

    @cached_property
    def num_event_tokens(self) -> int:
        """순수 생성된 (첫 BAR 포함 이후) 토큰들의 수"""
        return len(self.sequence) - self.num_info_tokens

    # EXTRACTING SUBSTRUCTURES

    @cached_property
    def items(self) -> list[Item]:
        """self.sequence내의 토큰들을 Item 객체 단위로 묶어 리스트로 반환"""
        return ItemParser().parse_items(tokens=self.tokens)

    @cached_property
    def bar_unions(self) -> list[BarUnion]:
        """마디 단위로 item들을 묶은 BarUnion 객체들의 리스트를 반환"""
        return ItemUnionParser().parse_bar_unions(items=self.items)

    @cached_property
    def harmonic_unions(self) -> list[HarmonicUnion]:
        """하나의 코드가 지속될 때의 item들을 묶은 HarmonicUnion 객체들의 리스트를 반환"""
        chord_info = ChordInfo(**self.info)
        return ItemUnionParser().parse_harmonic_unions(items=self.items, info=chord_info)

    # ORIGINAL FEATURES

    @cached_property
    def logits(self) -> Logits:
        """self._Logits를 Logits 클래스로 wrap 후 리턴하는 fast routine"""
        return Logits(
            value=self._logits[:-1],  # TODO: next step 샘플링용 last_logit 및 last_prob를 따로..?
            row_axis=self._sequence[1:],  # causal shift
            col_axis=vocab_utils.TOKEN_IDS,
        )

    @cached_property
    def probs(self) -> Probs:
        """self._Probs를 Probs 클래스로 wrap 후 리턴하는 fast routine"""
        return Probs(
            value=self._probs[:-1],  # TODO: next step 샘플링용 last_logit 및 last_prob를 따로..?
            row_axis=self._sequence[1:],  # causal shift
            col_axis=vocab_utils.TOKEN_IDS,
        )

    @cached_property
    def self_attention_map(self) -> SelfAttentionMap:
        """self._AttentionScores를 AttenstionScores로 wrap 후 리턴하는 fast routine"""
        return SelfAttentionMap(
            value=self._self_attention_map,
            queries=self._sequence,
            keys=self._sequence,  # self attn, i.e., keys == queries
        )

    # EXTRACTING SEQUENCE-LEVEL FEATURES

    def stepwise_probabilities(
        self,
        target_field: Optional[str] = None,
        values_only: bool = False,
    ) -> Union[FeatureValues, TimeSeries]:
        """self.probs를 이용해 해당 시퀀스가 순차적으로 샘플링 될 때의 확률값들을 샘플링 스텝별로 플릇"""
        return self.probs.slice(
            row=target_field,
        ).target_probabilities(values_only=values_only)

    def stepwise_nll(
        self,
        target_field: Optional[str] = None,
        values_only: bool = False,
    ) -> Union[FeatureValues, TimeSeries]:
        """self.logits로부터 다음 스텝의 토큰을 타겟으로 하는 negative log loss를 계산"""
        return self.probs.slice(
            row=target_field,
        ).target_nll(values_only=values_only)

    def stepwise_entropy(
        self,
        target_field: Optional[str] = None,
        vocab_field: Optional[str] = None,
        values_only: bool = False,
    ) -> Union[FeatureValues, TimeSeries]:
        """self.probs의 샘플링 스텝별 확률의 엔트로피를 계산"""
        return self.probs.slice(
            row=target_field,
            col=vocab_field,
        ).stepwise_entropy(values_only=values_only)

    def joint_sampling_prob(self, target_field: Optional[str] = None) -> float:
        """self.logits로부터 구한 nll값을 이용해, 해당 시퀀스 전체가 샘플링 될 확률값을 계산"""
        return self.stepwise_nll(
            target_field=target_field,
            values_only=True,
        ).mean()

    def grooving_pattern_similarity(
        self, resolution: int = 48, values_only: bool = False
    ) -> Union[FeatureValues, BarUnionFeatureMatrix]:
        """각 마디들 내 onset의 pair-wise 코사인 유사도를 계산"""
        bar_unions = self.bar_unions
        n = len(bar_unions)
        sim_scores = np.zeros((n, n))
        for i, bar1 in enumerate(bar_unions):
            for j, bar2 in enumerate(bar_unions):
                bar1_onsets = bar1.onsets(
                    resolution=resolution,
                    values_only=True,
                )
                bar2_onsets = bar2.onsets(
                    resolution=resolution,
                    values_only=True,
                )
                sim_scores[i, j] = cosine_similarity(bar1_onsets, bar2_onsets)

        if values_only:
            return sim_scores
        return BarUnionFeatureMatrix(
            sim_scores,
            name="Grooving Pattern Similarity",
        )

    def piano_roll(self, tick_resolution: int = 5, values_only: bool = False) -> PianoRoll:
        """self.sequence를 시각화 가능한 PianoRoll객체로 리턴"""
        decoded_midi = self.decode()
        ticks_per_beat = decoded_midi.ticks_per_beat
        time_signature = self.info.get("time_signature")
        if time_signature is None:
            raise ValueError(f"wrong info: {self.info}")
        beats_per_bar = int(time_signature.split("/")[0])
        resolution_per_beat = ticks_per_beat / tick_resolution
        # num_bars = ceil(notes[-1].end / (tpb * num_beats))
        num_bars = self.sequence.count(BAR)
        notes = decoded_midi.instruments[0].notes
        piano_roll = np.zeros((int(resolution_per_beat * num_bars * beats_per_bar), 128))
        pr_ = parser.notes2pianoroll(
            notes,
            resample_factor=1 / tick_resolution,
        )
        piano_roll[: pr_.shape[0]] = pr_

        if values_only:
            return piano_roll
        return PianoRoll(piano_roll, num_bars=num_bars)

    def extract_all_statistics(self) -> Any:
        """모든 statistics를 한 번에 추출"""
        ...  # TODO: 작성

    # AUXILIARY ROUTINES FOR CONVENIENCE

    def disp_items(self) -> None:
        """self.items내의 item index를 같이 출력하여 가독성 보조"""
        for i, item in enumerate(self.items):
            print(f"{TC.BOLD}{TC.GREEN}{i}{TC.ENDC}\t: {item.__repr__()}")

    def disp_unions(self, union_type: str = UF.BAR) -> None:
        """self.item_unions내의 union index를 같이 출력하여 가독성 보조"""
        unions = self.bar_unions if union_type == UF.BAR else self.harmonic_unions
        for i, union in enumerate(unions):
            print(f"{TC.BOLD}{TC.GREEN}{i}{TC.ENDC}:\n{union.__repr__()}")

    def decode(self, save_path: Optional[str] = None) -> MidiFile:
        """self.sequence를 MidiFile로 디코딩"""
        if vocab == InstGenVocab:
            decoded_midi = decode_inst_midi(self.info, self.sequence)
        elif vocab == VocalGenVocab:
            decoded_midi = decode_vocal_midi(self.info, self.sequence)

        if save_path is not None:
            decoded_midi.dump(save_path)

        return decoded_midi

    def render(self, save_path: str = None) -> Audio:
        """self.sequence를 오디오 형태로 렌더링"""
        # TODO: BytesIO buffer 사용 (근데 왜 안 되지..)
        save_path = "./tmp.mid" if save_path is None else save_path
        self.decode(save_path)
        pretty_midi_ported = PrettyMIDI(save_path)

        return Audio(pretty_midi_ported.synthesize(), rate=44100)

    # STEP-BY-STEP SAMPLING ROUTINES
    # TODO: 아직 작업 중... -----------------------------------------------------------------

    def sample(self, counts: int = 1) -> None:
        """self.probs의 index번째 토큰 파생 확률값들로부터 1개를 샘플링하여 붙임(forward action)"""
        current_length = len(self.sequence)
        for ct in range(counts):
            token = torch.multinomial(self._Logits[-1], 1).item() + 1  # index 1 밀린 것 원복
            print(
                f"Sampled {parse_token_representation_by_id(token)} "
                f"index: {current_length + ct}"
            )
            self.sequence = np.r_[self.sequence, token]  # inserts at last
            self.interpreter.interpret(sequence=self, pad=True)

    def force(self, tokens: Union[list[int], int]) -> None:
        """self.probs의 값과 관계없이 tokens로 할당받은 시퀀스를 강제로 self.sequence에 붙임"""
        tokens = [tokens] if isinstance(tokens, int) else tokens
        current_length = len(self.sequence)
        for ct, token in enumerate(tokens):
            print(
                f"Sampled {parse_token_representation_by_id(token)} "
                f"index: {current_length + ct}"
            )
            self.sequence = np.r_[self.sequence, token]  # inserts at last

        self.interpreter.interpret(sequence=self, pad=True)

    def drop(self, counts: int = 1) -> None:
        """self.sequence와 그에 관계된 모든 original feature들의 정보를 counts만큼 끝에서부터 잘라냄"""
        if counts < 1:
            print("Dropping count must be greater than or equal to 1")
            return
        for ct in range(1, counts + 1):
            print(f"Dropped {self.tokens[-ct]}")

        self.sequence = self.sequence[:-counts]
        self._Logits = self._Logits[:-counts]
        self._Probs = self._Probs[:-counts]
        if self.mems is not None:
            self.mems = self.mems[:, :-counts]
        # self.attentions = ...    # TODO: 작성
        self.tokens = self.tokens[:-counts]


class SequenceInterpreter:
    """
    raw sequence와 info를 모델에 태워, 해석할 수 있는 InterpretedSequence 객체로 변환.
    변환은 .interpret() 메서드가 담당하며, info가 없어도 변환이 가능하지만:
    1. raw sequence만으론 InterpretedSequence의 정확한 메타정보를 알 수 없고,
    2. 해당 객체의 .decode(), .pianoroll(), .render()등 일부 편의기능 사용이 불가함.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model: Optional[MemTransformerLM] = None,
        map_location: str = "cuda:0",
        return_attns: bool = True,
        lighter_mode: bool = False,  # for fast overall computation (do not parses TokenHistory)
    ):
        self.device = torch.device(map_location)
        vocab_utils.TOKEN_IDS, vocab_utils.FIELD_VOCAB_RANGE = vocab_variables_to_device(
            self.device
        )  # monkey-patching constants: to device

        self.prob_func = partial(temperatured_softmax, temperature=DEFAULT_SOFTMAX_TEMP)
        self.sampling_func: Callable = partial(top_k, k=DEFAULT_TOP_K)

        if checkpoint_path is not None:
            if vocab == InstGenVocab:
                self.model = self._initialize_inst_gen(checkpoint_path, map_location)
            elif vocab == VocalGenVocab:
                self.model = self._initialize_vocal_gen(checkpoint_path, map_location)
            else:
                raise ValueError(f"Cannot infer unknown vocab type: {vocab}")
        else:
            self.model = model

        self.return_attns = return_attns
        self.lighter_mode = lighter_mode

    def _initialize_inst_gen(self, model_path: str, map_location: str) -> MemTransformerLM:
        cfg_initializer = InstGenConfigInitializer(
            model_path, inference_cfg_path="./midiai/inst_gen/inference_config.yml"
        )
        cfg = cfg_initializer.get_config()
        model_initializer = InstGenModelInitializer(
            model=MemTransformerLM(vocab_size=vocab.vocab_size, cfg=cfg),
            map_location=map_location,
            device=self.device,
        )
        model = model_initializer.initialize(model_path)
        return model

    def _initialize_vocal_gen(self, model_path: str, map_location: str) -> MemTransformerLM:
        cfg_initializer = VocalGenConfigInitializer(
            model_path, inference_cfg_path="./midiai/vocal_gen/inference_config.yml"
        )
        cfg = cfg_initializer.get_config()
        model_initializer = VocalGenModelInitializer(
            model=MemTransformerLM(vocab_size=vocab.vocab_size, cfg=cfg),
            map_location=map_location,
            device=self.device,
        )
        model = model_initializer.initialize(model_path)
        return model

    def set_prob_func(self, prob_func: Optional[Callable]) -> None:
        """
        logits -> probs를 산출하기 위해 적용해야 할 criterion을 설정.
        (probs = prob_func(logits))
        여기서 설정된 값들은 .interpret() 메서드를 통해 파생 InterpretedSequence로 전달됨.
        """
        self.prob_func = prob_func

    def set_sampling_func(self, sampling_func: Optional[Callable]) -> None:
        """
        probs -> sampling_probs를 산출하기 위해 적용해야 할 criterion을 설정.
        (sampling_probs = sampling_func(probs, sequence))  # TODO: sequence 전달 검토
        여기서 설정된 값들은 .interpret() 메서드를 통해 파생 InterpretedSequence로 전달됨.
        """
        self.sampling_func = sampling_func

    # @timer
    def _forward_sequence(
        self, sequence: list[int]
    ) -> tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """raw sequence를"""
        with torch.no_grad():
            input = torch.tensor(sequence).unsqueeze(dim=1).to(self.device)
            if self.return_attns:
                logits, *_, attns = self.model(input, return_attns=True)
                attns = torch.stack(attns)  # layer x head x seq_len x seq_len
            else:
                logits, *_ = self.model(input)
                attns = None
        return logits, attns

    # FORWARDING RAW-SEQUENCE
    def interpret(
        self,
        sequence: Union[list[int], np.ndarray, InterpretedSequence],
        info: dict[str, Any] = None,
        pad: bool = True,
    ) -> InterpretedSequence:
        """
        self._forward_sequence()를 이용해, InterpretedSequence로 해석하여 리턴하는 메서드.
        이미 InterpretedSequence를 받을 경우, raw sequence를 추출해 _forward_sequence()로 전달.
        """
        if isinstance(sequence, InterpretedSequence):  # sequence가 이미 Interpreted된 경우
            # info = sequence.info
            logits, self_attention_map = self._forward_sequence(sequence)
            sequence._update_interpreted_info(
                logits=logits.squeeze(dim=1),
                self_attention_map=self_attention_map,
            )
            return sequence

        # sequence가 raw sequence인 경우 (list[int]이거나, ndarray)
        if pad and (sequence[0] != PAD):  # padding
            sequence.insert(0, PAD)
        logits, self_attention_map = self._forward_sequence(sequence)

        logits = logits.squeeze(dim=1)
        if self_attention_map is not None:
            self_attention_map = self_attention_map
        else:
            self_attention_map = None

        return InterpretedSequence(
            sequence=sequence,
            info=info,
            logits=logits,
            self_attention_map=self_attention_map,
            prob_func=self.prob_func,
            sampling_func=self.sampling_func,
            interpreter=self,  # TODO: 인터프리터 넣지 말고 step 샘플링용 컨트롤러를 따로 ..?
            lighter_mode=self.lighter_mode,
        )

    # EXTRACTING MODEL PARAMETER-LEVEL FEATURES

    def embedding_weights(self, values_only: bool = False) -> EmbeddingWeights:
        """모델의 임베딩 weights를 해석할 수 있는 클래스를 리턴"""
        weights: Tensor = self.model.word_embedder.emb_layers[0].weight

        if values_only:
            return weights
        return EmbeddingWeights(weights.cpu().numpy())
