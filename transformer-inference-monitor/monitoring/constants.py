from yacs.config import CfgNode

from midiai.inspector.quality_check.dissonance.constants import NUM_TO_PITCH
from midiai.inst_gen.vocab import InstGenVocab
from midiai.inst_gen.vocab.maps import ChordEncodingMap
from midiai.typevs import V
from midiai.vocal_gen.vocab import VocalGenVocab

# loading package config
monitoring_cfg = CfgNode()
monitoring_cfg.VOCAB_TYPE = "inst-gen"
monitoring_cfg.NOTE_COMPLETION = [
    "NOTE_POSITION",
    "VELOCITY",
    "PITCH",
    "NOTE_DURATION",
]
monitoring_cfg.DEFAULT_TOP_K = 32
monitoring_cfg.DEFAULT_SOFTMAX_TEMP = 0.95
monitoring_cfg.merge_from_file("./midiai/inspector/monitoring/monitoring_config.yml")

# Initializing vocab
VOCAB_FACTORY: dict[str, V] = {
    "inst-gen": InstGenVocab,
    "vocal-gen": VocalGenVocab,
}

vocab = VOCAB_FACTORY.get(monitoring_cfg.VOCAB_TYPE)
if vocab is None:
    raise ValueError("Unknown vocabulary info!")

NOTE_FIELDS: tuple[str] = tuple(monitoring_cfg.NOTE_COMPLETION)

# setting default constants
DEFAULT_TOP_K: int = monitoring_cfg.DEFAULT_TOP_K
DEFAULT_SOFTMAX_TEMP: float = monitoring_cfg.DEFAULT_SOFTMAX_TEMP

# static constants
OCTAVES: list[str] = ["8", "7", "6", "5", "4", "3", "2", "1", "0", "-1", "-2"]
NUM_OCTAVES = len(OCTAVES)

PITCH_MAP: dict[int, str] = NUM_TO_PITCH
NUM_PITCHES = len(PITCH_MAP)

CHORD_NOTE_MAP = {
    "": (0, 4, 7),
    "maj7": (0, 4, 7, 11),
    "7": (0, 4, 7, 10),
    "6": (0, 4, 7, 9),
    "m": (0, 3, 7),
    "m6": (0, 3, 7, 9),
    "m7": (0, 3, 7, 10),
    "dim": (0, 3, 6),
    "dim7": (0, 3, 6, 9),
    "m7b5": (0, 3, 6, 10),
    "+": (0, 4, 8),
    "+7": (0, 4, 8, 10),
    "sus4": (0, 5, 7),
    "sus2": (0, 2, 7),
    "add2": (0, 2, 4, 7),
    "madd2": (0, 2, 3, 7),
    "madd4": (0, 3, 5, 7),
    "7sus4": (0, 5, 7, 10),
    "mM7": (0, 3, 7, 11),
}

CHORD_QUALITY_MAP = ChordEncodingMap.QUALITY_MAP
chord_quality_map = {}
for k, v in CHORD_QUALITY_MAP.items():
    if v not in chord_quality_map:
        chord_quality_map[v] = k
CHORD_QUALITY_MAP = chord_quality_map
NUM_CHORD_QUALITY = len(CHORD_QUALITY_MAP)

CHORD_TENSION_MAP = {v: k for k, v in ChordEncodingMap.TENSION_MAP.items()}
