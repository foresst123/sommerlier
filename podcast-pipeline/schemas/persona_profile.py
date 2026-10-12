"""Persona profile of one recording: what profile.json and stats.json hold.

None means missing data, 0 / 0.0 means "had the chance and did not", [] on
llm_candidate_forms means "the LLM ran and found nothing". Never swap them.
"""
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional


@dataclass
class Dist:                       # distribution; p* are None when n is too small
    p10: Optional[float]
    p50: Optional[float]
    p90: Optional[float]
    n: int


@dataclass
class Rate:                       # frequency; value None when exposure is too small
    value: Optional[float]
    count: int
    exposure_min: float


@dataclass
class Ratio:                      # proportion; value None when n is too small
    value: Optional[float]
    n: int


@dataclass
class Filler:
    form: str
    per_min: float


@dataclass
class Particle:
    form: str
    per_100_sent: float


@dataclass
class AddressTerm:
    form: str
    count: int


@dataclass
class LexiconItem:
    form: str
    share: float


@dataclass
class Style:
    speech_rate_wps: Dist
    words_per_sentence: Dist
    fillers: Optional[List[Filler]]            # None = low exposure; never []
    particles: Optional[List[Particle]]
    address_terms: Optional[List[AddressTerm]]
    llm_candidate_forms: Optional[List[str]]   # None = LLM did not run / unreadable; [] = ran, nothing new
    description: Optional[str]


@dataclass
class BackchannelProfile:
    per_min_of_other: Rate
    lexicon: Optional[List[LexiconItem]]
    overlap_ratio: Ratio


@dataclass
class InterruptionProfile:
    per_hour_of_other: Rate
    yield_rate: Ratio


@dataclass
class TurnProfile:
    duration_s: Dist
    response_latency_s: Dist                   # negative = came in before the other finished


@dataclass
class SpeakerProfile:
    speech_s: float
    style: Style
    backchannel: BackchannelProfile
    interruption: InterruptionProfile
    turn: TurnProfile


@dataclass
class PersonaProfile:                          # = profile.json
    schema_version: int
    params_version: str
    source_type: str                           # "real" | "synthetic"
    recording_id: str
    usable_ratio: Optional[float]
    speakers: Dict[str, SpeakerProfile]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PersonaStats:                            # = stats.json
    excluded_s: Dict[str, float]
    unknown_gaps_skipped: int
    llm_windows: int
    llm_ok: int
    warnings: List[str]

    def to_dict(self) -> dict:
        return asdict(self)
