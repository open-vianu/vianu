from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher

import cologne_phonetics as cologne
from abydos.phonetic import fonem


class Comparator(ABC):
    """Base class for comparators."""

    _precision: int = 2

    def __init__(self, ref: str):
        self._ref = ref

    @abstractmethod
    def apply(self, term: str) -> float:
        pass


class SoundsAlikeKolner(Comparator):
    """Uses the cologne_phonetics library to perform phonetic comparison of two strings."""

    def __init__(self, ref: str):
        super().__init__(ref)
        self._encoded_ref = cologne.encode(ref.upper())

    def apply(self, term: str) -> float:
        encoded_term = cologne.encode(term.upper())
        max_ratio = max(
            [
                SequenceMatcher(a=t[1], b=r[1]).ratio()
                for t in encoded_term
                for r in self._encoded_ref
            ]
        )
        return float(round(max_ratio, self._precision))


class SoundsAlikeFonem(Comparator):
    """Uses the abydos library to perform phonetic comparison of two strings."""

    def __init__(self, ref: str):
        super().__init__(ref)
        self._encoded_ref = fonem(ref.upper())

    def apply(self, term: str) -> float:
        encoded_term = fonem(term.upper())
        ratio = SequenceMatcher(a=encoded_term, b=self._encoded_ref).ratio()
        return float(round(ratio, self._precision))


class LooksAlike(Comparator):
    """Uses the difflib.SequenceMatcher to perform orthographic comparison of two strings."""

    def __init__(self, ref: str):
        super().__init__(ref)

    def apply(self, term: str) -> float:
        ratio = SequenceMatcher(a=term.upper(), b=self._ref.upper()).ratio()
        return float(round(ratio, self._precision))


@dataclass
class Match:
    """Match between a term and a reference."""

    term: str
    ref: str
    sounds_alike: float
    looks_alike: float
    source: str | None = None

    def __repr__(self):
        return f"<class 'Match'>: term={self.term}, ref={self.ref}, sounds_alike={self.sounds_alike}, looks_alike={self.looks_alike}, combined={self.combined}"

    def __str__(self):
        return self.__repr__()

    @property
    def combined(self) -> float:
        return (self.sounds_alike + self.looks_alike) / 2

    def to_dict(self) -> dict:
        dict_ = asdict(self)
        dict_["combined"] = self.combined
        return dict_


class LASA:
    """Look-And-Sound-Alike comparator."""

    def __init__(self, ref: str):
        self._ref = ref
        self._sounds_alike_kolner = SoundsAlikeKolner(ref=ref)
        self._sounds_alike_fonem = SoundsAlikeFonem(ref=ref)
        self._looks_alike = LooksAlike(ref=ref)

    def apply(self, term: str) -> Match:
        # Compute max sounds-alike value
        sa_kolner_val = self._sounds_alike_kolner.apply(term=term)
        sa_fonem_val = self._sounds_alike_fonem.apply(term=term)
        sa_val = max(sa_kolner_val, sa_fonem_val)

        # Compute looks-alike value
        la_val = self._looks_alike.apply(term=term)

        return Match(term=term, ref=self._ref, sounds_alike=sa_val, looks_alike=la_val)
