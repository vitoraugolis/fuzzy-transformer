"""Funções de pertinência (MFs) usadas na fuzzificação dos valores de processo.

Todas as funções recebem `x` (escalar ou ndarray) e devolvem graus de
pertinência em [0, 1] com o mesmo shape de `x`.

As MFs são objetos serializáveis (`to_dict`/`from_dict`) porque a definição das
variáveis linguísticas de cada tag é um artefato de engenharia — versionado
junto com o modelo, revisado por especialistas de processo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

Array = np.ndarray


class MembershipFunction:
    """Interface comum das MFs."""

    kind: str = "abstract"

    def __call__(self, x: Array) -> Array:  # pragma: no cover - interface
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        d["kind"] = self.kind
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MembershipFunction":
        d = dict(d)
        kind = d.pop("kind")
        try:
            cls = _REGISTRY[kind]
        except KeyError as exc:  # pragma: no cover - erro de configuração
            raise ValueError(f"MF desconhecida: {kind!r}") from exc
        return cls(**d)


@dataclass
class Triangular(MembershipFunction):
    """MF triangular definida por a <= b <= c."""

    a: float
    b: float
    c: float
    kind: str = field(default="tri", init=False, repr=False)

    def __call__(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        left = _safe_ratio(x - self.a, self.b - self.a)
        right = _safe_ratio(self.c - x, self.c - self.b)
        return np.clip(np.minimum(left, right), 0.0, 1.0)


@dataclass
class Trapezoidal(MembershipFunction):
    """MF trapezoidal definida por a <= b <= c <= d."""

    a: float
    b: float
    c: float
    d: float
    kind: str = field(default="trap", init=False, repr=False)

    def __call__(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        left = _safe_ratio(x - self.a, self.b - self.a)
        right = _safe_ratio(self.d - x, self.d - self.c)
        inside = np.ones_like(x)
        return np.clip(np.minimum(np.minimum(left, right), inside), 0.0, 1.0)


@dataclass
class Gaussian(MembershipFunction):
    """MF gaussiana; diferenciável, útil quando os centros são aprendidos."""

    center: float
    sigma: float
    kind: str = field(default="gauss", init=False, repr=False)

    def __call__(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        s = max(float(self.sigma), 1e-9)
        return np.exp(-0.5 * ((x - self.center) / s) ** 2)


@dataclass
class Sigmoidal(MembershipFunction):
    """MF sigmoidal (ombros abertos: 'muito alto', 'muito baixo')."""

    center: float
    slope: float
    kind: str = field(default="sigmoid", init=False, repr=False)

    def __call__(self, x: Array) -> Array:
        x = np.asarray(x, dtype=float)
        return 1.0 / (1.0 + np.exp(-self.slope * (x - self.center)))


def _safe_ratio(num: Array, den: float) -> Array:
    """num/den tolerando den == 0 (bordas verticais de trapézios/triângulos)."""
    if abs(den) < 1e-12:
        return np.where(num >= 0.0, np.inf, -np.inf)
    return num / den


_REGISTRY = {
    "tri": Triangular,
    "trap": Trapezoidal,
    "gauss": Gaussian,
    "sigmoid": Sigmoidal,
}
