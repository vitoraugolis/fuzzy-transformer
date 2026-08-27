"""Vocabulário fuzzy do processo: variáveis linguísticas por tag.

Este módulo define o "livro de variáveis" (`VariableBook`) — o artefato que diz,
para cada tag do processo (T-102, P-204, V-097, ...), quais são os termos
linguísticos que descrevem seu estado e como um valor numérico se converte em
graus de pertinência.

Cada tag pode ter mais de uma *dimensão* linguística:

* ``level``  — o valor em si ("alto", "mediano", "baixo");
* ``trend``  — a taxa de variação ("subindo rápido", "estável", ...);
* ``error``  — o desvio em relação ao set-point (só para variáveis controladas).

A separação em dimensões é deliberada: o token de estado de uma tag é a soma
das contribuições de todas as suas dimensões, o que permite representar
"T-102 alto e subindo rápido" sem explosão combinatória de termos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .membership import MembershipFunction, Trapezoidal, Triangular

Array = np.ndarray

#: Nomes canônicos das dimensões linguísticas suportadas.
DIMENSIONS = ("level", "trend", "error")


@dataclass
class LinguisticVariable:
    """Uma dimensão linguística de uma tag: termos + MFs.

    Parameters
    ----------
    dimension:
        Uma de :data:`DIMENSIONS`.
    terms:
        Nomes dos termos, na ordem em que indexam o vetor de pertinência.
    functions:
        MFs, uma por termo, na mesma ordem.
    """

    dimension: str
    terms: List[str]
    functions: List[MembershipFunction]

    def __post_init__(self) -> None:
        if self.dimension not in DIMENSIONS:
            raise ValueError(
                f"dimensão {self.dimension!r} inválida; esperado uma de {DIMENSIONS}"
            )
        if len(self.terms) != len(self.functions):
            raise ValueError("terms e functions precisam ter o mesmo tamanho")
        if len(set(self.terms)) != len(self.terms):
            raise ValueError(f"termos duplicados em {self.dimension!r}: {self.terms}")

    def __len__(self) -> int:
        return len(self.terms)

    def memberships(self, x: Array, normalize: bool = True) -> Array:
        """Fuzzifica ``x``; devolve shape ``(..., n_termos)``.

        Com ``normalize=True`` os graus somam 1 (partição fuzzy), o que torna o
        token de estado uma *combinação convexa* dos embeddings dos termos —
        é essa propriedade que dá a leitura "30% alto, 50% mediano, ...".
        """
        x = np.asarray(x, dtype=float)
        mu = np.stack([f(x) for f in self.functions], axis=-1)
        if normalize:
            total = mu.sum(axis=-1, keepdims=True)
            mu = np.where(total > 1e-9, mu / np.maximum(total, 1e-9), 1.0 / mu.shape[-1])
        return mu

    def describe(self, mu: Array, top_k: int = 3, min_share: float = 0.05) -> str:
        """Descrição textual de um vetor de pertinência ('50% mediano, 30% alto')."""
        mu = np.asarray(mu, dtype=float).reshape(-1)
        order = np.argsort(-mu)[:top_k]
        parts = [
            f"{mu[i] * 100:.0f}% {self.terms[i]}" for i in order if mu[i] >= min_share
        ]
        return ", ".join(parts) if parts else "indefinido"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dimension": self.dimension,
            "terms": list(self.terms),
            "functions": [f.to_dict() for f in self.functions],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "LinguisticVariable":
        return LinguisticVariable(
            dimension=d["dimension"],
            terms=list(d["terms"]),
            functions=[MembershipFunction.from_dict(f) for f in d["functions"]],
        )


@dataclass
class TagUniverse:
    """Tudo que o modelo sabe *a priori* sobre uma tag.

    Além das variáveis linguísticas, guarda metadados de engenharia (faixa,
    unidade, limites de alarme, equipamento associado) que são usados para
    normalização, para o envelope de segurança e para ligar a tag às
    documentações (WOs, HAZOP) no estágio Walk/Run.
    """

    tag: str
    kind: str  # temperature | pressure | flow | level | valve | speed | ...
    unit: str = ""
    lo: float = 0.0
    hi: float = 1.0
    role: str = "measurement"  # measurement | manipulated | setpoint | state
    equipment: Optional[str] = None
    alarm_lo: Optional[float] = None
    alarm_hi: Optional[float] = None
    variables: Dict[str, LinguisticVariable] = field(default_factory=dict)

    def add(self, variable: LinguisticVariable) -> "TagUniverse":
        self.variables[variable.dimension] = variable
        return self

    @property
    def dimensions(self) -> List[str]:
        """Dimensões presentes, em ordem canônica."""
        return [d for d in DIMENSIONS if d in self.variables]

    @property
    def n_terms(self) -> int:
        """Número total de termos somando todas as dimensões."""
        return sum(len(self.variables[d]) for d in self.dimensions)

    def normalize(self, x: Array) -> Array:
        """Mapeia a faixa de engenharia para [0, 1] (canal numérico residual)."""
        span = max(self.hi - self.lo, 1e-9)
        return (np.asarray(x, dtype=float) - self.lo) / span

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "tag": self.tag,
            "kind": self.kind,
            "unit": self.unit,
            "lo": self.lo,
            "hi": self.hi,
            "role": self.role,
            "equipment": self.equipment,
            "alarm_lo": self.alarm_lo,
            "alarm_hi": self.alarm_hi,
            "variables": {k: v.to_dict() for k, v in self.variables.items()},
        }
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TagUniverse":
        return TagUniverse(
            tag=d["tag"],
            kind=d["kind"],
            unit=d.get("unit", ""),
            lo=d.get("lo", 0.0),
            hi=d.get("hi", 1.0),
            role=d.get("role", "measurement"),
            equipment=d.get("equipment"),
            alarm_lo=d.get("alarm_lo"),
            alarm_hi=d.get("alarm_hi"),
            variables={
                k: LinguisticVariable.from_dict(v)
                for k, v in d.get("variables", {}).items()
            },
        )


class VariableBook:
    """Coleção ordenada de :class:`TagUniverse` — o "vocabulário" do modelo.

    A ordem das tags fixa os índices usados nas tabelas de embedding, então o
    livro é serializado junto com o checkpoint. Mudar a ordem invalida o
    modelo treinado (verificado por :meth:`fingerprint`).
    """

    def __init__(self, universes: Iterable[TagUniverse] = ()) -> None:
        self._universes: List[TagUniverse] = list(universes)
        self._index = {u.tag: i for i, u in enumerate(self._universes)}
        if len(self._index) != len(self._universes):
            raise ValueError("tags duplicadas no VariableBook")

    def __len__(self) -> int:
        return len(self._universes)

    def __iter__(self):
        return iter(self._universes)

    def __getitem__(self, key) -> TagUniverse:
        if isinstance(key, str):
            return self._universes[self._index[key]]
        return self._universes[key]

    def __contains__(self, tag: object) -> bool:
        return tag in self._index

    def add(self, universe: TagUniverse) -> "VariableBook":
        if universe.tag in self._index:
            raise ValueError(f"tag duplicada: {universe.tag}")
        self._index[universe.tag] = len(self._universes)
        self._universes.append(universe)
        return self

    def index(self, tag: str) -> int:
        return self._index[tag]

    @property
    def tags(self) -> List[str]:
        return [u.tag for u in self._universes]

    def term_slots(self) -> List[tuple]:
        """Lista achatada de ``(tag, dimensão, termo)`` — o vocabulário de estado.

        O índice nessa lista é a linha da tabela de embeddings de estado: é a
        materialização da ideia de "um vetor de alta dimensão para 'alto em
        T-102'", distinto de "'alto' em P-204".
        """
        slots = []
        for u in self._universes:
            for dim in u.dimensions:
                for term in u.variables[dim].terms:
                    slots.append((u.tag, dim, term))
        return slots

    @property
    def n_state_slots(self) -> int:
        return sum(u.n_terms for u in self._universes)

    def slot_offsets(self) -> Dict[tuple, int]:
        """Offset inicial de cada ``(tag, dimensão)`` no vetor de estado achatado."""
        offsets, cursor = {}, 0
        for u in self._universes:
            for dim in u.dimensions:
                offsets[(u.tag, dim)] = cursor
                cursor += len(u.variables[dim])
        return offsets

    def fingerprint(self) -> str:
        """Hash estável do vocabulário, para checar compatibilidade de checkpoints."""
        import hashlib

        payload = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {"tags": [u.to_dict() for u in self._universes]}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "VariableBook":
        return VariableBook(TagUniverse.from_dict(t) for t in d["tags"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @staticmethod
    def load(path: str | Path) -> "VariableBook":
        return VariableBook.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------
# Construtores de conveniência
# ---------------------------------------------------------------------------

#: Termos padrão da dimensão `level` (5 termos, partição de Ruspini).
DEFAULT_LEVEL_TERMS = ("muito_baixo", "baixo", "mediano", "alto", "muito_alto")

#: Termos padrão da dimensão `trend`.
DEFAULT_TREND_TERMS = ("caindo_rapido", "caindo", "estavel", "subindo", "subindo_rapido")


def ruspini_partition(
    lo: float, hi: float, terms: Sequence[str] = DEFAULT_LEVEL_TERMS, dimension: str = "level"
) -> LinguisticVariable:
    """Partição fuzzy uniforme (triângulos com ombros trapezoidais) em [lo, hi].

    É o *ponto de partida* — não a resposta final. Os centros e larguras reais
    devem vir de conhecimento de processo (limites operacionais, alarmes) ou
    ser aprendidos (ver `docs/03-tokenizacao-e-fuzzificacao.md`).
    """
    n = len(terms)
    if n < 2:
        raise ValueError("é preciso ao menos 2 termos")
    centers = np.linspace(lo, hi, n)
    step = (hi - lo) / (n - 1)
    fns: List[MembershipFunction] = []
    for i, c in enumerate(centers):
        if i == 0:
            fns.append(Trapezoidal(a=lo - step, b=lo - step, c=c, d=c + step))
        elif i == n - 1:
            fns.append(Trapezoidal(a=c - step, b=c, c=hi + step, d=hi + step))
        else:
            fns.append(Triangular(a=c - step, b=c, c=c + step))
    return LinguisticVariable(dimension=dimension, terms=list(terms), functions=fns)


def partition_from_breakpoints(
    breakpoints: Sequence[float], terms: Sequence[str], dimension: str = "level"
) -> LinguisticVariable:
    """Partição fuzzy a partir de pontos de engenharia (limites, alarmes).

    ``breakpoints`` são os centros dos termos, em ordem crescente. Use isto para
    ancorar os termos em números que o operador reconhece: "alto" centrado no
    alarme H, "muito_alto" no HH, etc.
    """
    c = np.asarray(breakpoints, dtype=float)
    if len(c) != len(terms):
        raise ValueError("breakpoints e terms precisam ter o mesmo tamanho")
    if np.any(np.diff(c) <= 0):
        raise ValueError("breakpoints precisam ser estritamente crescentes")
    fns: List[MembershipFunction] = []
    for i in range(len(c)):
        left = c[i] - (c[i] - c[i - 1] if i > 0 else (c[1] - c[0]))
        right = c[i] + (c[i + 1] - c[i] if i < len(c) - 1 else (c[-1] - c[-2]))
        if i == 0:
            fns.append(Trapezoidal(a=left, b=left, c=c[i], d=right))
        elif i == len(c) - 1:
            fns.append(Trapezoidal(a=left, b=c[i], c=right, d=right))
        else:
            fns.append(Triangular(a=left, b=c[i], c=right))
    return LinguisticVariable(dimension=dimension, terms=list(terms), functions=fns)


def standard_tag(
    tag: str,
    kind: str,
    lo: float,
    hi: float,
    unit: str = "",
    role: str = "measurement",
    equipment: Optional[str] = None,
    alarm_lo: Optional[float] = None,
    alarm_hi: Optional[float] = None,
    level_terms: Sequence[str] = DEFAULT_LEVEL_TERMS,
    with_trend: bool = True,
    trend_span: Optional[float] = None,
) -> TagUniverse:
    """Cria uma tag com partição uniforme em `level` e (opcionalmente) `trend`.

    ``trend_span`` é a variação por amostra considerada "rápida"; por padrão 2%
    da faixa de engenharia por passo.
    """
    u = TagUniverse(
        tag=tag,
        kind=kind,
        unit=unit,
        lo=lo,
        hi=hi,
        role=role,
        equipment=equipment,
        alarm_lo=alarm_lo,
        alarm_hi=alarm_hi,
    )
    u.add(ruspini_partition(lo, hi, level_terms, dimension="level"))
    if with_trend:
        span = trend_span if trend_span is not None else 0.02 * (hi - lo)
        u.add(ruspini_partition(-span, span, DEFAULT_TREND_TERMS, dimension="trend"))
    return u
