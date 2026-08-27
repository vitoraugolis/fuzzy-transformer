"""Fuzzificação de janelas de processo.

Converte a entrada crua do modelo — um dicionário ``{tag: [x_{k-n}, ..., x_k]}``
— na representação que o tokenizador consome: para cada par (tag, instante),
um conjunto esparso de *slots* de estado com seus graus de pertinência.

    {"T-102": [..., 101.563]}  ->  slots [("T-102","level","mediano"),
                                          ("T-102","level","alto"), ...]
                                   pesos [0.50, 0.30, ...]

O resultado é deliberadamente esparso (P slots por token, P pequeno): o token
de estado é uma *combinação convexa* de poucos embeddings, o que mantém o
custo baixo e a leitura linguística direta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .variables import VariableBook

Array = np.ndarray


@dataclass
class FuzzifiedWindow:
    """Saída da fuzzificação de uma janela; todos os arrays têm 1ª dim = S.

    Attributes
    ----------
    tag_index : (S,) int
        Índice da tag no :class:`VariableBook` — indexa a tabela de embeddings de ID.
    lag : (S,) int
        Defasagem temporal do token: 0 = instante k, 1 = k-1, ..., n = k-n.
    slot_ids : (S, P) int
        Índices no vocabulário achatado de estado (``book.term_slots()``).
    weights : (S, P) float
        Graus de pertinência correspondentes; somam 1 por dimensão linguística.
    mask : (S, P) bool
        True onde o slot é válido (o resto é padding).
    value : (S,) float
        Valor cru normalizado em [0, 1] — canal numérico residual (ver
        `docs/03`, questão de pesquisa QP-3).
    valid : (S,) bool
        False quando o dado original era NaN (falha de instrumento / gap).
    tags : list[str]
        Tag de cada token, para inspeção/depuração.
    """

    tag_index: Array
    lag: Array
    slot_ids: Array
    weights: Array
    mask: Array
    value: Array
    valid: Array
    tags: List[str]

    def __len__(self) -> int:
        return int(self.tag_index.shape[0])

    @property
    def n_tokens(self) -> int:
        return len(self)


class Fuzzifier:
    """Fuzzificador de janelas, dirigido por um :class:`VariableBook`.

    Parameters
    ----------
    book:
        Vocabulário de tags e variáveis linguísticas.
    top_p:
        Número máximo de slots retidos por dimensão e por token. Com partições
        de Ruspini triangulares, 2 slots já cobrem 100% da massa; ``top_p=3``
        dá folga para MFs gaussianas.
    layout:
        ``"tag_major"`` agrupa os tokens por tag (T-102 k..k-n, depois P-204...);
        ``"time_major"`` agrupa por instante. Não muda a matemática (a atenção é
        permutação-equivariante e a posição entra por embedding), mas muda o
        padrão de acesso e a leitura dos mapas de atenção.
    """

    def __init__(
        self,
        book: VariableBook,
        top_p: int = 3,
        layout: str = "tag_major",
    ) -> None:
        if layout not in ("tag_major", "time_major"):
            raise ValueError("layout deve ser 'tag_major' ou 'time_major'")
        self.book = book
        self.top_p = int(top_p)
        self.layout = layout
        self._offsets = book.slot_offsets()

    # -- API principal ------------------------------------------------------

    def __call__(self, window: Mapping[str, Sequence[float]], **kw) -> FuzzifiedWindow:
        return self.transform(window, **kw)

    def transform(
        self,
        window: Mapping[str, Sequence[float]],
        setpoints: Optional[Mapping[str, float]] = None,
    ) -> FuzzifiedWindow:
        """Fuzzifica uma janela ``{tag: valores}``.

        Os valores devem estar em ordem cronológica crescente: o **último**
        elemento é o instante ``k``. Todas as tags precisam ter o mesmo
        comprimento de janela.
        """
        tags = [t for t in self.book.tags if t in window]
        if not tags:
            raise ValueError("a janela não contém nenhuma tag conhecida do VariableBook")
        lengths = {len(window[t]) for t in tags}
        if len(lengths) != 1:
            raise ValueError(f"janelas com comprimentos diferentes: {lengths}")
        n_steps = lengths.pop()
        setpoints = setpoints or {}

        per_tag = {t: self._fuzzify_tag(t, window[t], setpoints.get(t)) for t in tags}
        max_p = max(p["slot_ids"].shape[1] for p in per_tag.values())

        order = self._token_order(tags, n_steps)
        S = len(order)
        tag_index = np.zeros(S, dtype=np.int64)
        lag = np.zeros(S, dtype=np.int64)
        slot_ids = np.zeros((S, max_p), dtype=np.int64)
        weights = np.zeros((S, max_p), dtype=np.float32)
        mask = np.zeros((S, max_p), dtype=bool)
        value = np.zeros(S, dtype=np.float32)
        valid = np.zeros(S, dtype=bool)
        token_tags: List[str] = []

        for row, (tag, step) in enumerate(order):
            p = per_tag[tag]
            width = p["slot_ids"].shape[1]
            tag_index[row] = self.book.index(tag)
            lag[row] = n_steps - 1 - step
            slot_ids[row, :width] = p["slot_ids"][step]
            weights[row, :width] = p["weights"][step]
            mask[row, :width] = p["mask"][step]
            value[row] = p["value"][step]
            valid[row] = p["valid"][step]
            token_tags.append(tag)

        return FuzzifiedWindow(
            tag_index=tag_index,
            lag=lag,
            slot_ids=slot_ids,
            weights=weights,
            mask=mask,
            value=value,
            valid=valid,
            tags=token_tags,
        )

    def batch(
        self,
        windows: Sequence[Mapping[str, Sequence[float]]],
        setpoints: Optional[Sequence[Mapping[str, float]]] = None,
    ) -> Dict[str, Array]:
        """Fuzzifica várias janelas e empilha em arrays ``(B, S, ...)``."""
        sps = setpoints or [None] * len(windows)
        fws = [self.transform(w, sp) for w, sp in zip(windows, sps)]
        widths = {f.slot_ids.shape[1] for f in fws}
        if len(widths) != 1:
            raise ValueError("janelas heterogêneas no batch")
        return {
            "tag_index": np.stack([f.tag_index for f in fws]),
            "lag": np.stack([f.lag for f in fws]),
            "slot_ids": np.stack([f.slot_ids for f in fws]),
            "weights": np.stack([f.weights for f in fws]),
            "mask": np.stack([f.mask for f in fws]),
            "value": np.stack([f.value for f in fws]),
            "valid": np.stack([f.valid for f in fws]),
        }

    def describe_token(self, fw: FuzzifiedWindow, row: int, top_k: int = 3) -> str:
        """Leitura em linguagem natural de um token — base da interpretabilidade."""
        slots = self.book.term_slots()
        tag = fw.tags[row]
        pairs = [
            (slots[int(s)][2], float(w))
            for s, w, m in zip(fw.slot_ids[row], fw.weights[row], fw.mask[row])
            if m
        ]
        pairs.sort(key=lambda p: -p[1])
        body = ", ".join(f"{w * 100:.0f}% {name}" for name, w in pairs[:top_k])
        return f"{tag}[k-{int(fw.lag[row])}]: {body}"

    # -- internos -----------------------------------------------------------

    def _token_order(self, tags: Sequence[str], n_steps: int):
        if self.layout == "tag_major":
            return [(t, s) for t in tags for s in range(n_steps)]
        return [(t, s) for s in range(n_steps) for t in tags]

    def _fuzzify_tag(
        self, tag: str, series: Sequence[float], setpoint: Optional[float]
    ) -> Dict[str, Array]:
        u = self.book[tag]
        x = np.asarray(series, dtype=float)
        valid = np.isfinite(x)
        # Gaps: preenche por hold (último valor válido) apenas para não propagar
        # NaN; o flag `valid` preserva a informação de que o dado faltava.
        x = _hold_fill(x)

        chunks_ids: List[Array] = []
        chunks_w: List[Array] = []
        for dim in u.dimensions:
            var = u.variables[dim]
            if dim == "level":
                signal = x
            elif dim == "trend":
                signal = np.diff(x, prepend=x[:1])
            elif dim == "error":
                if setpoint is None:
                    continue
                signal = x - float(setpoint)
            else:  # pragma: no cover - dimensões futuras
                continue
            mu = var.memberships(signal)  # (T, L)
            ids, w = _top_p(mu, self.top_p)
            chunks_ids.append(ids + self._offsets[(tag, dim)])
            chunks_w.append(w)

        slot_ids = np.concatenate(chunks_ids, axis=1)
        weights = np.concatenate(chunks_w, axis=1)
        return {
            "slot_ids": slot_ids,
            "weights": weights.astype(np.float32),
            "mask": weights > 0.0,
            "value": np.clip(u.normalize(x), -0.5, 1.5).astype(np.float32),
            "valid": valid,
        }


def _hold_fill(x: Array) -> Array:
    """Preenche NaNs com o último valor válido (e o primeiro válido no início)."""
    x = np.array(x, dtype=float, copy=True)
    good = np.isfinite(x)
    if not good.any():
        return np.zeros_like(x)
    idx = np.where(good, np.arange(len(x)), 0)
    np.maximum.accumulate(idx, out=idx)
    x = x[idx]
    first = np.argmax(good)
    x[:first] = x[first]
    return x


def _top_p(mu: Array, p: int):
    """Mantém os ``p`` maiores graus por linha, renormalizando para somar 1."""
    p = min(p, mu.shape[-1])
    idx = np.argsort(-mu, axis=-1)[..., :p]
    w = np.take_along_axis(mu, idx, axis=-1)
    w = np.where(w < 1e-6, 0.0, w)
    total = w.sum(axis=-1, keepdims=True)
    w = np.where(total > 1e-9, w / np.maximum(total, 1e-9), 0.0)
    return idx.astype(np.int64), w
