"""Cabeças de saída do FT-IC.

O modelo não devolve apenas um número. A tese do projeto é que a saída útil para
uma malha industrial tem três partes:

* **ação de controle** — expressa primeiro em termos linguísticos ("aumentar
  pouco") e depois defuzzificada em um valor numérico;
* **faixa admissível** — o envelope dentro do qual a ação foi filtrada, que é o
  mecanismo de contingência (ex.: limitar a abertura de uma válvula com
  problema mecânico conhecido);
* **orientações** — rótulos acionáveis para os times (manutenção, operação,
  processo) com a tag/equipamento a que se referem.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ControlOutput:
    term_logits: torch.Tensor   # (B, n_actions, T) distribuição linguística da ação
    delta_raw: torch.Tensor     # (B, n_actions) ação defuzzificada, antes do envelope
    band_lo: torch.Tensor       # (B, n_actions) limite inferior admissível
    band_hi: torch.Tensor       # (B, n_actions) limite superior admissível
    delta: torch.Tensor         # (B, n_actions) ação final (defuzzificada e filtrada)


class ControlHead(nn.Module):
    """Saída fuzzy da ação + defuzzificação por centroide + envelope aprendido.

    A defuzzificação usa centros ``c_t`` aprendidos por termo (inicializados
    uniformemente em [-1, 1], em unidades de ação normalizada):

        Δu = Σ_t softmax(logits)_t · c_t          (média ponderada / centroide)

    O envelope ``[lo, hi]`` é previsto pela própria rede a partir do contexto —
    é aqui que entra "a válvula está com folga mecânica, então trabalhe entre
    40% e 60%". O recorte final é feito por ``clamp``, o que mantém o gradiente
    fluindo pelos limites (eles são treináveis com supervisão de contingência).
    """

    def __init__(self, d_model: int, n_actions: int, n_terms: int = 7) -> None:
        super().__init__()
        self.n_actions, self.n_terms = n_actions, n_terms
        self.to_terms = nn.Linear(d_model, n_terms)
        self.to_band = nn.Linear(d_model, 2)
        self.centers = nn.Parameter(torch.linspace(-1.0, 1.0, n_terms))

    def forward(
        self, readout: torch.Tensor, hard_limits: Optional[torch.Tensor] = None
    ) -> ControlOutput:
        """``readout``: (B, n_actions, d) — um vetor por variável manipulada."""
        logits = self.to_terms(readout)
        delta_raw = (logits.softmax(dim=-1) * self.centers).sum(dim=-1)

        band = self.to_band(readout)
        center = torch.tanh(band[..., 0])
        half = F.softplus(band[..., 1]) + 1e-3
        lo, hi = center - half, center + half
        if hard_limits is not None:  # limites duros da instrumentação/IHM
            lo = torch.maximum(lo, hard_limits[..., 0])
            hi = torch.minimum(hi, hard_limits[..., 1])
        delta = torch.clamp(delta_raw, min=lo, max=hi)
        return ControlOutput(
            term_logits=logits,
            delta_raw=delta_raw,
            band_lo=lo,
            band_hi=hi,
            delta=delta,
        )


@dataclass
class AdvisoryOutput:
    logits: torch.Tensor                 # (B, n_advisories)
    pointer: Optional[torch.Tensor]      # (B, n_advisories, S) atribuição a tokens


class AdvisoryHead(nn.Module):
    """Orientações multi-rótulo + ponteiro de atribuição (a qual tag se refere).

    O ponteiro é uma atenção de uma consulta por rótulo sobre os tokens da
    janela: é o que transforma "notificar manutenção" em "notificar manutenção
    **sobre a V-097**".
    """

    def __init__(self, d_model: int, n_advisories: int) -> None:
        super().__init__()
        self.n_advisories = n_advisories
        self.query = nn.Parameter(torch.randn(n_advisories, d_model) * 0.02)
        self.key = nn.Linear(d_model, d_model)
        self.score = nn.Linear(d_model, 1)

    def forward(
        self,
        tokens: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_pointer: bool = True,
    ) -> AdvisoryOutput:
        k = self.key(tokens)                                   # (B,S,d)
        att = torch.einsum("ad,bsd->bas", self.query, k) / k.shape[-1] ** 0.5
        if key_padding_mask is not None:
            att = att.masked_fill(
                ~key_padding_mask[:, None, :], torch.finfo(att.dtype).min
            )
        w = att.softmax(dim=-1)
        pooled = torch.einsum("bas,bsd->bad", w, tokens)
        logits = self.score(pooled).squeeze(-1)
        return AdvisoryOutput(logits=logits, pointer=w if need_pointer else None)


class StateHead(nn.Module):
    """Prevê o estado fuzzy de um token (slots do vocabulário).

    Usada tanto no pré-treino auto-supervisionado (mascarar tokens de estado e
    reconstruí-los) quanto na previsão de k+1. Os pesos são amarrados à tabela
    de embeddings de estado — mesma ideia do *weight tying* dos modelos de
    linguagem: o espaço de saída é o mesmo vocabulário da entrada.
    """

    def __init__(self, state_embedding: nn.Embedding) -> None:
        super().__init__()
        self.state_embedding = state_embedding
        d = state_embedding.embedding_dim
        self.proj = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.bias = nn.Parameter(torch.zeros(state_embedding.num_embeddings))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.norm(F.gelu(self.proj(tokens)))
        return h @ self.state_embedding.weight.t() + self.bias
