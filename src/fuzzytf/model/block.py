"""Bloco do FT-IC: self-attention (contexto) + ANFIS (conhecimento).

A ordem espelha o transformer clássico — pré-normalização e conexões residuais —
mas com o MLP substituído pela camada neuro-fuzzy:

    h ← h + Attn(LN(h))          # extrai contexto entre tags e instantes
    h ← h + ANFIS(LN(h))         # aplica o conhecimento de processo (regras)

Empilhar N desses blocos é o que permite composição: o primeiro reconhece
"válvula saturando", o seguinte compõe isso com "temperatura subindo" para
inferir "trocador incrustado", e assim por diante.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..config import ModelConfig
from .anfis import AnfisLayer, AnfisTrace
from .attention import MultiHeadSelfAttention


@dataclass
class BlockTrace:
    attention: Optional[torch.Tensor]
    anfis: Optional[AnfisTrace]


class _MLPMixer(nn.Module):
    """MLP ponto-a-ponto com a mesma assinatura da :class:`AnfisLayer`."""

    def __init__(self, d: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d)
        )

    def forward(self, x, return_trace: bool = False):
        return self.net(x), None

    def rule_entropy(self):
        return torch.zeros((), device=next(self.parameters()).device)


class FuzzyTransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, n_tags: int) -> None:
        super().__init__()
        d = cfg.d_model
        self.norm_attn = nn.LayerNorm(d, eps=cfg.norm_eps)
        self.attn = MultiHeadSelfAttention(d, cfg.attention, n_tags=n_tags)
        self.norm_anfis = nn.LayerNorm(d, eps=cfg.norm_eps)
        if cfg.mixer == "anfis":
            self.anfis = AnfisLayer(d, cfg.anfis)
        elif cfg.mixer == "mlp":
            # Ablação de referência: o MLP clássico do transformer, para medir
            # quanto a camada neuro-fuzzy realmente agrega (ver docs/06).
            hidden = int(d * cfg.mlp_ratio)
            self.anfis = _MLPMixer(d, hidden, cfg.dropout)
        else:
            raise ValueError("model.mixer deve ser 'anfis' ou 'mlp'")
        self.drop = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        tag_index: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ):
        a, attn_w = self.attn(
            self.norm_attn(x),
            tag_index=tag_index,
            key_padding_mask=key_padding_mask,
            need_weights=return_trace,
        )
        x = x + self.drop(a)
        f, anfis_trace = self.anfis(self.norm_anfis(x), return_trace=return_trace)
        x = x + self.drop(f)
        trace = BlockTrace(attention=attn_w, anfis=anfis_trace) if return_trace else None
        return x, trace
