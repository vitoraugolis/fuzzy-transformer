"""Self-attention sobre a sequência de tokens de processo.

É a atenção padrão ``softmax(QKᵀ/√d_k)·V``, com duas adições específicas do
domínio:

1. **Viés relacional por par de tags** — um termo aprendido ``B[h, tag_i, tag_j]``
   somado aos scores. Ele pode ser inicializado a partir da topologia do P&ID
   (quais tags pertencem à mesma malha/equipamento), injetando estrutura de
   planta como prior em vez de exigir que a atenção a descubra do zero.
2. **Máscara de validade** — tokens de instrumento em falha não recebem atenção.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AttentionConfig


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, cfg: AttentionConfig, n_tags: int = 0) -> None:
        super().__init__()
        if d_model % cfg.n_heads:
            raise ValueError("d_model precisa ser divisível por n_heads")
        self.h = cfg.n_heads
        self.dk = d_model // cfg.n_heads
        self.cfg = cfg
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.tag_bias = (
            nn.Parameter(torch.zeros(cfg.n_heads, n_tags, n_tags))
            if (cfg.use_tag_bias and n_tags > 0)
            else None
        )

    def set_topology_prior(self, adjacency: torch.Tensor, strength: float = 1.0) -> None:
        """Inicializa o viés relacional com a topologia da planta.

        ``adjacency`` é ``(n_tags, n_tags)`` com 1 onde as tags estão ligadas no
        P&ID (mesma malha, mesmo equipamento, montante/jusante).
        """
        if self.tag_bias is None:
            raise RuntimeError("viés relacional desativado nesta configuração")
        with torch.no_grad():
            self.tag_bias.copy_(strength * adjacency.to(self.tag_bias).unsqueeze(0))

    def forward(
        self,
        x: torch.Tensor,
        tag_index: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(B, S, self.h, self.dk).transpose(1, 2) for t in (q, k, v))

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)  # (B,h,S,S)
        if self.tag_bias is not None and tag_index is not None:
            n = self.tag_bias.shape[-1]
            pair = tag_index.unsqueeze(-1) * n + tag_index.unsqueeze(1)  # (B,S,S)
            bias = self.tag_bias.reshape(self.h, n * n)[:, pair]         # (h,B,S,S)
            scores = scores + bias.permute(1, 0, 2, 3)
        if key_padding_mask is not None:
            neg = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~key_padding_mask[:, None, None, :], neg)
        if self.cfg.causal:
            causal = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~causal, torch.finfo(scores.dtype).min)

        attn = self.drop(scores.softmax(dim=-1))
        y = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.out(y), (attn if need_weights else None)
