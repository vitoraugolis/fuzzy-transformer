"""Camada de embedding: da janela fuzzificada para vetores de alta dimensão.

Cada token é a soma de quatro contribuições:

    h = E_id[tag]  +  Σ_p w_p · E_state[slot_p]  +  E_lag[k - t]  +  v · W_val

* ``E_id``    — identidade da tag (T-102 é diferente de P-204);
* ``E_state`` — um vetor por *slot* ``(tag, dimensão, termo)``; a soma ponderada
  pelos graus de pertinência realiza a ideia de "vetor correspondente a 30%
  alto em T-102";
* ``E_lag``   — posição temporal relativa (k, k-1, ...);
* ``W_val``   — canal numérico residual, que devolve a resolução perdida na
  fuzzificação (desligável para estudar o trade-off; ver QP-3).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig, TokenizerConfig


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        n_tags: int,
        n_state_slots: int,
        cfg: ModelConfig,
        tok: TokenizerConfig,
    ) -> None:
        super().__init__()
        d = cfg.d_model
        self.cfg, self.tok = cfg, tok
        self.tag_emb = nn.Embedding(n_tags, d)
        self.state_emb = nn.Embedding(n_state_slots + 1, d, padding_idx=n_state_slots)
        self.pad_slot = n_state_slots
        self.lag_emb = (
            nn.Embedding(tok.window, d) if tok.use_lag_embedding else None
        )
        self.value_proj = nn.Linear(2, d) if tok.use_value_channel else None
        self.norm = nn.LayerNorm(d, eps=cfg.norm_eps)
        self.drop = nn.Dropout(cfg.dropout)
        self._init()

    def _init(self) -> None:
        for emb in (self.tag_emb, self.state_emb, self.lag_emb):
            if emb is not None:
                nn.init.normal_(emb.weight, std=0.02)
        with torch.no_grad():
            self.state_emb.weight[self.pad_slot].zero_()

    def forward(self, batch: dict) -> torch.Tensor:
        """``batch`` traz os tensores produzidos por :class:`ProcessTokenizer`.

        Espera ``tag_index (B,S)``, ``slot_ids (B,S,P)``, ``weights (B,S,P)``,
        ``mask (B,S,P)``, ``lag (B,S)``, ``value (B,S)``, ``valid (B,S)``.
        """
        slot_ids = torch.where(batch["mask"], batch["slot_ids"], self.pad_slot)
        state = self.state_emb(slot_ids)                      # (B,S,P,d)
        w = (batch["weights"] * batch["mask"]).unsqueeze(-1)  # (B,S,P,1)
        h = (state * w).sum(dim=2)                            # (B,S,d)
        h = h + self.tag_emb(batch["tag_index"])
        if self.lag_emb is not None:
            h = h + self.lag_emb(batch["lag"].clamp(max=self.lag_emb.num_embeddings - 1))
        if self.value_proj is not None:
            v = torch.stack(
                [batch["value"], batch["valid"].to(batch["value"].dtype)], dim=-1
            )
            h = h + self.value_proj(v)
        return self.drop(self.norm(h))
