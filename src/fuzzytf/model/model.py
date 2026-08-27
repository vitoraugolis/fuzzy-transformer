"""FT-IC — Fuzzy Transformer for Industrial Control.

Montagem completa: embedding fuzzy → N × (self-attention + ANFIS) → cabeças.

Além dos tokens de processo, a sequência recebe tokens especiais:

* ``[CLS]``   — leitura global (orientações, diagnóstico);
* ``[ACT_i]`` — uma consulta por variável manipulada; o estado final desse token
  é o que a cabeça de controle lê;
* tokens de contexto documental (WO/PO/HAZOP/manual), quando fornecidos — é o
  ponto de entrada do conhecimento textual nos estágios Walk e Run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
import torch.nn as nn

from ..config import ModelConfig, TokenizerConfig
from ..fuzzy import VariableBook
from .block import BlockTrace, FuzzyTransformerBlock
from .embedding import TokenEmbedding
from .heads import AdvisoryHead, AdvisoryOutput, ControlHead, ControlOutput, StateHead


@dataclass
class ModelOutput:
    control: ControlOutput
    advisory: Optional[AdvisoryOutput]
    state_logits: torch.Tensor          # (B, S_proc, n_slots+1)
    fault_logits: Optional[torch.Tensor]  # (B, n_fault_classes)
    hidden: torch.Tensor                # (B, S_total, d)
    traces: Optional[List[BlockTrace]]


class FTIC(nn.Module):
    def __init__(
        self,
        book: VariableBook,
        cfg: Optional[ModelConfig] = None,
        tok: Optional[TokenizerConfig] = None,
        action_tags: Sequence[str] = (),
    ) -> None:
        super().__init__()
        cfg = cfg or ModelConfig()
        tok = tok or TokenizerConfig()
        if action_tags:
            cfg.n_actions = len(action_tags)
        self.cfg, self.tok_cfg = cfg, tok
        self.book_fingerprint = book.fingerprint()
        self.action_tags = list(action_tags)
        self.action_index = [book.index(t) for t in self.action_tags]

        n_tags = len(book)
        self.n_tags = n_tags
        self.cls_slot = n_tags
        self.act_slot0 = n_tags + 1
        self.ctx_slot = n_tags + 1 + cfg.n_actions
        n_tag_slots = self.ctx_slot + 1

        self.embed = TokenEmbedding(n_tag_slots, book.n_state_slots, cfg, tok)
        self.blocks = nn.ModuleList(
            FuzzyTransformerBlock(cfg, n_tags=n_tag_slots) for _ in range(cfg.n_blocks)
        )
        self.norm_out = nn.LayerNorm(cfg.d_model, eps=cfg.norm_eps)

        self.control_head = ControlHead(cfg.d_model, cfg.n_actions, cfg.action_terms)
        self.advisory_head = (
            AdvisoryHead(cfg.d_model, cfg.n_advisories) if cfg.n_advisories > 0 else None
        )
        self.state_head = StateHead(self.embed.state_emb)
        self.fault_head = (
            nn.Linear(cfg.d_model, cfg.n_fault_classes) if cfg.n_fault_classes > 0 else None
        )
        self._scale_residual_branches()

    # ------------------------------------------------------------------
    def _scale_residual_branches(self) -> None:
        """Reduz a projeção de saída de cada ramo residual por 1/√(2N).

        Sem isso, a variância do fluxo residual cresce com a profundidade e a
        informação específica de cada amostra some diante do que os blocos
        acrescentam: medido aqui, um modelo de 3 blocos ficava preso no
        preditor trivial enquanto o de 2 blocos aprendia normalmente. É o mesmo
        ajuste de inicialização usado nos transformers profundos, e aqui ele é
        necessário mais cedo porque a camada ANFIS soma K consequentes.
        """
        scale = (2.0 * max(self.cfg.n_blocks, 1)) ** -0.5
        with torch.no_grad():
            for blk in self.blocks:
                blk.attn.out.weight.mul_(scale)
                mixer = blk.anfis
                out = getattr(mixer, "out", None)
                if out is not None:
                    out.weight.mul_(scale)
                elif hasattr(mixer, "net"):
                    mixer.net[-1].weight.mul_(scale)

    # ------------------------------------------------------------------
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        batch: dict,
        context: Optional[torch.Tensor] = None,
        hard_limits: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> ModelOutput:
        h_proc = self.embed(batch)                      # (B, S, d)
        B, S, _ = h_proc.shape
        dev = h_proc.device

        special_tags = torch.tensor(
            [self.cls_slot] + [self.act_slot0 + i for i in range(self.cfg.n_actions)],
            device=dev,
        ).expand(B, -1)
        # Os tokens especiais não são consultas "vazias": [CLS] parte do estado
        # médio do instante k e cada [ACT_i] parte do estado atual da própria
        # variável manipulada. Uma consulta constante teria de reconstruir o
        # instante corrente só pela atenção, e na prática não reconstrói — a
        # média sobre a janela dilui exatamente a diferença entre k e k-1 que a
        # ação depende. Semear o token com o estado de k resolve isso sem abrir
        # mão da atenção, que continua trazendo o contexto.
        h_special = self.embed.tag_emb(special_tags) + self._seed_special(h_proc, batch)
        pieces, tag_ids, valid = [h_special, h_proc], [special_tags, batch["tag_index"]], [
            torch.ones(B, special_tags.shape[1], dtype=torch.bool, device=dev),
            batch["valid"],
        ]
        if context is not None:
            ctx_tags = torch.full(
                (B, context.shape[1]), self.ctx_slot, dtype=torch.long, device=dev
            )
            pieces.append(context + self.embed.tag_emb(ctx_tags))
            tag_ids.append(ctx_tags)
            valid.append(torch.ones(B, context.shape[1], dtype=torch.bool, device=dev))

        x = torch.cat(pieces, dim=1)
        tag_index = torch.cat(tag_ids, dim=1)
        key_padding_mask = torch.cat(valid, dim=1)

        traces: List[BlockTrace] = []
        for blk in self.blocks:
            x, tr = blk(
                x,
                tag_index=tag_index,
                key_padding_mask=key_padding_mask,
                return_trace=return_trace,
            )
            if return_trace:
                traces.append(tr)
        x = self.norm_out(x)

        n_special = 1 + self.cfg.n_actions
        readout = x[:, 1:n_special, :]                       # tokens [ACT_i]
        proc = x[:, n_special : n_special + S, :]            # tokens de processo

        control = self.control_head(readout, hard_limits=hard_limits)
        advisory = (
            self.advisory_head(x, key_padding_mask=key_padding_mask)
            if self.advisory_head is not None
            else None
        )
        return ModelOutput(
            fault_logits=self.fault_head(x[:, 0]) if self.fault_head is not None else None,
            control=control,
            advisory=advisory,
            state_logits=self.state_head(proc),
            hidden=x,
            traces=traces if return_trace else None,
        )

    # ------------------------------------------------------------------
    def _seed_special(self, h_proc: torch.Tensor, batch: dict) -> torch.Tensor:
        """Semeia [CLS] e [ACT_i] com o estado do instante k."""
        lag0 = (batch["lag"] == 0).to(h_proc.dtype)               # (B,S)

        def pooled(weight: torch.Tensor) -> torch.Tensor:
            w = weight / weight.sum(dim=1, keepdim=True).clamp(min=1e-6)
            return torch.einsum("bs,bsd->bd", w, h_proc)

        seeds = [pooled(lag0)]                                    # [CLS]
        for tag_i in self.action_index:
            sel = lag0 * (batch["tag_index"] == tag_i).to(h_proc.dtype)
            seeds.append(pooled(sel) if float(sel.sum()) > 0 else pooled(lag0))
        while len(seeds) < 1 + self.cfg.n_actions:                # sem action_tags
            seeds.append(pooled(lag0))
        return torch.stack(seeds, dim=1)

    # ------------------------------------------------------------------
    def rule_entropy(self) -> torch.Tensor:
        """Média da entropia dos bancos de regras de todos os blocos."""
        vals = [blk.anfis.rule_entropy() for blk in self.blocks]
        return torch.stack(vals).mean()

    def set_topology_prior(self, adjacency: torch.Tensor, strength: float = 1.0) -> None:
        """Injeta a topologia da planta no viés relacional de todos os blocos."""
        n = self.embed.tag_emb.num_embeddings
        full = torch.zeros(n, n)
        m = adjacency.shape[0]
        full[:m, :m] = adjacency
        for blk in self.blocks:
            blk.attn.set_topology_prior(full, strength=strength)
