"""Camada ANFIS — o "MLP" do FT-IC.

Onde um transformer de linguagem aplica um MLP ponto-a-ponto, o FT-IC aplica um
sistema de inferência fuzzy Takagi-Sugeno-Kang (TSK) diferenciável. A motivação
é epistemológica antes de ser numérica: o conhecimento de processo que queremos
armazenar nos pesos *já tem forma de regra* ("se a válvula está saturando e a
temperatura está subindo, então ..."), e é assim que ele aparece nos HAZOPs,
nas matrizes de causa-e-efeito e nos relatórios de turno.

Estrutura (as cinco camadas clássicas do ANFIS, vetorizadas):

1. **Fuzzificação latente** — o embedding ``h`` é projetado em ``A`` eixos
   antecedentes; cada eixo tem ``R`` MFs gaussianas com centro e largura
   aprendidos ⇒ ``μ[a, r]``.
2. **Disparo das regras** — em vez da grade completa ``R^A`` (inviável), o
   banco tem ``K`` regras aprendidas: cada regra escolhe, por eixo, uma
   distribuição ``p[k, a, ·]`` sobre as MFs (softmax de logits). A força de
   disparo é o produto t-norma ``w_k = Π_a Σ_r p[k,a,r]·μ[a,r]``, calculada em
   log-espaço. Com ``hard_rules=True`` a escolha vira discreta (Gumbel-softmax
   no treino, argmax na inferência) e cada regra torna-se literalmente legível.
3. **Normalização** — ``w̄ = softmax_k(log w_k / τ)``.
4. **Consequentes** — TSK-0 (vetor por regra) ou TSK-1 de posto baixo
   (transformação linear por regra sobre o embedding).
5. **Agregação** — soma ponderada por ``w̄``, concatenação das cabeças,
   projeção de saída.

As forças de disparo são expostas em ``AnfisTrace`` porque são a principal
alavanca de interpretabilidade do modelo: dizem *qual regra* explicou cada
decisão, e é sobre elas que o estágio Run ancora as regras documentais.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AnfisConfig


@dataclass
class AnfisTrace:
    """Diagnóstico de uma passagem pela camada (interpretabilidade)."""

    firing: torch.Tensor        # (B, S, H, K) forças normalizadas
    log_firing: torch.Tensor    # (B, S, H, K) antes da normalização
    axes: torch.Tensor          # (B, S, H, A) valores nos eixos antecedentes


class AnfisLayer(nn.Module):
    def __init__(self, d_model: int, cfg: AnfisConfig) -> None:
        super().__init__()
        if d_model % cfg.n_heads:
            raise ValueError("d_model precisa ser divisível por anfis.n_heads")
        self.cfg = cfg
        H, A, R, K = cfg.n_heads, cfg.n_axes, cfg.n_mfs, cfg.n_rules
        self.dh = d_model // H

        # 1) projeção para os eixos antecedentes
        self.axis_proj = nn.Linear(d_model, H * A)
        self.axis_norm = nn.LayerNorm(A, eps=1e-5)

        # 1) MFs gaussianas por (cabeça, eixo, termo)
        centers = torch.linspace(-1.5, 1.5, R).view(1, 1, R).repeat(H, A, 1)
        self.mf_center = nn.Parameter(centers)
        self.mf_log_sigma = nn.Parameter(torch.full((H, A, R), -0.35))

        # 2) banco de regras: logits de seleção de MF por eixo
        self.rule_logits = nn.Parameter(torch.randn(H, K, A, R) * 0.5)

        # 4) consequentes
        self.basis = nn.Parameter(torch.randn(H, d_model, cfg.rank) * d_model**-0.5)
        if cfg.consequent == "tsk1":
            self.cons_w = nn.Parameter(torch.randn(H, K, self.dh, cfg.rank) * cfg.rank**-0.5)
        elif cfg.consequent != "tsk0":
            raise ValueError("consequent deve ser 'tsk0' ou 'tsk1'")
        self.cons_b = nn.Parameter(torch.zeros(H, K, self.dh))

        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(cfg.dropout)

    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor, return_trace: bool = False
    ) -> tuple[torch.Tensor, Optional[AnfisTrace]]:
        B, S, D = x.shape
        cfg = self.cfg
        H, A, R, K = cfg.n_heads, cfg.n_axes, cfg.n_mfs, cfg.n_rules

        # --- 1) fuzzificação latente ---------------------------------
        z = self.axis_proj(x).view(B, S, H, A)
        z = self.axis_norm(z)
        sigma = self.mf_log_sigma.exp().clamp(min=1e-3)          # (H,A,R)
        log_mu = -0.5 * ((z.unsqueeze(-1) - self.mf_center) / sigma) ** 2  # (B,S,H,A,R)

        # --- 2) disparo das regras (t-norma produto, estabilizada) ---
        # Σ_r p[k,a,r]·μ[a,r] é feito como contração; o log só é tomado depois,
        # com o máximo por eixo removido antes para evitar underflow ao
        # multiplicar A fatores pequenos.
        mmax = log_mu.amax(dim=-1, keepdim=True)                   # (B,S,H,A,1)
        mu_shift = (log_mu - mmax).exp()                           # (B,S,H,A,R)
        p_rule = self._rule_weights()                              # (H,K,A,R)
        mixed = torch.einsum("hkar,bshar->bshka", p_rule, mu_shift)
        per_axis = (mixed.clamp_min(1e-12)).log() + mmax.squeeze(-1).unsqueeze(3)
        log_w = per_axis.sum(dim=-1)                               # (B,S,H,K)

        # --- 3) normalização -----------------------------------------
        firing = torch.softmax(log_w / max(cfg.firing_temperature, 1e-3), dim=-1)
        firing = self.drop(firing)

        # --- 4/5) consequentes e agregação ---------------------------
        y = torch.einsum("bsd,hdr->bshr", x, self.basis)           # base compartilhada
        if cfg.consequent == "tsk1":
            # contrai firing×base antes de tocar em d_head: evita materializar
            # o tensor por regra (B,S,H,K,d_head).
            yk = firing.unsqueeze(-1) * y.unsqueeze(-2)            # (B,S,H,K,rank)
            agg = torch.einsum("bshkr,hkcr->bshc", yk, self.cons_w)
            agg = agg + torch.einsum("bshk,hkc->bshc", firing, self.cons_b)
        else:
            agg = torch.einsum("bshk,hkc->bshc", firing, self.cons_b)
        out = self.out(agg.reshape(B, S, D))

        trace = AnfisTrace(firing=firing, log_firing=log_w, axes=z) if return_trace else None
        return out, trace

    # ------------------------------------------------------------------
    def _rule_weights(self) -> torch.Tensor:
        """p[k, a, r] — distribuição de cada regra sobre as MFs de cada eixo."""
        if self.cfg.hard_rules:
            if self.training:
                return F.gumbel_softmax(self.rule_logits, tau=1.0, hard=True, dim=-1)
            return F.one_hot(self.rule_logits.argmax(dim=-1), self.cfg.n_mfs).to(
                self.rule_logits.dtype
            )
        return F.softmax(self.rule_logits, dim=-1)

    # ------------------------------------------------------------------
    def rule_entropy(self) -> torch.Tensor:
        """Entropia média da seleção de MFs — regularizador de legibilidade.

        Minimizá-la empurra cada regra para escolher *um* termo por eixo, isto é,
        para virar uma regra de fato ("SE eixo_3 é alto E eixo_7 é baixo ...").
        """
        log_p = F.log_softmax(self.rule_logits, dim=-1)
        return -(log_p.exp() * log_p).sum(dim=-1).mean()

    @torch.no_grad()
    def rule_table(self) -> torch.Tensor:
        """Termo dominante de cada regra por eixo: ``(H, K, A)`` de índices."""
        return self.rule_logits.argmax(dim=-1)
