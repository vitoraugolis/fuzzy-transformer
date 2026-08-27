"""Perdas multitarefa do FT-IC.

O modelo é treinado em várias tarefas ao mesmo tempo porque nenhuma delas,
sozinha, obriga a rede a aprender o que queremos:

* **ação** — imitação do controlador de referência (professor);
* **envelope** — a faixa de contingência (o "filtro inteligente" da ação);
* **orientações** — o que os times precisam fazer;
* **diagnóstico** — qual falha está ativa (auxiliar, dá sinal denso e precoce);
* **previsão fuzzy** — prever o estado linguístico em k+1 (auto-supervisão);
* **estado mascarado** — reconstruir tokens de estado ocultados (auto-supervisão,
  análoga ao MLM: é o que permite pré-treinar em histórico sem rótulo).

O regularizador de entropia de regras não é uma perda de desempenho, e sim de
*legibilidade*: empurra o banco de regras para escolhas nítidas, de modo que
cada regra possa ser lida como uma regra de engenharia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class LossTerms:
    total: torch.Tensor
    parts: Dict[str, float]


def soft_cross_entropy(logits: torch.Tensor, slots: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """CE com alvos suaves esparsos (graus de pertinência sobre slots)."""
    logp = F.log_softmax(logits, dim=-1)
    picked = logp.gather(-1, slots)
    return -(picked * weights).sum(dim=-1).mean()


def band_loss(lo: torch.Tensor, hi: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Erro do envelope, penalizando assimetricamente a faixa *larga demais*.

    Prever uma faixa maior que a admissível é um erro de segurança (deixa passar
    uma ação proibida); prever menor é apenas conservador. O peso 2× no excesso
    codifica essa assimetria.
    """
    t_lo, t_hi = target[..., 0], target[..., 1]
    too_wide = F.relu(t_lo - lo) + F.relu(hi - t_hi)
    too_narrow = F.relu(lo - t_lo) + F.relu(t_hi - hi)
    return (2.0 * too_wide + too_narrow).mean()


def control_loss(out, batch, huber_beta: float = 0.2) -> torch.Tensor:
    target = batch["target_delta"].unsqueeze(-1)
    raw = F.smooth_l1_loss(out.control.delta_raw, target, beta=huber_beta)
    clipped = F.smooth_l1_loss(
        out.control.delta,
        target.clamp(min=batch["target_band"][:, :1], max=batch["target_band"][:, 1:]),
        beta=huber_beta,
    )
    return raw + clipped


def total_loss(
    model,
    out,
    batch: Dict[str, torch.Tensor],
    weights,
    advisory_pos_weight: Optional[torch.Tensor] = None,
) -> LossTerms:
    parts: Dict[str, float] = {}

    loss = weights.w_action * control_loss(out, batch)
    parts["action"] = float(loss.detach())

    lb = band_loss(out.control.band_lo, out.control.band_hi, batch["target_band"])
    loss = loss + weights.w_action * lb
    parts["band"] = float(lb.detach())

    if out.advisory is not None and "target_advisory" in batch:
        la = F.binary_cross_entropy_with_logits(
            out.advisory.logits, batch["target_advisory"], pos_weight=advisory_pos_weight
        )
        loss = loss + weights.w_advisory * la
        parts["advisory"] = float(la.detach())

    if out.fault_logits is not None and "target_fault" in batch:
        lf = F.cross_entropy(out.fault_logits, batch["target_fault"])
        loss = loss + weights.w_advisory * lf
        parts["fault"] = float(lf.detach())

    if weights.w_forecast > 0 and "forecast_slots" in batch:
        pos = batch["forecast_pos"]
        logits = out.state_logits.gather(
            1, pos.unsqueeze(-1).expand(-1, -1, out.state_logits.shape[-1])
        )
        lfc = soft_cross_entropy(logits, batch["forecast_slots"], batch["forecast_weights"])
        loss = loss + weights.w_forecast * lfc
        parts["forecast"] = float(lfc.detach())

    if weights.w_masked_state > 0 and "masked_pos" in batch:
        pos = batch["masked_pos"]
        logits = out.state_logits.gather(
            1, pos.unsqueeze(-1).expand(-1, -1, out.state_logits.shape[-1])
        )
        lm = soft_cross_entropy(logits, batch["masked_slots"], batch["masked_weights"])
        loss = loss + weights.w_masked_state * lm
        parts["masked_state"] = float(lm.detach())

    if weights.w_rule_entropy > 0:
        le = model.rule_entropy()
        loss = loss + weights.w_rule_entropy * le
        parts["rule_entropy"] = float(le.detach())

    parts["total"] = float(loss.detach())
    return LossTerms(total=loss, parts=parts)


def mask_state_tokens(
    batch: Dict[str, torch.Tensor],
    p: float = 0.15,
    n_max: int = 8,
    protect_current: bool = True,
):
    """Oculta tokens de estado e devolve o batch aumentado com os alvos.

    O token mascarado mantém a identidade da tag e a posição temporal — some
    apenas o *estado*. Isso força o modelo a inferir "o que a T-102 deveria
    estar marcando" a partir das demais tags, que é exatamente a competência
    exigida para detectar instrumento em falha.

    ``protect_current`` mantém intactos os tokens do instante ``k``. Sem isso, a
    máscara compete diretamente com a tarefa de controle: a ação depende da
    diferença entre ``k`` e ``k-1``, e esconder o estado atual em 15% dos passos
    apaga justamente o sinal que se quer aprender — na prática, o modelo fica
    preso no preditor trivial. A reconstrução continua sendo tarefa cheia sobre
    todo o passado da janela.
    """
    b = dict(batch)
    B, S, P = batch["slot_ids"].shape
    n = max(1, min(n_max, int(S * p)))
    dev = batch["slot_ids"].device
    if protect_current and "lag" in batch:
        # sorteia apenas entre os tokens de lag > 0
        score = torch.rand(B, S, device=dev).masked_fill(batch["lag"] == 0, -1.0)
        pos = score.topk(min(n, S), dim=1).indices
    else:
        pos = torch.stack([torch.randperm(S, device=dev)[:n] for _ in range(B)])
    idx = pos.unsqueeze(-1).expand(-1, -1, P)

    b["masked_pos"] = pos
    b["masked_slots"] = batch["slot_ids"].gather(1, idx)
    b["masked_weights"] = batch["weights"].gather(1, idx) * batch["mask"].gather(1, idx)

    weights = batch["weights"].clone()
    mask = batch["mask"].clone()
    weights.scatter_(1, idx, torch.zeros_like(b["masked_slots"], dtype=weights.dtype))
    mask.scatter_(1, idx, torch.zeros_like(b["masked_slots"], dtype=torch.bool))
    b["weights"], b["mask"] = weights, mask
    return b
