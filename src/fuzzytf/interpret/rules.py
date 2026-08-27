"""Interpretabilidade: ler o que o modelo aprendeu.

A justificativa central para usar ANFIS no lugar de um MLP é que o conhecimento
armazenado deve poder ser *auditado por um engenheiro de processo*. Isso só se
sustenta se houver ferramentas para extrair:

* qual regra disparou em cada decisão (``firing``);
* sobre quais tags/instantes ela dispara (atribuição);
* como o antecedente da regra se lê ("eixo 3 é alto E eixo 7 é baixo");
* o que a atenção olhou antes de decidir.

Os "eixos" latentes não são, por construção, variáveis de processo — são
projeções aprendidas. A ponte entre eixo latente e grandeza física é feita por
atribuição empírica (quais tokens maximizam cada eixo), e é uma das questões de
pesquisa em aberto (QP-7).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..data.dataset import DatasetConfig, collate
from ..data.simulator import ADVISORIES, FAULTS
from ..fuzzy import VariableBook

MF_LABELS = {
    2: ("baixo", "alto"),
    3: ("baixo", "medio", "alto"),
    4: ("muito_baixo", "baixo", "alto", "muito_alto"),
    5: ("muito_baixo", "baixo", "medio", "alto", "muito_alto"),
}


def mf_label(idx: int, n_mfs: int) -> str:
    return MF_LABELS.get(n_mfs, tuple(f"t{i}" for i in range(n_mfs)))[idx]


@torch.no_grad()
def explain_sample(
    model,
    book: VariableBook,
    sample: Dict[str, np.ndarray],
    cfg: Optional[DatasetConfig] = None,
    advisories: Sequence[str] = ADVISORIES,
    faults: Sequence[str] = FAULTS,
) -> str:
    """Relatório textual de uma única decisão do modelo.

    ``advisories``/``faults`` são os vocabulários de saída; o padrão é o do
    simulador, e o estudo de caso passa o seu (``u200.ADVISORIES_U200``).
    """
    cfg = cfg or DatasetConfig()
    model.eval()
    batch = collate([sample])
    out = model(batch, return_trace=True)

    lines: List[str] = ["=== Estado observado (instante k) ==="]
    slots = book.term_slots()
    lag0 = np.where(sample["lag"] == 0)[0]
    for row in lag0:
        pairs = [
            (slots[int(s)][1], slots[int(s)][2], float(w))
            for s, w, m in zip(sample["slot_ids"][row], sample["weights"][row], sample["mask"][row])
            if m and w > 0.05
        ]
        pairs.sort(key=lambda p: -p[2])
        desc = ", ".join(f"{w * 100:.0f}% {term} ({dim})" for dim, term, w in pairs[:4])
        lines.append(f"  {book.tags[int(sample['tag_index'][row])]:8s}: {desc}")

    delta = float(out.control.delta[0, 0])
    lo, hi = float(out.control.band_lo[0, 0]), float(out.control.band_hi[0, 0])
    u_prev = float(sample.get("u_prev", 0.0))
    lines += [
        "",
        "=== Ação de controle ===",
        f"  Δu previsto      : {delta:+.3f} (={delta * cfg.delta_scale * 100:+.2f}% de curso)",
        f"  Δu antes do filtro: {float(out.control.delta_raw[0, 0]):+.3f}",
        f"  envelope admissível: [{lo:+.3f}, {hi:+.3f}] "
        f"⇒ comando em [{u_prev + lo * cfg.delta_scale:.3f}, {u_prev + hi * cfg.delta_scale:.3f}]",
        f"  alvo do professor : {float(sample['target_delta']):+.3f}",
        "",
        "  distribuição linguística da ação:",
    ]
    terms = out.control.term_logits[0, 0].softmax(-1).cpu().numpy()
    names = _action_term_names(len(terms))
    for i in np.argsort(-terms)[:3]:
        lines.append(f"    {terms[i] * 100:5.1f}% {names[i]}")

    if out.fault_logits is not None:
        p = out.fault_logits[0].softmax(-1).cpu().numpy()
        lines += ["", "=== Diagnóstico ==="]
        for i in np.argsort(-p)[:3]:
            lines.append(f"  {p[i] * 100:5.1f}% {_name(faults, i)}")
        lines.append(f"  (verdade: {_name(faults, int(sample['target_fault']))})")

    if out.advisory is not None:
        p = torch.sigmoid(out.advisory.logits[0]).cpu().numpy()
        lines += ["", "=== Orientações às equipes ==="]
        for i in np.argsort(-p):
            if p[i] < 0.5:
                continue
            tag = _pointer_tag(out, i, sample, book)
            alvo = "✓" if sample["target_advisory"][i] > 0.5 else "✗"
            lines.append(f"  [{alvo}] {p[i] * 100:5.1f}% {_name(advisories, i)}  → {tag}")
        if all(p < 0.5):
            lines.append("  (nenhuma orientação acima do limiar)")

    if out.traces:
        lines += ["", "=== Regras dominantes por bloco ==="]
        for b, tr in enumerate(out.traces):
            if tr.anfis is None:
                lines.append(f"  bloco {b}: (mixer MLP — sem regras)")
                continue
            firing = tr.anfis.firing[0].mean(dim=0)          # (H, K)
            for h in range(firing.shape[0]):
                k = int(firing[h].argmax())
                lines.append(
                    f"  bloco {b} cabeça {h}: regra #{k} (força média {float(firing[h, k]):.3f}) "
                    f"| {rule_antecedent(model, b, h, k)}"
                )
    return "\n".join(lines) + "\n"


def rule_antecedent(model, block: int, head: int, rule: int) -> str:
    """Lê o antecedente de uma regra: termo dominante em cada eixo latente."""
    layer = model.blocks[block].anfis
    if not hasattr(layer, "rule_logits"):
        return "(sem banco de regras)"
    logits = layer.rule_logits[head, rule]                    # (A, R)
    p = torch.softmax(logits, dim=-1)
    n_mfs = logits.shape[-1]
    parts = []
    for a in range(logits.shape[0]):
        idx = int(p[a].argmax())
        conf = float(p[a, idx])
        if conf < 0.5:            # eixo indiferente: não entra na leitura
            continue
        parts.append(f"eixo{a} é {mf_label(idx, n_mfs)}[{conf:.2f}]")
    return "SE " + " E ".join(parts) if parts else "SE (antecedente difuso)"


@torch.no_grad()
def top_rules(model, dataset, cfg: Optional[DatasetConfig] = None, n: int = 10, n_batches: int = 8) -> str:
    """Regras mais ativas do banco, com as tags que mais as disparam."""
    cfg = cfg or DatasetConfig()
    model.eval()
    book = dataset.book
    acc: Dict[tuple, float] = {}
    tag_acc: Dict[tuple, np.ndarray] = {}
    n_seen = 0
    for i in range(n_batches):
        items = [dataset[(i * 37 + j) % len(dataset)] for j in range(8)]
        batch = collate(items)
        out = model(batch, return_trace=True)
        n_special = 1 + model.cfg.n_actions
        tag_idx = batch["tag_index"].cpu().numpy()
        for b, tr in enumerate(out.traces or []):
            if tr.anfis is None:
                continue
            f = tr.anfis.firing[:, n_special : n_special + tag_idx.shape[1]].cpu().numpy()
            H, K = f.shape[2], f.shape[3]
            for h in range(H):
                for k in range(K):
                    acc[(b, h, k)] = acc.get((b, h, k), 0.0) + float(f[:, :, h, k].mean())
                    per_tag = np.zeros(len(book))
                    for t in range(len(book)):
                        sel = tag_idx == t
                        if sel.any():
                            per_tag[t] = f[:, :, h, k][sel].mean()
                    key = (b, h, k)
                    tag_acc[key] = tag_acc.get(key, np.zeros(len(book))) + per_tag
        n_seen += 1

    if not acc:
        return "(modelo sem camadas ANFIS — nada a reportar)\n"

    lines = ["Regras mais ativas (média de força de disparo)", "=" * 56]
    for (b, h, k), v in sorted(acc.items(), key=lambda kv: -kv[1])[:n]:
        share = tag_acc[(b, h, k)] / max(tag_acc[(b, h, k)].sum(), 1e-9)
        top_tags = ", ".join(
            f"{book.tags[i]} {share[i] * 100:.0f}%" for i in np.argsort(-share)[:3]
        )
        lines.append(
            f"bloco {b} cabeça {h} regra #{k:3d} | força {v / n_seen:.4f}\n"
            f"    {rule_antecedent(model, b, h, k)}\n"
            f"    dispara sobre: {top_tags}"
        )
    return "\n".join(lines) + "\n"


def _name(names: Sequence[str], i: int) -> str:
    """Nome do rótulo, tolerando vocabulários de tamanho diferente do padrão."""
    return names[i] if 0 <= i < len(names) else f"classe_{i}"


def _action_term_names(n: int) -> Sequence[str]:
    if n == 7:
        return (
            "reduzir_muito",
            "reduzir",
            "reduzir_pouco",
            "manter",
            "aumentar_pouco",
            "aumentar",
            "aumentar_muito",
        )
    return tuple(f"termo_{i}" for i in range(n))


def _pointer_tag(out, advisory: int, sample, book) -> str:
    if out.advisory.pointer is None:
        return "-"
    w = out.advisory.pointer[0, advisory].cpu().numpy()
    n_special = 1 + out.control.delta.shape[1]
    proc = w[n_special : n_special + len(sample["tag_index"])]
    if proc.size == 0:
        return "-"
    row = int(proc.argmax())
    return f"{book.tags[int(sample['tag_index'][row])]}[k-{int(sample['lag'][row])}]"
