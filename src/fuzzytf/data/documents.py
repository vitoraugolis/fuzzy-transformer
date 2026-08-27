"""Conhecimento textual: WOs, POs, relatórios de turno, manuais, HAZOP.

Este é o módulo que materializa a parte mais ambiciosa da proposta — treinar o
modelo com informação que **não** é dado de processo. A estratégia adotada aqui
tem três camadas, em ordem crescente de dificuldade (e de estágio):

``recuperação`` (Walk)
    Os documentos relevantes ao contexto atual são recuperados e injetados como
    tokens de contexto na sequência. Barato, sem alterar a arquitetura.

``aterramento de regras`` (Walk/Run)
    Linhas de HAZOP e de matrizes de causa-e-efeito são convertidas em regras
    fuzzy candidatas, usadas para *inicializar* e *regularizar* o banco de
    regras da camada ANFIS. O conhecimento entra como viés, não como dado.

``supervisão distante`` (Run)
    WOs e relatórios de turno rotulam retroativamente as janelas de processo:
    se houve uma WO mecânica na V-097 no dia 12, as janelas anteriores a ela
    são exemplos positivos de "acionar manutenção da válvula". É assim que se
    obtém rótulo de orientação em escala, sem anotação manual.

O codificador de texto aqui é um *placeholder* determinístico (hashing +
projeção fixa). Ele existe para que o caminho de contexto do modelo seja
exercitável e testável hoje; a substituição por um codificador treinado é uma
decisão do estágio Walk (ver QP-8).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9\-]+")
TAG_RE = re.compile(r"\b[A-Z]{1,3}-?\d{2,4}\b")


@dataclass
class Document:
    """Um documento (ou trecho) do acervo de conhecimento."""

    doc_id: str
    kind: str                      # wo | po | shift_report | manual | hazop | pid
    text: str
    timestamp: Optional[str] = None
    equipment: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    meta: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tags:
            self.tags = sorted(set(TAG_RE.findall(self.text)))


def documents_from_events(events: Dict[str, List[dict]]) -> List[Document]:
    """Converte os CSVs do case study em :class:`Document`."""
    docs: List[Document] = []
    for row in events.get("work_orders", []):
        text = " ".join(
            str(row.get(k, "")) for k in ("type", "priority", "description", "action_taken")
        )
        docs.append(
            Document(
                doc_id=f"WO:{row.get('wo_id')}",
                kind="wo",
                text=text,
                timestamp=row.get("opened_at"),
                equipment=row.get("equipment"),
                tags=[row["tag"]] if row.get("tag") else [],
                meta={k: str(v) for k, v in row.items()},
            )
        )
    for row in events.get("production_orders", []):
        text = " ".join(str(row.get(k, "")) for k in ("product", "rate_target", "notes"))
        docs.append(
            Document(
                doc_id=f"PO:{row.get('po_id')}",
                kind="po",
                text=text,
                timestamp=row.get("start"),
                meta={k: str(v) for k, v in row.items()},
            )
        )
    for i, row in enumerate(events.get("hazop", [])):
        text = " ".join(
            str(row.get(k, ""))
            for k in ("node", "deviation", "cause", "consequence", "safeguard", "recommendation")
        )
        docs.append(
            Document(
                doc_id=f"HAZOP:{row.get('node', i)}:{i}",
                kind="hazop",
                text=text,
                meta={k: str(v) for k, v in row.items()},
            )
        )
    return docs


def chunk(text: str, size: int = 120, overlap: int = 20) -> List[str]:
    """Quebra texto longo (manuais) em trechos de ``size`` palavras."""
    words = TOKEN_RE.findall(text)
    if not words:
        return []
    step = max(size - overlap, 1)
    return [" ".join(words[i : i + size]) for i in range(0, len(words), step)]


# ---------------------------------------------------------------------------
# Recuperação
# ---------------------------------------------------------------------------

class KeywordRetriever:
    """Recuperação BM25-lite por palavra-chave, com bônus para a tag citada.

    Simples de propósito: no estágio Walk o que importa é medir se *ter*
    contexto documental melhora as orientações — não qual recuperador é o
    melhor. Trocar por embeddings densos é uma otimização posterior.
    """

    def __init__(self, docs: Sequence[Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = list(docs)
        self.k1, self.b = k1, b
        self.tokens = [[t.lower() for t in TOKEN_RE.findall(d.text)] for d in self.docs]
        self.lengths = np.array([max(len(t), 1) for t in self.tokens], dtype=float)
        self.avg_len = float(self.lengths.mean()) if len(self.lengths) else 1.0
        self.df: Dict[str, int] = {}
        for toks in self.tokens:
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.n = max(len(self.docs), 1)

    def search(self, query: str, tags: Sequence[str] = (), top_k: int = 5) -> List[tuple]:
        q = [t.lower() for t in TOKEN_RE.findall(query)]
        scores = np.zeros(len(self.docs))
        for i, toks in enumerate(self.tokens):
            counts: Dict[str, int] = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            s = 0.0
            for t in q:
                f = counts.get(t, 0)
                if not f:
                    continue
                idf = np.log(1 + (self.n - self.df.get(t, 0) + 0.5) / (self.df.get(t, 0) + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                s += idf * f * (self.k1 + 1) / denom
            if tags and set(tags) & set(self.docs[i].tags):
                s += 2.0        # o documento fala da tag em questão
            scores[i] = s
        order = np.argsort(-scores)[:top_k]
        return [(self.docs[i], float(scores[i])) for i in order if scores[i] > 0]


class HashingEncoder:
    """Codificador de texto determinístico (truque do hashing) — placeholder.

    Produz vetores de dimensão ``d`` estáveis entre execuções e independentes de
    treino, para que o caminho de contexto do :class:`FTIC` possa ser exercitado
    e testado antes de existir um codificador de linguagem próprio.
    """

    def __init__(self, d: int = 128, seed: int = 0) -> None:
        self.d = d
        self.seed = seed

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        out = []
        for text in texts:
            v = np.zeros(self.d, dtype=np.float32)
            for tok in TOKEN_RE.findall(text.lower()):
                h = int(hashlib.md5(f"{self.seed}:{tok}".encode()).hexdigest()[:8], 16)
                v[h % self.d] += 1.0 if (h >> 31) & 1 == 0 else -1.0
            norm = np.linalg.norm(v)
            out.append(v / norm if norm > 0 else v)
        return np.stack(out) if out else np.zeros((0, self.d), dtype=np.float32)


# ---------------------------------------------------------------------------
# Aterramento de regras
# ---------------------------------------------------------------------------

@dataclass
class GroundingRule:
    """Regra candidata extraída de documentação (HAZOP, matriz de causa-efeito)."""

    source: str
    antecedent: List[tuple]        # [(tag, dimensão, termo), ...]
    consequent: List[str]          # rótulos de orientação
    note: str = ""


#: Mapeamento de desvios de HAZOP para termos linguísticos.
DEVIATION_TERMS = {
    "mais": ("level", "alto"),
    "more": ("level", "alto"),
    "alta": ("level", "alto"),
    "alto": ("level", "alto"),
    "menos": ("level", "baixo"),
    "less": ("level", "baixo"),
    "baixa": ("level", "baixo"),
    "baixo": ("level", "baixo"),
    "nenhum": ("level", "muito_baixo"),
    "no": ("level", "muito_baixo"),
    "reverso": ("trend", "caindo_rapido"),
}

#: Palavras que indicam qual equipe deve ser acionada.
CONSEQUENT_HINTS = {
    "manutenção": "acionar_manutencao_valvula",
    "manutencao": "acionar_manutencao_valvula",
    "válvula": "acionar_manutencao_valvula",
    "instrumento": "inspecionar_instrumento",
    "transmissor": "inspecionar_instrumento",
    "incrustação": "programar_limpeza_trocador",
    "incrustacao": "programar_limpeza_trocador",
    "limpeza": "programar_limpeza_trocador",
    "carga": "reduzir_carga",
    "processo": "acionar_engenharia_de_processo",
}


def hazop_to_rules(rows: Sequence[dict], known_tags: Sequence[str] = ()) -> List[GroundingRule]:
    """Extrai regras candidatas de linhas de HAZOP.

    A extração é intencionalmente conservadora e auditável: casa o desvio com um
    termo linguístico, a tag citada com uma tag conhecida, e as recomendações
    com o vocabulário de orientações. O que não casar fica de fora — é melhor
    aterrar poucas regras corretas do que muitas duvidosas.
    """
    rules: List[GroundingRule] = []
    known = set(known_tags)
    for i, row in enumerate(rows):
        text = " ".join(str(v) for v in row.values()).lower()
        deviation = str(row.get("deviation", "")).lower()
        term = next(
            ((d, t) for word, (d, t) in DEVIATION_TERMS.items() if word in deviation), None
        )
        if term is None:
            continue
        tags = [t for t in TAG_RE.findall(" ".join(str(v) for v in row.values())) if t in known]
        if not tags:
            continue
        consequent = sorted(
            {label for word, label in CONSEQUENT_HINTS.items() if word in text}
        )
        rules.append(
            GroundingRule(
                source=f"HAZOP:{row.get('node', i)}",
                antecedent=[(tag, term[0], term[1]) for tag in tags],
                consequent=consequent,
                note=str(row.get("recommendation", ""))[:200],
            )
        )
    return rules
