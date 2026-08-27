"""Adaptador do case study real.

O estudo de caso ainda não está neste repositório (será adicionado). Este
módulo define **o contrato**: o layout de diretórios e o esquema de colunas que
o restante do código espera, mais um validador que diz exatamente o que falta.

Layout esperado::

    case_study/
      tags.csv                      # dicionário de tags (obrigatório)
      process/*.csv                 # histórico de processo (obrigatório)
      events/work_orders.csv        # ordens de trabalho (WOs)
      events/production_orders.csv  # ordens de produção (POs)
      events/shift_reports/*.md     # relatórios de turno
      events/interventions.csv      # ações do operador (alvo supervisionado)
      knowledge/hazop.csv           # HAZOP / bowtie / matriz de causa-e-efeito
      knowledge/topology.json       # adjacência do P&ID
      knowledge/manuals/*           # documentação de fabricante

Esquemas mínimos (nomes de coluna, em qualquer ordem)::

    tags.csv          tag,kind,unit,lo,hi,role,equipment,alarm_lo,alarm_hi
    process/*.csv     timestamp,<tag1>,<tag2>,...        (formato largo)
                      ou timestamp,tag,value             (formato longo)
    work_orders.csv   wo_id,opened_at,closed_at,equipment,tag,type,priority,description
    production_orders.csv po_id,start,end,product,rate_target,notes
    interventions.csv timestamp,tag,action,band_lo,band_hi,advisories
    hazop.csv         node,deviation,cause,consequence,safeguard,recommendation,tags

Tudo é lido com a biblioteca padrão (``csv``/``json``) — sem pandas — para que
o adaptador funcione em qualquer ambiente e para que os erros de esquema
apareçam como mensagens claras, não como exceções de terceiros.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..fuzzy.variables import (
    DEFAULT_LEVEL_TERMS,
    TagUniverse,
    VariableBook,
    partition_from_breakpoints,
    ruspini_partition,
    standard_tag,
)

#: Variável de ambiente que aponta para a raiz do case study.
ENV_VAR = "FTIC_CASE_STUDY"

REQUIRED_TAG_COLUMNS = ("tag", "kind", "lo", "hi")
WO_COLUMNS = ("wo_id", "opened_at", "equipment")
PO_COLUMNS = ("po_id", "start")
HAZOP_COLUMNS = ("deviation", "cause", "consequence")


def case_study_root(path: Optional[str | Path] = None) -> Optional[Path]:
    """Resolve a raiz do case study (argumento > variável de ambiente > ./case_study)."""
    for candidate in (path, os.environ.get(ENV_VAR), "case_study", "../case_study"):
        if candidate:
            p = Path(candidate).expanduser()
            if p.exists():
                return p
    return None


# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    root: Optional[Path]
    found: Dict[str, str] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.root is not None and not self.missing and not self.problems

    def __str__(self) -> str:
        if self.root is None:
            return (
                "case study não encontrado.\n"
                f"  defina {ENV_VAR}=/caminho/para/case_study ou passe --path\n"
            )
        lines = [f"case study: {self.root}", ""]
        for k, v in self.found.items():
            lines.append(f"  [ok]     {k}: {v}")
        for k in self.missing:
            lines.append(f"  [FALTA]  {k}")
        for p in self.problems:
            lines.append(f"  [ERRO]   {p}")
        lines.append("")
        lines.append("pronto para o estágio Walk" if self.ok else "faltam artefatos (ver acima)")
        return "\n".join(lines)


def validate(path: Optional[str | Path] = None) -> ValidationReport:
    """Verifica o layout e os esquemas do case study, sem carregar tudo."""
    root = case_study_root(path)
    rep = ValidationReport(root=root)
    if root is None:
        return rep

    checks = [
        ("tags.csv", root / "tags.csv", REQUIRED_TAG_COLUMNS),
        ("events/work_orders.csv", root / "events/work_orders.csv", WO_COLUMNS),
        ("events/production_orders.csv", root / "events/production_orders.csv", PO_COLUMNS),
        ("knowledge/hazop.csv", root / "knowledge/hazop.csv", HAZOP_COLUMNS),
    ]
    for name, p, cols in checks:
        if not p.exists():
            rep.missing.append(name)
            continue
        header = _read_header(p)
        absent = [c for c in cols if c not in header]
        if absent:
            rep.problems.append(f"{name}: faltam colunas {absent} (encontradas: {header})")
        else:
            rep.found[name] = f"{len(header)} colunas"

    for name, pattern in [
        ("process/*.csv", "process/*.csv"),
        ("events/shift_reports/*", "events/shift_reports/*"),
        ("knowledge/manuals/*", "knowledge/manuals/*"),
    ]:
        files = sorted(root.glob(pattern))
        if files:
            rep.found[name] = f"{len(files)} arquivo(s)"
        else:
            rep.missing.append(name)

    topo = root / "knowledge/topology.json"
    if topo.exists():
        try:
            json.loads(topo.read_text())
            rep.found["knowledge/topology.json"] = "json válido"
        except json.JSONDecodeError as exc:
            rep.problems.append(f"topology.json inválido: {exc}")
    else:
        rep.missing.append("knowledge/topology.json")
    return rep


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------

def load_variable_book(
    path: Optional[str | Path] = None,
    process: Optional[Dict[str, np.ndarray]] = None,
    quantile_partition: bool = True,
) -> VariableBook:
    """Constrói o vocabulário fuzzy a partir de ``tags.csv``.

    Com ``quantile_partition=True`` e o histórico em ``process``, os centros dos
    termos vêm dos quantis observados em vez da faixa de engenharia. É quase
    sempre o que se quer: a faixa do instrumento (0–100%) costuma ser muito mais
    larga que a faixa realmente operada, e uma partição uniforme na faixa do
    instrumento desperdiça quase todos os termos.
    """
    root = case_study_root(path)
    if root is None:
        raise FileNotFoundError("case study não encontrado; ver case_study_root()")
    rows = _read_csv(root / "tags.csv")
    universes: List[TagUniverse] = []
    for r in rows:
        lo, hi = float(r["lo"]), float(r["hi"])
        u = standard_tag(
            r["tag"],
            r.get("kind", "measurement"),
            lo,
            hi,
            unit=r.get("unit", ""),
            role=r.get("role", "measurement"),
            equipment=r.get("equipment") or None,
            alarm_lo=_maybe_float(r.get("alarm_lo")),
            alarm_hi=_maybe_float(r.get("alarm_hi")),
        )
        if quantile_partition and process is not None and r["tag"] in process:
            x = np.asarray(process[r["tag"]], dtype=float)
            x = x[np.isfinite(x)]
            if x.size > 50:
                qs = np.linspace(2, 98, len(DEFAULT_LEVEL_TERMS))
                centers = np.percentile(x, qs)
                centers = _make_increasing(centers)
                u.add(partition_from_breakpoints(centers, DEFAULT_LEVEL_TERMS, "level"))
                d = np.diff(x)
                span = float(max(np.percentile(np.abs(d), 95), 1e-6))
                u.add(ruspini_partition(-span, span, u.variables["trend"].terms, "trend"))
        universes.append(u)
    return VariableBook(universes)


def load_process(
    path: Optional[str | Path] = None, tags: Optional[Sequence[str]] = None
) -> Dict[str, np.ndarray]:
    """Lê ``process/*.csv`` (formato largo ou longo) em ``{tag: série}``."""
    root = case_study_root(path)
    if root is None:
        raise FileNotFoundError("case study não encontrado")
    series: Dict[str, List[float]] = {}
    for f in sorted((root / "process").glob("*.csv")):
        rows = _read_csv(f)
        if not rows:
            continue
        header = list(rows[0].keys())
        if {"tag", "value"} <= set(header):          # formato longo
            for r in rows:
                series.setdefault(r["tag"], []).append(_maybe_float(r["value"], np.nan))
        else:                                        # formato largo
            for col in header:
                if col.lower() in ("timestamp", "time", "ts", "datetime"):
                    continue
                series.setdefault(col, []).extend(_maybe_float(r[col], np.nan) for r in rows)
    out = {k: np.asarray(v, dtype=float) for k, v in series.items()}
    return {k: v for k, v in out.items() if tags is None or k in tags} if tags else out


def load_events(path: Optional[str | Path] = None) -> Dict[str, List[dict]]:
    """Lê WOs, POs, intervenções e HAZOP como listas de dicionários."""
    root = case_study_root(path)
    if root is None:
        raise FileNotFoundError("case study não encontrado")
    files = {
        "work_orders": root / "events/work_orders.csv",
        "production_orders": root / "events/production_orders.csv",
        "interventions": root / "events/interventions.csv",
        "hazop": root / "knowledge/hazop.csv",
    }
    return {k: (_read_csv(p) if p.exists() else []) for k, p in files.items()}


def load_topology(path: Optional[str | Path] = None, book: Optional[VariableBook] = None) -> Optional[np.ndarray]:
    """Adjacência do P&ID como matriz ``(n_tags, n_tags)``.

    Formato de ``topology.json``: ``{"edges": [["T-102","V-097"], ...]}`` ou
    ``{"loops": {"TIC-102": ["T-102","V-097"]}}``.
    """
    root = case_study_root(path)
    if root is None or book is None:
        return None
    f = root / "knowledge/topology.json"
    if not f.exists():
        return None
    spec = json.loads(f.read_text())
    n = len(book)
    a = np.eye(n, dtype=np.float32)

    def link(x: str, y: str) -> None:
        if x in book and y in book:
            i, j = book.index(x), book.index(y)
            a[i, j] = a[j, i] = 1.0

    for edge in spec.get("edges", []):
        link(edge[0], edge[1])
    for members in spec.get("loops", {}).values():
        for i, x in enumerate(members):
            for y in members[i + 1 :]:
                link(x, y)
    return a


# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _read_header(path: Path) -> List[str]:
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return next(csv.reader(fh), [])


def _maybe_float(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _make_increasing(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    for i in range(1, len(x)):
        if x[i] <= x[i - 1]:
            x[i] = x[i - 1] + eps
    return x
