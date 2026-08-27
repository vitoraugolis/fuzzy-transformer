"""Adaptador do estudo de caso U-200 (QUIMIVALE / reator R-201).

O caso é um benchmark multissilo: a válvula FV-201 perde capacidade, o PID
compensa perfeitamente (100% do produto em especificação, IAE baixíssimo) e a
unidade consome toda a reserva de curso **sem que nenhuma métrica de erro
registre nada**. Quando chega uma perturbação de 3 K na carga, a válvula satura
e a unidade perde o controle por 21 h.

É exatamente o caso motivador deste projeto, com uma diferença importante: aqui
a causa raiz **não está escrita em nenhum documento isolado** — exige o
cruzamento de pelo menos três silos (manutenção, engenharia/fabricante e
operação). Ver `docs/08-case-study.md`.

Este módulo faz a ponte entre o acervo do caso e o FT-IC:

* vocabulário fuzzy ancorado nos limites documentados (envelope, TAH, ZAH);
* leitura do historian, alarmes, eventos e LIMS;
* extração de texto dos PDFs dos silos para o acervo documental;
* construção de episódios com alvos de imitação e supervisão distante.
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..fuzzy.variables import (
    DEFAULT_LEVEL_TERMS,
    DEFAULT_TREND_TERMS,
    TagUniverse,
    VariableBook,
    partition_from_breakpoints,
    ruspini_partition,
)
from .documents import Document

# --- constantes do caso (espelham tools/u200_case.py) ----------------------
TF_K = 406.7                 # temperatura de referência da adimensionalização
GAMMA = 20.0
DT_K = TF_K / GAMMA          # 20,335 K por unidade de x2
SP_C = 229.23                # set-point da campanha
ENVELOPE_C = (224.0, 235.0)  # envelope operacional documentado
TAH_C = 235.2                # alarme alto de temperatura
TAHH_C = 244.4               # trip do SIS
TRAVEL_ZAH = 0.80            # ZAH-201: alarme de curso alto de FV-201
FC_DESIGN_M3H = 60.0

#: Vocabulário de orientações do caso U-200.
ADVISORIES_U200 = (
    "acionar_manutencao_fv201",
    "verificar_capacidade_resfriamento",
    "inspecionar_analisador_at205",
    "revisar_temperatura_de_carga",
    "reduzir_carga",
    "abrir_ordem_de_trabalho",
    "registrar_no_relatorio_de_turno",
    "acionar_engenharia_de_processo",
    "operar_em_contingencia",
)


def x2_to_c(x2: float | np.ndarray) -> np.ndarray:
    """Converte a variável adimensional x₂ do modelo para °C."""
    return TF_K * (1.0 + np.asarray(x2, dtype=float) / GAMMA) - 273.15


def c_to_x2(t_c: float | np.ndarray) -> np.ndarray:
    """Converte °C para a variável adimensional x₂."""
    return ((np.asarray(t_c, dtype=float) + 273.15) / TF_K - 1.0) * GAMMA


# ---------------------------------------------------------------------------
# Vocabulário fuzzy
# ---------------------------------------------------------------------------

#: (tag, tipo, unidade, papel, equipamento, faixa, centros de nível, span de trend)
TAG_SPEC = [
    ("TT201_PV_C",    "temperature", "degC", "measurement", "R-201",  (215.0, 250.0),
     [226.0, 228.5, 229.23, 232.0, 235.2], 0.36),
    ("TIC201_SP_C",   "temperature", "degC", "setpoint",    "TIC-201", (215.0, 250.0),
     [226.0, 228.5, 229.23, 232.0, 235.2], 0.30),
    ("TIC201_OUT_pct", "valve",      "%",    "manipulated", "FV-201", (0.0, 100.0),
     [30.0, 55.0, 70.0, 80.0, 95.0], 1.00),
    ("ZT201_pct",     "valve",       "%",    "measurement", "FV-201", (0.0, 100.0),
     [30.0, 55.0, 70.0, 80.0, 95.0], 0.50),
    ("FT201_m3h",     "flow",        "m3/h", "measurement", "E-201",  (0.0, 40.0),
     [14.0, 17.0, 20.0, 23.0, 26.0], 0.70),
    ("TT202_C",       "temperature", "degC", "measurement", "E-201",  (100.0, 160.0),
     [125.0, 127.0, 129.0, 132.0, 135.0], 0.20),
    ("TT203_C",       "temperature", "degC", "measurement", "TQ-utilidades", (0.0, 60.0),
     [26.0, 28.0, 30.5, 33.0, 35.5], 0.60),
    ("TT204_C",       "temperature", "degC", "measurement", "R-201",  (100.0, 160.0),
     [133.4, 133.6, 134.5, 135.8, 136.8], 0.05),
    ("TT207_C",       "temperature", "degC", "measurement", "E-201",  (60.0, 120.0),
     [88.0, 90.0, 92.0, 94.5, 96.5], 0.60),
    ("AT205_X",       "composition", "-",    "measurement", "R-201",  (0.60, 0.90),
     [0.755, 0.762, 0.768, 0.785, 0.805], 0.012),
    ("FT204_m3h",     "flow",        "m3/h", "measurement", "R-201",  (0.0, 15.0),
     [8.90, 8.96, 9.00, 9.04, 9.10], 0.010),
]

#: Tag manipulada (saída do controlador).
ACTION_TAG = "TIC201_OUT_pct"


def variable_book(tags: Optional[Sequence[str]] = None) -> VariableBook:
    """Vocabulário fuzzy do U-200, ancorado nos limites documentados.

    Os centros de `TT201_PV_C` são o piso do envelope, a região normal, o
    set-point, a região alta e o **TAH-201**: "alto" passa a significar
    literalmente "na região do alarme". Os centros de `ZT201_pct` incluem o
    **ZAH-201 (80 %)**, de modo que "curso alto" é o mesmo conceito que a
    filosofia de alarmes da unidade usa.
    """
    keep = set(tags) if tags else None
    universes: List[TagUniverse] = []
    for tag, kind, unit, role, equip, (lo, hi), centers, span in TAG_SPEC:
        if keep and tag not in keep:
            continue
        u = TagUniverse(
            tag=tag, kind=kind, unit=unit, lo=lo, hi=hi, role=role, equipment=equip,
            alarm_hi=TAH_C if tag == "TT201_PV_C" else (TRAVEL_ZAH * 100 if "pct" in tag else None),
        )
        u.add(partition_from_breakpoints(centers, DEFAULT_LEVEL_TERMS, "level"))
        u.add(ruspini_partition(-span, span, DEFAULT_TREND_TERMS, "trend"))
        if tag == "TT201_PV_C":
            u.add(
                partition_from_breakpoints(
                    [-3.0, -1.0, 0.0, 1.0, 3.0],
                    ("muito_negativo", "negativo", "nulo", "positivo", "muito_positivo"),
                    "error",
                )
            )
        universes.append(u)
    return VariableBook(universes)


def topology() -> Dict[str, List[str]]:
    """Malhas do P&ID U-200-PID-201, para o viés relacional da atenção."""
    return {
        "TIC-201": ["TT201_PV_C", "TIC201_SP_C", "TIC201_OUT_pct", "ZT201_pct"],
        "loop_agua_temperada": ["FT201_m3h", "TT202_C", "TT207_C", "TT203_C", "ZT201_pct"],
        "reator": ["TT201_PV_C", "TT204_C", "AT205_X", "FT204_m3h"],
    }


def adjacency(book: VariableBook) -> np.ndarray:
    n = len(book)
    a = np.eye(n, dtype=np.float32)
    for members in topology().values():
        present = [m for m in members if m in book]
        for i, x in enumerate(present):
            for y in present[i + 1 :]:
                ix, iy = book.index(x), book.index(y)
                a[ix, iy] = a[iy, ix] = 1.0
    return a


# ---------------------------------------------------------------------------
# Leitura do acervo
# ---------------------------------------------------------------------------

@dataclass
class Historian:
    """Historian do U-200: séries numéricas + colunas categóricas."""

    timestamps: List[datetime]
    series: Dict[str, np.ndarray]
    mode: List[str] = field(default_factory=list)
    state: List[str] = field(default_factory=list)
    po: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.timestamps)

    @property
    def dt_seconds(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return (self.timestamps[1] - self.timestamps[0]).total_seconds()


CATEGORICAL = ("TIC201_MODE", "UNIT_STATE", "PO")


def load_historian(root: str | Path, name: str = "historian.csv") -> Historian:
    rows = _read_csv(Path(root) / "data" / name)
    if not rows:
        raise FileNotFoundError(f"historian vazio ou ausente: {root}/data/{name}")
    numeric = [c for c in rows[0] if c not in ("timestamp",) + CATEGORICAL]
    return Historian(
        timestamps=[_parse_ts(r["timestamp"]) for r in rows],
        series={c: np.array([_f(r[c]) for r in rows]) for c in numeric},
        mode=[r.get("TIC201_MODE", "") for r in rows],
        state=[r.get("UNIT_STATE", "") for r in rows],
        po=[r.get("PO", "") for r in rows],
    )


def load_tables(root: str | Path) -> Dict[str, List[dict]]:
    """Alarmes, log de eventos e LIMS."""
    r = Path(root)
    return {
        "alarms": _read_csv(r / "data/alarms.csv"),
        "events": _read_csv(r / "data/event_log.csv"),
        "quality": _read_csv(r / "data/quality.csv"),
    }


#: Silos do caso e o tipo de documento de cada um.
SILOS = {
    "maintenance": "wo",
    "production": "po",
    "operations": "shift_report",
    "process": "process_doc",
    "engineering": "engineering",
    "vendor": "vendor",
    "safety": "hazop",
    "specialist": "specialist",
}


def load_documents(
    root: str | Path,
    silos: Optional[Sequence[str]] = None,
    max_chars: int = 20000,
) -> List[Document]:
    """Extrai o texto dos PDFs dos silos.

    Requer ``pypdf``. Documentos ilegíveis (digitalizados sem OCR) são
    reportados como texto vazio em vez de quebrar a ingestão — o acervo real
    sempre tem alguns.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependência opcional
        raise ImportError("extração de PDF requer `pip install pypdf`") from exc

    r = Path(root)
    docs: List[Document] = []
    for silo, kind in SILOS.items():
        if silos and silo not in silos:
            continue
        for f in sorted((r / silo).glob("*.pdf")):
            try:
                text = "\n".join((p.extract_text() or "") for p in PdfReader(f).pages)
            except Exception:  # noqa: BLE001 - PDF corrompido não pode parar a ingestão
                text = ""
            docs.append(
                Document(
                    doc_id=f.stem,
                    kind=kind,
                    text=text[:max_chars],
                    timestamp=_date_in_name(f.stem),
                    meta={"silo": silo, "arquivo": f.name, "chars": str(len(text))},
                )
            )
    return docs


# ---------------------------------------------------------------------------
# Episódios para treino
# ---------------------------------------------------------------------------

class HistorianEpisode:
    """Episódio construído a partir do historian real.

    Compatível com :class:`~fuzzytf.data.dataset.EpisodeDataset`. Os alvos são:

    ``u_command``
        a saída real do TIC-201 — imitação do que a planta de fato fez;
    ``band``
        envelope de contingência derivado do ZAH-201 e do estado da unidade;
    ``advisories``
        supervisão distante a partir do log de eventos e das WOs (janela Δ
        anterior a cada evento de manutenção em FV-201);
    ``fault_active``
        estado ANORMAL declarado pela própria unidade.

    Nada aqui é *ground truth* de causa raiz — esse arquivo é interno ao caso e
    não é usado no treino, apenas na avaliação.
    """

    def __init__(
        self,
        hist: Historian,
        events: Sequence[dict] = (),
        alarms: Sequence[dict] = (),
        advisory_window_h: float = 8.0,
        advisories: Sequence[str] = ADVISORIES_U200,
    ) -> None:
        self.hist = hist
        self.advisory_names = list(advisories)
        n = len(hist)
        self.series = dict(hist.series)
        self.setpoint = hist.series["TIC201_SP_C"].copy()
        self.u_command = hist.series[ACTION_TAG] / 100.0
        self.valve_true = hist.series["ZT201_pct"] / 100.0
        self.T_true = hist.series["TT201_PV_C"].copy()
        self.fault_active = np.array([s != "NORMAL" for s in hist.state], dtype=bool)
        self.fault = _FaultLabel("anormal" if self.fault_active.any() else "none")
        self.band = self._bands()
        self.advisories = self._distant_supervision(events, alarms, advisory_window_h)

    def __len__(self) -> int:
        return len(self.u_command)

    def window(self, k: int, n: int) -> Dict[str, List[float]]:
        lo = max(0, k - n + 1)
        pad = n - (k - lo + 1)
        out = {}
        for tag, v in self.series.items():
            seq = v[lo : k + 1]
            if pad:
                seq = np.concatenate([np.full(pad, seq[0]), seq])
            out[tag] = seq.astype(float).tolist()
        return out

    # -- alvos ----------------------------------------------------------
    def _bands(self) -> np.ndarray:
        """Faixa admissível de comando, em fração de curso.

        Regra declarada (não aprendida): acima do ZAH-201 a unidade está
        consumindo a reserva de curso, e a faixa admissível se estreita para
        forçar a decisão a subir de nível — reduzir carga ou intervir no
        equipamento — em vez de continuar abrindo a válvula.
        """
        z = self.valve_true
        band = np.tile(np.array([0.0, 1.0]), (len(z), 1))
        alto = z > TRAVEL_ZAH
        band[alto, 1] = TRAVEL_ZAH + 0.10
        return band

    def _distant_supervision(
        self, events: Sequence[dict], alarms: Sequence[dict], window_h: float
    ) -> np.ndarray:
        """Rotula as janelas a partir de eventos, alarmes e regras declaradas.

        Achado do caso U-200: **as WOs que explicam o incidente são anteriores
        à janela do historian** (a revisão da FV-201 é de 29/07; o registro
        começa em 08/08). Supervisão distante por evento *dentro* da janela,
        que é a receita usual, não produz nenhum rótulo de manutenção aqui.
        O que sobra dentro da janela é o log de alarmes — e é dele que sai o
        sinal de curso alto. A história de manutenção anterior entra por outro
        caminho, como contexto de equipamento (:func:`equipment_context`), não
        como rótulo.
        """
        idx = {a: i for i, a in enumerate(self.advisory_names)}
        adv = np.zeros((len(self), len(self.advisory_names)), dtype=np.float32)
        ts = np.array([t.timestamp() for t in self.hist.timestamps])
        delta = window_h * 3600.0

        for ev in events:
            when = _parse_ts(ev.get("timestamp", ""))
            if when is None:
                continue
            sel = (ts <= when.timestamp()) & (ts >= when.timestamp() - delta)
            if not sel.any():
                continue
            text = f"{ev.get('event_type','')} {ev.get('description','')}".lower()
            for label in _labels_for(text):
                if label in idx:
                    adv[sel, idx[label]] = 1.0

        for al in alarms:
            when = _parse_ts(al.get("timestamp", ""))
            if when is None or str(al.get("estado", "")).upper() != "ATIVO":
                continue
            sel = (ts <= when.timestamp()) & (ts >= when.timestamp() - delta)
            if not sel.any():
                continue
            text = f"{al.get('tag','')} {al.get('descricao','')}".lower()
            for label in _labels_for(text):
                if label in idx:
                    adv[sel, idx[label]] = 1.0

        # a própria unidade declarando condição anormal já justifica registro
        adv[self.fault_active, idx["registrar_no_relatorio_de_turno"]] = 1.0
        adv[self.valve_true > TRAVEL_ZAH, idx["operar_em_contingencia"]] = 1.0
        adv[self.valve_true > TRAVEL_ZAH, idx["verificar_capacidade_resfriamento"]] = 1.0
        return adv


@dataclass
class _FaultLabel:
    kind: str


_LABEL_HINTS = {
    "fv-201": ["acionar_manutencao_fv201", "abrir_ordem_de_trabalho"],
    "válvula": ["acionar_manutencao_fv201"],
    "valvula": ["acionar_manutencao_fv201"],
    "at-205": ["inspecionar_analisador_at205"],
    "analisador": ["inspecionar_analisador_at205"],
    "carga": ["revisar_temperatura_de_carga"],
    "alimentação": ["revisar_temperatura_de_carga"],
    "trocador": ["verificar_capacidade_resfriamento"],
    "e-201": ["verificar_capacidade_resfriamento"],
    "investigação": ["acionar_engenharia_de_processo"],
    "zah-201": ["acionar_manutencao_fv201", "verificar_capacidade_resfriamento",
                "operar_em_contingencia"],
    "curso alto": ["acionar_manutencao_fv201", "operar_em_contingencia"],
    "tah-203": ["verificar_capacidade_resfriamento", "acionar_engenharia_de_processo"],
    "água de torre": ["verificar_capacidade_resfriamento"],
    "torre": ["verificar_capacidade_resfriamento"],
    "condição anormal": ["acionar_engenharia_de_processo",
                         "registrar_no_relatorio_de_turno"],
}


def _labels_for(text: str) -> List[str]:
    out: List[str] = []
    for key, labels in _LABEL_HINTS.items():
        if key in text:
            out.extend(labels)
    return sorted(set(out))


def episodes_from_case(
    root: str | Path,
    include_baseline: bool = True,
    advisory_window_h: float = 8.0,
) -> List[HistorianEpisode]:
    """Constrói os episódios de treino a partir do acervo.

    Por padrão inclui o historian do incidente **e** o baseline: sem o baseline,
    o modelo só vê a unidade degradada e não tem contra-exemplo do que é operar
    com reserva de curso saudável.
    """
    tables = load_tables(root)
    eps = [
        HistorianEpisode(
            load_historian(root), tables["events"], tables["alarms"], advisory_window_h
        )
    ]
    if include_baseline and (Path(root) / "data/historian_baseline.csv").exists():
        eps.append(
            HistorianEpisode(
                load_historian(root, "historian_baseline.csv"), [], [], advisory_window_h
            )
        )
    return eps


def equipment_context(
    root: str | Path, reference: datetime, equipment: str = "FV-201"
) -> Dict[str, float]:
    """Condição do equipamento a partir da história de manutenção anterior.

    Este é o canal que o U-200 mostra ser indispensável e que nenhum dado de
    processo fornece: quantas intervenções o equipamento sofreu, há quanto
    tempo foi a última, e se houve troca de peça. A perda de capacidade da
    FV-201 **não é medida por nenhum instrumento da unidade** — só a história
    documental a explica.

    Devolve um vetor pequeno e interpretável, pensado para entrar como token de
    contexto (ver `docs/05` e QP-11).
    """
    tables = load_tables(root)
    ref = reference.timestamp()
    key = equipment.lower().replace("-", "")
    hits = []
    for ev in tables["events"]:
        when = _parse_ts(ev.get("timestamp", ""))
        text = str(ev.get("description", "")).lower().replace("-", "")
        if when is None or when.timestamp() > ref:
            continue
        if key in text or "manuten" in str(ev.get("event_type", "")).lower() and key in text:
            hits.append(when)
    dias = [(ref - h.timestamp()) / 86400.0 for h in hits]
    return {
        "n_intervencoes": float(len(hits)),
        "dias_desde_ultima": float(min(dias)) if dias else float("nan"),
        "dias_desde_primeira": float(max(dias)) if dias else float("nan"),
    }


# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _f(v, default=float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _parse_ts(v: str) -> Optional[datetime]:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(v.strip(), fmt)
        except (ValueError, AttributeError):
            continue
    return None


def _date_in_name(name: str) -> Optional[str]:
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", name)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None
