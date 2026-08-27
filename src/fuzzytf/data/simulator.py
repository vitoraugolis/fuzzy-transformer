"""Simulador de uma malha industrial com falhas — o ambiente do estágio Crawl.

Não é um substituto do case study: é o banco de ensaio onde a arquitetura é
depurada com verdade-fundamental conhecida (sabemos exatamente quando a falha
começou, qual era a ação correta e qual orientação deveria ter sido emitida).

Planta simulada — aquecimento de um vaso com vapor:

    TIC-102 (temperatura)  ← controlada pela válvula de vapor V-097
    PT-204  (pressão do vaso)
    FT-301  (vazão de carga, distúrbio medido)
    V-097   (posição da válvula, variável manipulada)

Falhas injetáveis:

``valve_travel_limit``
    A válvula satura mecanicamente antes de 100% (o caso motivador: a saturação
    aparece muito antes de qualquer alarme de temperatura).
``valve_stiction``
    Atrito estático: a haste só se move quando o erro de comando excede a banda.
``fouling``
    Incrustação no trocador: eficiência térmica cai lentamente.
``sensor_drift``
    Deriva no transmissor de temperatura (a leitura mente, o processo não).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

TAGS = ("T-102", "SP-102", "P-204", "F-301", "V-097")

FAULTS = ("none", "valve_travel_limit", "valve_stiction", "fouling", "sensor_drift")

#: Vocabulário de orientações (saída não numérica do modelo).
ADVISORIES = (
    "acionar_manutencao_valvula",
    "inspecionar_instrumento",
    "programar_limpeza_trocador",
    "reduzir_carga",
    "abrir_ordem_de_trabalho",
    "registrar_no_relatorio_de_turno",
    "acionar_engenharia_de_processo",
    "operar_em_contingencia",
)
ADVISORY_INDEX = {a: i for i, a in enumerate(ADVISORIES)}


@dataclass
class PlantConfig:
    dt: float = 1.0            # s por amostra
    tau_T: float = 40.0        # constante de tempo térmica
    tau_P: float = 15.0
    tau_valve: float = 5.0
    gain_steam: float = 80.0   # K por unidade de abertura efetiva
    T_in: float = 60.0
    T_setpoint: float = 100.0
    P_base: float = 1.0
    P_gain: float = 0.02       # bar por K acima de T_in
    noise_T: float = 0.05
    noise_P: float = 0.004
    noise_F: float = 0.01
    F_nominal: float = 1.0
    alarm_T_hi: float = 108.0
    ranges: Dict[str, tuple] = field(
        default_factory=lambda: {
            "T-102": (60.0, 120.0),
            "SP-102": (60.0, 120.0),
            "P-204": (0.0, 3.0),
            "F-301": (0.0, 2.0),
            "V-097": (0.0, 1.0),
        }
    )


@dataclass
class FaultSpec:
    kind: str = "none"
    start: int = 10**9          # amostra em que a falha começa
    severity: float = 0.5       # 0..1
    # parâmetros derivados
    travel_limit: float = 0.6
    stiction_band: float = 0.06
    fouling_rate: float = 2e-4  # perda de eficiência por amostra
    drift_rate: float = 5e-3    # K por amostra


@dataclass
class Episode:
    """Um episódio simulado, com verdade-fundamental para supervisão."""

    series: Dict[str, np.ndarray]        # tags observadas (o que o histórico veria)
    u_command: np.ndarray               # comando enviado à válvula
    valve_true: np.ndarray              # posição real da haste (não observada!)
    T_true: np.ndarray                  # temperatura real (difere de T-102 sob drift)
    fault: FaultSpec
    fault_active: np.ndarray            # bool por amostra
    advisories: np.ndarray              # (N, n_advisories) rótulos multi-label
    band: np.ndarray                    # (N, 2) faixa admissível de comando
    setpoint: np.ndarray

    def __len__(self) -> int:
        return len(self.u_command)

    def window(self, k: int, n: int) -> Dict[str, List[float]]:
        """Janela ``[k-n+1, k]`` no formato de entrada do modelo."""
        lo = max(0, k - n + 1)
        pad = n - (k - lo + 1)
        out = {}
        for tag, v in self.series.items():
            seq = v[lo : k + 1]
            if pad:
                seq = np.concatenate([np.full(pad, seq[0]), seq])
            out[tag] = seq.astype(float).tolist()
        return out


class PIController:
    """Controlador PI com anti-windup — o "professor" do estágio Crawl."""

    def __init__(self, kp: float = 0.08, ki: float = 0.01, u_min: float = 0.0, u_max: float = 1.0):
        self.kp, self.ki = kp, ki
        self.u_min, self.u_max = u_min, u_max
        self.integral = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def step(self, error: float, u_min: Optional[float] = None, u_max: Optional[float] = None) -> float:
        lo = self.u_min if u_min is None else u_min
        hi = self.u_max if u_max is None else u_max
        raw = self.kp * error + self.ki * (self.integral + error)
        u = float(np.clip(raw, lo, hi))
        if lo < raw < hi:            # integra só fora da saturação (anti-windup)
            self.integral += error
        return u


class LoopSimulator:
    """Simula episódios da malha TIC-102/V-097 com falhas e supervisão."""

    def __init__(self, cfg: Optional[PlantConfig] = None, seed: int = 0) -> None:
        self.cfg = cfg or PlantConfig()
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def sample_fault(self, n_steps: int, p_fault: float = 0.75) -> FaultSpec:
        if self.rng.random() > p_fault:
            return FaultSpec(kind="none")
        kind = str(self.rng.choice(FAULTS[1:]))
        start = int(self.rng.integers(n_steps // 6, max(n_steps // 6 + 1, n_steps // 2)))
        sev = float(self.rng.uniform(0.3, 1.0))
        return FaultSpec(
            kind=kind,
            start=start,
            severity=sev,
            travel_limit=float(np.clip(0.95 - 0.30 * sev, 0.55, 0.95)),
            stiction_band=0.02 + 0.10 * sev,
            fouling_rate=1e-4 + 4e-4 * sev,
            drift_rate=2e-3 + 8e-3 * sev,
        )

    def run(
        self,
        n_steps: int = 600,
        fault: Optional[FaultSpec] = None,
        setpoint_changes: bool = True,
        policy=None,
    ) -> Episode:
        """Simula um episódio.

        ``policy`` é a lei de controle em malha fechada, com assinatura
        ``policy(k, series, setpoint, u_prev, band) -> u``. Quando omitida, usa
        o professor (PI + supervisório) que gera os alvos de treino. Passar o
        modelo aqui é o que fecha a malha na avaliação (ver ``eval/rollout.py``).
        """
        c = self.cfg
        fault = fault if fault is not None else self.sample_fault(n_steps)
        pid = PIController()

        T_true = c.T_setpoint + self.rng.normal(0, 0.5)
        valve = 0.45
        P = c.P_base + c.P_gain * (T_true - c.T_in)
        eff, drift = 1.0, 0.0
        u = u_prev = valve

        sp = np.full(n_steps, c.T_setpoint)
        if setpoint_changes:
            for _ in range(int(self.rng.integers(0, 3))):
                t0 = int(self.rng.integers(0, n_steps))
                sp[t0:] = c.T_setpoint + float(self.rng.uniform(-4, 4))

        # distúrbio de carga: passeio aleatório suave em torno do nominal
        load = c.F_nominal + np.cumsum(self.rng.normal(0, 0.01, n_steps))
        load = np.clip(load - (load.mean() - c.F_nominal), 0.5, 1.6)

        rec = {t: np.zeros(n_steps) for t in TAGS}
        u_cmd = np.zeros(n_steps)
        v_true = np.zeros(n_steps)
        T_rec = np.zeros(n_steps)
        active = np.zeros(n_steps, dtype=bool)
        adv = np.zeros((n_steps, len(ADVISORIES)), dtype=np.float32)
        band = np.tile(np.array([0.0, 1.0]), (n_steps, 1))

        for k in range(n_steps):
            on = k >= fault.start and fault.kind != "none"
            active[k] = on
            if on and fault.kind == "fouling":
                eff = max(0.45, eff - fault.fouling_rate)
            if on and fault.kind == "sensor_drift":
                drift += fault.drift_rate

            T_meas = T_true + drift + self.rng.normal(0, c.noise_T)
            lo_b, hi_b = self._contingency_band(fault, on, valve)
            band[k] = (lo_b, hi_b)
            adv[k] = self._advisories(fault, on, k)

            if policy is None:
                u = pid.step(sp[k] - T_meas, u_min=lo_b, u_max=hi_b)
            else:
                rec["T-102"][k] = T_meas
                rec["SP-102"][k] = sp[k]
                rec["P-204"][k] = P
                rec["F-301"][k] = load[k]
                rec["V-097"][k] = valve
                view = {t: v[: k + 1] for t, v in rec.items()}
                u = float(np.clip(policy(k, view, float(sp[k]), float(u_prev), (lo_b, hi_b)), 0.0, 1.0))
            u_prev = u

            # --- atuador --------------------------------------------
            target = u
            if on and fault.kind == "valve_stiction":
                if abs(target - valve) < fault.stiction_band:
                    target = valve      # haste presa
            valve += (target - valve) * c.dt / c.tau_valve
            if on and fault.kind == "valve_travel_limit":
                valve = min(valve, fault.travel_limit)
            valve = float(np.clip(valve, 0.0, 1.0))

            # --- processo -------------------------------------------
            dT = (
                -(T_true - c.T_in) * load[k] / c.F_nominal
                + c.gain_steam * valve * eff
            ) / c.tau_T
            T_true += c.dt * dT + self.rng.normal(0, c.noise_T)
            P += c.dt * ((c.P_base + c.P_gain * (T_true - c.T_in) - P) / c.tau_P)
            P += self.rng.normal(0, c.noise_P)

            rec["T-102"][k] = T_meas
            rec["SP-102"][k] = sp[k]
            rec["P-204"][k] = P
            rec["F-301"][k] = load[k] + self.rng.normal(0, c.noise_F)
            rec["V-097"][k] = valve
            u_cmd[k], v_true[k], T_rec[k] = u, valve, T_true

        return Episode(
            series=rec,
            u_command=u_cmd,
            valve_true=v_true,
            T_true=T_rec,
            fault=fault,
            fault_active=active,
            advisories=adv,
            band=band,
            setpoint=sp,
        )

    def rollout(self, policy, n_steps: int = 600, **kw) -> Episode:
        """Açúcar sintático: ``run`` com uma política externa em malha fechada."""
        return self.run(n_steps=n_steps, policy=policy, **kw)

    def dataset(self, n_episodes: int, n_steps: int = 600, **kw) -> List[Episode]:
        return [self.run(n_steps=n_steps, **kw) for _ in range(n_episodes)]

    # ------------------------------------------------------------------
    def _contingency_band(self, fault: FaultSpec, on: bool, valve: float) -> tuple:
        """Faixa admissível de comando que o *supervisório ideal* imporia.

        É a supervisão do envelope: sob limitação mecânica, não adianta mandar
        100% — o correto é trabalhar dentro do curso útil e chamar a manutenção.
        """
        if not on:
            return 0.0, 1.0
        if fault.kind == "valve_travel_limit":
            return 0.0, float(np.clip(fault.travel_limit - 0.05, 0.1, 1.0))
        if fault.kind == "valve_stiction":
            # evita micro-movimentos que só desgastam a haste
            return float(max(0.0, valve - 0.25)), float(min(1.0, valve + 0.25))
        if fault.kind == "fouling":
            return 0.0, 0.95
        return 0.0, 1.0

    def _advisories(self, fault: FaultSpec, on: bool, k: int) -> np.ndarray:
        a = np.zeros(len(ADVISORIES), dtype=np.float32)
        if not on:
            return a
        mapping = {
            "valve_travel_limit": [
                "acionar_manutencao_valvula",
                "abrir_ordem_de_trabalho",
                "operar_em_contingencia",
                "registrar_no_relatorio_de_turno",
            ],
            "valve_stiction": [
                "acionar_manutencao_valvula",
                "operar_em_contingencia",
                "registrar_no_relatorio_de_turno",
            ],
            "fouling": [
                "programar_limpeza_trocador",
                "acionar_engenharia_de_processo",
                "reduzir_carga",
                "registrar_no_relatorio_de_turno",
            ],
            "sensor_drift": [
                "inspecionar_instrumento",
                "abrir_ordem_de_trabalho",
                "registrar_no_relatorio_de_turno",
            ],
        }
        for name in mapping.get(fault.kind, []):
            a[ADVISORY_INDEX[name]] = 1.0
        return a


def default_variable_book(cfg: Optional[PlantConfig] = None):
    """VariableBook coerente com o simulador (usado no Crawl).

    As partições **não** são uniformes na faixa do instrumento: são ancoradas na
    faixa realmente operada e nos limites de engenharia (set-point, alarme). Uma
    partição uniforme em 60–120 °C gastaria quatro dos cinco termos em regiões
    onde a planta nunca opera, e "alto" deixaria de significar "perto do alarme".
    Ver `docs/03-tokenizacao-e-fuzzificacao.md`.

    Os spans de `trend` vêm do percentil 95 de |Δx| observado no simulador: com
    span grande demais, tudo vira "estável" e a dimensão não informa nada.
    """
    from ..fuzzy.variables import (
        DEFAULT_LEVEL_TERMS,
        DEFAULT_TREND_TERMS,
        TagUniverse,
        VariableBook,
        partition_from_breakpoints,
        ruspini_partition,
    )

    c = cfg or PlantConfig()
    #        tag        kind          unit    role          equip     centros de nível              trend
    spec = [
        ("T-102",  "temperature", "degC", "measurement", "TIC-102", [94, 98, 100.5, 103.5, 108], 0.25),
        ("SP-102", "temperature", "degC", "setpoint",    "TIC-102", [94, 98, 100.5, 103.5, 108], 0.50),
        ("P-204",  "pressure",    "bar",  "measurement", "V-201",   [1.70, 1.77, 1.80, 1.85, 1.92], 0.010),
        ("F-301",  "flow",        "t/h",  "measurement", "FT-301",  [0.75, 0.90, 1.00, 1.10, 1.25], 0.035),
        ("V-097",  "valve",       "-",    "manipulated", "V-097",   [0.15, 0.35, 0.50, 0.68, 0.90], 0.020),
    ]
    universes = []
    for tag, kind, unit, role, equip, centers, span in spec:
        lo, hi = c.ranges[tag]
        u = TagUniverse(
            tag=tag, kind=kind, unit=unit, lo=lo, hi=hi, role=role, equipment=equip,
            alarm_hi=c.alarm_T_hi if tag == "T-102" else None,
        )
        u.add(partition_from_breakpoints(centers, DEFAULT_LEVEL_TERMS, "level"))
        u.add(ruspini_partition(-span, span, DEFAULT_TREND_TERMS, "trend"))
        if tag == "T-102":
            # desvio em relação ao set-point: a grandeza que o controlador usa
            u.add(
                partition_from_breakpoints(
                    [-4.0, -1.5, 0.0, 1.5, 4.0],
                    ("muito_negativo", "negativo", "nulo", "positivo", "muito_positivo"),
                    "error",
                )
            )
        universes.append(u)
    return VariableBook(universes)


def topology_adjacency(book) -> np.ndarray:
    """Adjacência do P&ID: quem está na mesma malha/equipamento."""
    pairs = [
        ("T-102", "V-097"), ("T-102", "P-204"), ("P-204", "V-097"),
        ("F-301", "T-102"), ("SP-102", "T-102"), ("SP-102", "V-097"),
    ]
    n = len(book)
    a = np.eye(n, dtype=np.float32)
    for x, y in pairs:
        if x in book and y in book:
            i, j = book.index(x), book.index(y)
            a[i, j] = a[j, i] = 1.0
    return a
