"""Avaliação em malha fechada: o modelo no lugar do controlador.

Métricas de malha aberta (o modelo imita bem o professor?) são necessárias mas
não suficientes: um erro pequeno e *sistemático* na ação pode desestabilizar a
malha. Por isso a avaliação do estágio Crawl termina sempre com um rollout.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..data.dataset import DatasetConfig
from ..data.simulator import Episode, FaultSpec, LoopSimulator, PIController
from ..fuzzy import VariableBook
from ..tokenizer import ProcessTokenizer
from .metrics import closed_loop_metrics


class ModelPolicy:
    """Envolve um :class:`FTIC` treinado como lei de controle da simulação.

    ``use_band`` liga o filtro de contingência previsto pelo próprio modelo:
    a ação é recortada na faixa que ele julga admissível. Desligá-lo é a
    ablação que mede quanto o envelope contribui de fato.
    """

    def __init__(
        self,
        model,
        book: VariableBook,
        cfg: Optional[DatasetConfig] = None,
        device: str = "cpu",
        use_band: bool = True,
    ) -> None:
        self.model = model.eval()
        self.cfg = cfg or DatasetConfig()
        self.tokenizer = ProcessTokenizer(book)
        self.tokenizer.fuzzifier.top_p = self.cfg.top_p
        self.device = device
        self.use_band = use_band
        self.trace: List[Dict[str, np.ndarray]] = []

    @torch.no_grad()
    def __call__(self, k: int, series: Dict[str, np.ndarray], sp: float, u_prev: float, band) -> float:
        n = self.cfg.window
        window = {}
        for tag, v in series.items():
            seq = v[max(0, len(v) - n) :]
            if len(seq) < n:
                seq = np.concatenate([np.full(n - len(seq), seq[0]), seq])
            window[tag] = seq.astype(float).tolist()

        batch = self.tokenizer.encode(
            [window], setpoints=[{"T-102": sp}], device=self.device
        )
        out = self.model(batch)
        delta = float(out.control.delta[0, 0]) if self.use_band else float(out.control.delta_raw[0, 0])
        self.trace.append(
            {
                "delta": delta,
                "band_lo": float(out.control.band_lo[0, 0]),
                "band_hi": float(out.control.band_hi[0, 0]),
                "fault": (
                    torch.softmax(out.fault_logits[0], -1).cpu().numpy()
                    if out.fault_logits is not None
                    else np.zeros(1)
                ),
                "advisory": (
                    torch.sigmoid(out.advisory.logits[0]).cpu().numpy()
                    if out.advisory is not None
                    else np.zeros(1)
                ),
            }
        )
        return u_prev + self.cfg.delta_scale * delta


class PIPolicy:
    """Linha de base: o mesmo PI do professor, sem supervisório."""

    def __init__(self, kp: float = 0.08, ki: float = 0.01) -> None:
        self.pi = PIController(kp=kp, ki=ki)

    def __call__(self, k: int, series, sp: float, u_prev: float, band) -> float:
        return self.pi.step(sp - series["T-102"][-1])


def compare_policies(
    policies: Dict[str, object],
    faults: Sequence[FaultSpec],
    n_steps: int = 600,
    seed: int = 100,
    alarm_hi: float = 108.0,
) -> Dict[str, Dict[str, float]]:
    """Roda cada política nos mesmos episódios (mesma semente ⇒ mesmo ruído)."""
    results: Dict[str, Dict[str, float]] = {}
    for name, policy in policies.items():
        rows = []
        for i, fault in enumerate(faults):
            if hasattr(policy, "pi"):
                policy.pi.reset()
            if hasattr(policy, "trace"):
                policy.trace = []
            sim = LoopSimulator(seed=seed + i)
            ep = sim.rollout(policy, n_steps=n_steps, fault=fault, setpoint_changes=False)
            rows.append(
                closed_loop_metrics(
                    ep.series["T-102"], ep.setpoint, ep.series["V-097"], alarm_hi=alarm_hi
                )
            )
        results[name] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return results
