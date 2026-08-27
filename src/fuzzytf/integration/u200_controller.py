"""Plugue do FT-IC no harness do estudo de caso U-200.

O caso define o contrato ``BaseController`` (``reset`` / ``update``) e reserva
``ProposedController`` para a estratégia a ser desenvolvida. Esta classe é essa
estratégia: recebe o que o DCS mede, tokeniza a janela, roda o modelo e devolve
a abertura comandada de FV-201 em fração de curso.

Duas decisões que valem registro:

* **Nada é lido além do contrato.** ``observation`` não traz o ``gain_trim`` da
  válvula, e é justamente essa a graça do benchmark: a perda de capacidade tem
  de ser inferida do comportamento e do acervo documental, não medida.
* **Tags ausentes entram como ausentes.** ``TT202_C`` e ``FT204_m3h`` não estão
  na observação do DCS; em vez de inventar valor, entram como NaN e o
  fuzzificador marca ``valid=False``. O modelo enxerga o buraco.
"""

from __future__ import annotations

import importlib.util
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..data import u200
from ..data.dataset import DatasetConfig
from ..fuzzy import VariableBook
from ..tokenizer import ProcessTokenizer

#: observação do DCS → tag do historian, com a conversão de unidade aplicada.
OBS_MAP = {
    "SP": ("TIC201_SP_C", u200.x2_to_c),
    "PV": ("TT201_PV_C", u200.x2_to_c),
    "ZT201": ("ZT201_pct", lambda v: np.asarray(v, dtype=float) * 100.0),
    "FT201": ("FT201_m3h", None),
    "TT203": ("TT203_C", None),
    "TT207": ("TT207_C", None),
    "TT204": ("TT204_C", None),
    "AT205": ("AT205_X", None),
}


class FTICController:
    """Controlador FT-IC compatível com ``BaseController`` do caso U-200."""

    nome = "FT-IC"

    def __init__(
        self,
        model,
        book: Optional[VariableBook] = None,
        cfg: Optional[DatasetConfig] = None,
        device: str = "cpu",
        use_band: bool = True,
        fallback=None,
    ) -> None:
        self.model = model.eval()
        self.book = book or u200.variable_book()
        self.cfg = cfg or DatasetConfig(window=32, delta_scale=0.05)
        self.tokenizer = ProcessTokenizer(self.book)
        self.tokenizer.fuzzifier.top_p = self.cfg.top_p
        self.device = device
        self.use_band = use_band
        self.fallback = fallback          # controlador de retaguarda (ex.: PID)
        self.history: deque = deque(maxlen=self.cfg.window)
        self.q_prev = 0.0
        self.log: List[Dict[str, object]] = []

    # ------------------------------------------------------------------
    def reset(self, operating_context: Optional[dict] = None) -> None:
        ctx = operating_context or {}
        self.q_prev = float(ctx.get("q_bias", 0.0))
        self.history.clear()
        self.log.clear()
        if self.fallback is not None and hasattr(self.fallback, "reset"):
            self.fallback.reset(operating_context)

    def update(self, observation: dict, context: dict, dt: float) -> float:
        import torch

        self.history.append(self._to_tags(observation))
        if len(self.history) < 2 and self.fallback is not None:
            # janela ainda vazia: quem responde é a retaguarda
            self.q_prev = float(self.fallback.update(observation, context, dt))
            return self.q_prev

        window = self._window()
        sp_c = float(u200.x2_to_c(observation["SP"]))
        batch = self.tokenizer.encode(
            [window], setpoints=[{"TT201_PV_C": sp_c}], device=self.device
        )
        with torch.no_grad():
            out = self.model(batch)
        delta = float(out.control.delta[0, 0] if self.use_band else out.control.delta_raw[0, 0])
        q = float(np.clip(self.q_prev + self.cfg.delta_scale * delta, 0.0, 1.0))

        self.log.append(
            {
                "q": q,
                "delta": delta,
                "band": (float(out.control.band_lo[0, 0]), float(out.control.band_hi[0, 0])),
                "advisories": (
                    torch.sigmoid(out.advisory.logits[0]).cpu().numpy()
                    if out.advisory is not None
                    else None
                ),
            }
        )
        self.q_prev = q
        return q

    # ------------------------------------------------------------------
    def advisories_raised(self, names=u200.ADVISORIES_U200, threshold: float = 0.5) -> Dict[str, int]:
        """Quantas amostras levantaram cada orientação ao longo da corrida."""
        counts = {n: 0 for n in names}
        for row in self.log:
            p = row.get("advisories")
            if p is None:
                continue
            for i, n in enumerate(names[: len(p)]):
                counts[n] += int(p[i] > threshold)
        return counts

    def _to_tags(self, observation: dict) -> Dict[str, float]:
        row = {tag: float("nan") for tag in self.book.tags}
        for key, (tag, conv) in OBS_MAP.items():
            if key in observation and tag in row:
                v = observation[key]
                row[tag] = float(conv(v)) if conv else float(v)
        if u200.ACTION_TAG in row:
            row[u200.ACTION_TAG] = self.q_prev * 100.0
        return row

    def _window(self) -> Dict[str, List[float]]:
        n = self.cfg.window
        rows = list(self.history)
        if len(rows) < n:
            rows = [rows[0]] * (n - len(rows)) + rows
        return {tag: [r[tag] for r in rows] for tag in self.book.tags}


def load_case_module(root: str | Path, name: str = "u200_case"):
    """Importa ``tools/u200_case.py`` do estudo de caso.

    Atenção: o módulo do caso executa os experimentos A, B e C **no import**, e
    depende de `matplotlib` e afins. Importar é caro; faça-o só quando for
    realmente rodar o harness.
    """
    path = Path(root) / "tools" / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"módulo do caso não encontrado: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
