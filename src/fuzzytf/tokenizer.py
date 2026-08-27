"""Ponte entre a fuzzificação (NumPy) e o modelo (PyTorch).

`ProcessTokenizer` é o objeto que o resto do código usa: recebe janelas
``{tag: valores}`` e devolve o batch de tensores esperado por :class:`FTIC`.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from .config import TokenizerConfig
from .fuzzy import Fuzzifier, VariableBook


class ProcessTokenizer:
    def __init__(self, book: VariableBook, cfg: Optional[TokenizerConfig] = None) -> None:
        self.cfg = cfg or TokenizerConfig()
        self.book = book
        self.fuzzifier = Fuzzifier(book, top_p=self.cfg.top_p, layout=self.cfg.layout)

    @property
    def n_tags(self) -> int:
        return len(self.book)

    @property
    def n_state_slots(self) -> int:
        return self.book.n_state_slots

    def encode(
        self,
        windows: Sequence[Mapping[str, Sequence[float]]],
        setpoints: Optional[Sequence[Mapping[str, float]]] = None,
        device: str = "cpu",
    ) -> Dict[str, "torch.Tensor"]:
        """Fuzzifica e converte para tensores ``(B, S, ...)``."""
        import torch

        arrays = self.fuzzifier.batch(windows, setpoints)
        dtypes = {
            "tag_index": torch.long,
            "lag": torch.long,
            "slot_ids": torch.long,
            "weights": torch.float32,
            "mask": torch.bool,
            "value": torch.float32,
            "valid": torch.bool,
        }
        return {k: torch.as_tensor(v, dtype=dtypes[k], device=device) for k, v in arrays.items()}

    def describe(self, windows: Sequence[Mapping[str, Sequence[float]]], row: int = 0) -> List[str]:
        """Leitura linguística dos tokens de uma janela (depuração/relatório)."""
        fw = self.fuzzifier.transform(windows[row])
        return [self.fuzzifier.describe_token(fw, i) for i in range(len(fw))]

    @staticmethod
    def collate(items: Sequence[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Empilha amostras já fuzzificadas (uso com DataLoader)."""
        return {k: np.stack([it[k] for it in items]) for k in items[0]}
