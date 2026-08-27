"""Construção de amostras de treino a partir de episódios simulados ou reais.

Uma amostra é a janela ``[k-n+1, k]`` de todas as tags mais os alvos naquele
instante:

======================  ==================================================
``delta``               ação incremental do professor, normalizada por
                        ``delta_scale`` (forma velocidade: Δu, não u)
``band``                faixa admissível de Δu imposta pela contingência
``advisories``          rótulos multi-label das orientações
``fault``               índice da falha ativa (tarefa auxiliar)
``next_slots/next_w``   estado fuzzy em k+1 (pré-treino de previsão)
======================  ==================================================

A forma incremental é deliberada: é como as malhas industriais realmente
recebem comando (velocity form), evita *bumps* na troca automático/manual e
mantém a saída do modelo em uma escala fixa independente do ponto de operação.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..fuzzy import Fuzzifier, VariableBook
from .simulator import ADVISORIES, Episode, FAULTS

FAULT_INDEX = {f: i for i, f in enumerate(FAULTS)}


@dataclass
class DatasetConfig:
    window: int = 32
    horizon: int = 1          # passos à frente na tarefa de previsão
    delta_scale: float = 0.1  # unidade de ação: 10% de curso por amostra
    stride: int = 1
    top_p: int = 3
    layout: str = "tag_major"


class EpisodeDataset:
    """Coleção indexável de amostras ``(janela, alvos)``.

    Compatível com ``torch.utils.data.DataLoader`` via :func:`collate`.
    """

    def __init__(
        self,
        episodes: Sequence[Episode],
        book: VariableBook,
        cfg: Optional[DatasetConfig] = None,
        action_tag: str = "V-097",
    ) -> None:
        self.cfg = cfg or DatasetConfig()
        self.book = book
        self.episodes = list(episodes)
        self.action_tag = action_tag
        self.fuzzifier = Fuzzifier(book, top_p=self.cfg.top_p, layout=self.cfg.layout)
        self.index: List[tuple] = [
            (e, k)
            for e, ep in enumerate(self.episodes)
            for k in range(1, len(ep) - self.cfg.horizon, self.cfg.stride)
        ]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, np.ndarray]:
        e, k = self.index[i]
        ep = self.episodes[e]
        cfg = self.cfg

        sp = {"T-102": float(ep.setpoint[k])}
        fw = self.fuzzifier.transform(ep.window(k, cfg.window), setpoints=sp)
        nxt = self.fuzzifier.transform(
            ep.window(k + cfg.horizon, cfg.window),
            setpoints={"T-102": float(ep.setpoint[k + cfg.horizon])},
        )

        u_prev = ep.u_command[k - 1]
        delta = (ep.u_command[k] - u_prev) / cfg.delta_scale
        lo = (ep.band[k, 0] - u_prev) / cfg.delta_scale
        hi = (ep.band[k, 1] - u_prev) / cfg.delta_scale

        # alvo de previsão: só os tokens do instante corrente (lag == 0)
        cur = fw.lag == 0
        nxt_cur = nxt.lag == 0

        return {
            "tag_index": fw.tag_index,
            "lag": fw.lag,
            "slot_ids": fw.slot_ids,
            "weights": fw.weights,
            "mask": fw.mask,
            "value": fw.value,
            "valid": fw.valid,
            "target_delta": np.float32(np.clip(delta, -1.0, 1.0)),
            "target_band": np.array(
                [np.clip(lo, -1.5, 1.5), np.clip(hi, -1.5, 1.5)], dtype=np.float32
            ),
            "target_advisory": ep.advisories[k],
            # Episódios reais podem trazer rótulos de falha fora do vocabulário
            # do simulador; caem em "1" (anormal genérico) em vez de quebrar.
            "target_fault": np.int64(
                FAULT_INDEX.get(ep.fault.kind, 1) if ep.fault_active[k] else 0
            ),
            "forecast_pos": np.where(cur)[0].astype(np.int64),
            "forecast_slots": nxt.slot_ids[nxt_cur],
            "forecast_weights": nxt.weights[nxt_cur],
            "u_prev": np.float32(u_prev),
        }

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, float]:
        """Estatísticas rápidas para dimensionar o experimento."""
        n_fault = sum(int(ep.fault.kind != "none") for ep in self.episodes)
        return {
            "episodes": len(self.episodes),
            "samples": len(self),
            "fraction_with_fault": n_fault / max(len(self.episodes), 1),
            "tokens_per_sample": len(self.book) * self.cfg.window,
        }


def collate(items: Sequence[Dict[str, np.ndarray]], device: str = "cpu") -> Dict[str, "torch.Tensor"]:
    """Empilha amostras em tensores torch."""
    import torch

    out: Dict[str, torch.Tensor] = {}
    long_keys = {"tag_index", "lag", "slot_ids", "target_fault", "forecast_pos", "forecast_slots"}
    bool_keys = {"mask", "valid"}
    for key in items[0]:
        stacked = np.stack([it[key] for it in items])
        if key in long_keys:
            out[key] = torch.as_tensor(stacked, dtype=torch.long, device=device)
        elif key in bool_keys:
            out[key] = torch.as_tensor(stacked, dtype=torch.bool, device=device)
        else:
            out[key] = torch.as_tensor(stacked, dtype=torch.float32, device=device)
    return out


def split_episodes(
    episodes: Sequence[Episode], val_fraction: float = 0.2, seed: int = 0
) -> tuple:
    """Divide episódios em treino/validação.

    A divisão é **por episódio**, nunca por amostra: janelas vizinhas se
    sobrepõem quase inteiramente, e dividir por amostra vazaria o alvo entre os
    conjuntos — um erro comum e silencioso em séries temporais.
    """
    idx = np.random.default_rng(seed).permutation(len(episodes))
    n_val = max(1, int(len(episodes) * val_fraction))
    val = [episodes[i] for i in idx[:n_val]]
    train = [episodes[i] for i in idx[n_val:]]
    return train, val
