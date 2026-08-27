#!/usr/bin/env python3
"""Experimento Crawl-01 — imitação do professor + envelope + orientações.

Uso:

    python scripts/run_crawl.py --config experiments/crawl_01_pid_imitation/config.json
    python scripts/run_crawl.py --smoke        # versão minúscula, ~1 min

Ao final:
  * treina o FT-IC no simulador,
  * avalia em malha aberta (imitação, envelope, orientações, diagnóstico),
  * fecha a malha e compara com o PI de referência,
  * escreve `report.json` e as regras mais ativas em `rules.txt`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from fuzzytf.config import ExperimentConfig
from fuzzytf.data import (
    ADVISORIES,
    FAULTS,
    DatasetConfig,
    EpisodeDataset,
    FaultSpec,
    LoopSimulator,
    default_variable_book,
    split_episodes,
    topology_adjacency,
)
from fuzzytf.eval import ModelPolicy, PIPolicy, compare_policies
from fuzzytf.interpret import explain_sample, top_rules
from fuzzytf.model import FTIC
from fuzzytf.train import train


def build(args) -> ExperimentConfig:
    default_cfg = Path(__file__).resolve().parents[1] / "experiments/crawl_01_pid_imitation/config.json"
    path = args.config or (default_cfg if default_cfg.exists() else None)
    cfg = ExperimentConfig.load(path) if path else ExperimentConfig(name="crawl-01")
    if args.smoke:
        cfg.name = "crawl-01-smoke"
        cfg.tokenizer.window = 16
        cfg.model.d_model = 64
        cfg.model.n_blocks = 2
        cfg.model.attention.n_heads = 4
        cfg.model.anfis.n_heads = 2
        cfg.model.anfis.n_rules = 16
        cfg.model.anfis.n_axes = 4
        cfg.train.epochs = 2
        cfg.train.batch_size = 32
    cfg.model.n_advisories = len(ADVISORIES)
    cfg.model.n_fault_classes = len(FAULTS)
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=2, help="passo entre janelas consecutivas")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mixer", type=str, default=None, choices=["anfis", "mlp"])
    args = ap.parse_args()

    cfg = build(args)
    if args.mixer:
        cfg.model.mixer = args.mixer
        cfg.name = f"{cfg.name}-{args.mixer}"
    n_ep = 8 if args.smoke else args.episodes
    n_steps = 200 if args.smoke else args.steps
    out_dir = Path(args.out or f"runs/{cfg.name}")

    print(f"[1/5] simulando {n_ep} episódios de {n_steps} amostras")
    sim = LoopSimulator(seed=args.seed)
    episodes = sim.dataset(n_ep, n_steps)
    train_eps, val_eps = split_episodes(episodes, val_fraction=0.25, seed=args.seed)

    book = default_variable_book()
    ds_cfg = DatasetConfig(
        window=cfg.tokenizer.window, top_p=cfg.tokenizer.top_p, stride=args.stride
    )
    train_set = EpisodeDataset(train_eps, book, ds_cfg)
    val_set = EpisodeDataset(val_eps, book, ds_cfg)
    print("      ", train_set.stats())

    print("[2/5] construindo o modelo")
    model = FTIC(book, cfg.model, cfg.tokenizer, action_tags=["V-097"])
    model.set_topology_prior(torch.as_tensor(topology_adjacency(book)), strength=0.5)
    print(f"       parâmetros: {model.n_parameters():,} | mixer={cfg.model.mixer}")

    print("[3/5] treinando")
    result = train(model, train_set, val_set, cfg, out_dir=out_dir)

    print("[4/5] avaliação em malha fechada")
    faults = [
        FaultSpec(kind="none"),
        FaultSpec(kind="valve_travel_limit", start=200, severity=0.7, travel_limit=0.66),
        FaultSpec(kind="valve_stiction", start=200, severity=0.8, stiction_band=0.09),
        FaultSpec(kind="fouling", start=200, severity=0.8, fouling_rate=4e-4),
    ]
    policies = {
        "PI (referência)": PIPolicy(),
        "FT-IC": ModelPolicy(model, book, ds_cfg),
        "FT-IC sem envelope": ModelPolicy(model, book, ds_cfg, use_band=False),
    }
    closed = compare_policies(policies, faults, n_steps=n_steps if not args.smoke else 200)
    for name, m in closed.items():
        print(f"       {name:22s} IAE={m['iae']:.3f} esforço={m['control_effort']:.2f} "
              f"alarme={m.get('alarm_fraction', float('nan')):.3f}")

    print("[5/5] interpretabilidade")
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = val_set[len(val_set) // 2]
    (out_dir / "explicacao.txt").write_text(explain_sample(model, book, sample, ds_cfg))
    (out_dir / "rules.txt").write_text(top_rules(model, val_set, ds_cfg, n=12))
    report = {"open_loop": result["final"], "closed_loop": closed, "config": cfg.to_dict()}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nresultados em {out_dir}/")


if __name__ == "__main__":
    main()
