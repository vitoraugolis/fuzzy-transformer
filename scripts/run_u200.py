#!/usr/bin/env python3
"""Estágio Walk — FT-IC sobre o estudo de caso U-200.

    python scripts/setup_case_study.py             # extrai case_study.zip
    python scripts/run_u200.py --path case_study   # ingere, treina e avalia

Faz o caminho completo com o acervo real: vocabulário fuzzy ancorado nos limites
documentados, episódios do historian (incidente + baseline), supervisão distante
por alarmes e eventos, treino multitarefa e relatório com as regras aprendidas.
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
from fuzzytf.data import u200
from fuzzytf.data.dataset import DatasetConfig, EpisodeDataset
from fuzzytf.data.documents import KeywordRetriever
from fuzzytf.interpret import explain_sample, top_rules
from fuzzytf.model import FTIC
from fuzzytf.train import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="case_study")
    ap.add_argument("--out", default="runs/walk-u200")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--advisory-window-h", type=float, default=8.0)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--mixer", default="anfis", choices=["anfis", "mlp"])
    ap.add_argument("--no-train", action="store_true", help="só ingere e reporta")
    args = ap.parse_args()

    root = Path(args.path)
    if not root.exists():
        raise SystemExit(f"acervo não encontrado em {root}; rode scripts/setup_case_study.py")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] ingerindo o acervo")
    book = u200.variable_book()
    episodes = u200.episodes_from_case(root, advisory_window_h=args.advisory_window_h)
    tables = u200.load_tables(root)
    print(f"       {len(book)} tags · {book.n_state_slots} slots · fingerprint {book.fingerprint()}")
    for i, ep in enumerate(episodes):
        print(
            f"       episódio {i}: {len(ep)} amostras · anormal {int(ep.fault_active.sum())} "
            f"· banda estreitada {int((ep.band[:, 1] < 1).sum())}"
        )
    balance = {a: float(v) for a, v in zip(u200.ADVISORIES_U200, episodes[0].advisories.mean(0))}
    print("       balanço dos rótulos:", {k: round(v, 2) for k, v in balance.items()})

    print("[2/5] acervo documental")
    docs = u200.load_documents(root)
    retriever = KeywordRetriever(docs)
    print(f"       {len(docs)} documentos em {len({d.meta['silo'] for d in docs})} silos")
    hits = retriever.search("válvula curso alto saturação capacidade", tags=["FV-201"], top_k=5)
    for d, s in hits:
        print(f"       {s:6.2f}  {d.meta['silo']:12s} {d.doc_id}")
    (out_dir / "retrieval.json").write_text(
        json.dumps(
            [{"doc": d.doc_id, "silo": d.meta["silo"], "score": s} for d, s in hits],
            indent=2,
            ensure_ascii=False,
        )
    )

    ds_cfg = DatasetConfig(window=args.window, delta_scale=0.05, stride=1)
    # Divisão temporal: o incidente (fim da série) é validação; nunca aleatória.
    split = int(len(episodes[0]) * 0.7)
    train_eps = [_slice(episodes[0], 0, split)] + episodes[1:]
    val_eps = [_slice(episodes[0], split, len(episodes[0]))]
    train_set = EpisodeDataset(train_eps, book, ds_cfg, action_tag=u200.ACTION_TAG)
    val_set = EpisodeDataset(val_eps, book, ds_cfg, action_tag=u200.ACTION_TAG)
    print(f"       treino {len(train_set)} amostras · validação {len(val_set)}")

    if args.no_train:
        return

    print("[3/5] modelo")
    cfg = ExperimentConfig(name="walk-u200", stage="walk")
    cfg.tokenizer.window = args.window
    cfg.model.d_model = args.d_model
    cfg.model.n_blocks = args.blocks
    cfg.model.mixer = args.mixer
    cfg.model.n_advisories = len(u200.ADVISORIES_U200)
    cfg.model.n_fault_classes = 2
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = 32
    model = FTIC(book, cfg.model, cfg.tokenizer, action_tags=[u200.ACTION_TAG])
    model.set_topology_prior(torch.as_tensor(u200.adjacency(book)), strength=0.5)
    print(f"       parâmetros: {model.n_parameters():,} · mixer={cfg.model.mixer}")

    print("[4/5] treinando")
    result = train(model, train_set, val_set, cfg, out_dir=out_dir)

    print("[5/5] interpretabilidade")
    sample = val_set[len(val_set) // 2]
    (out_dir / "explicacao.txt").write_text(
        explain_sample(
            model, book, sample, ds_cfg,
            advisories=u200.ADVISORIES_U200,
            faults=("normal", "anormal"),
        )
    )
    (out_dir / "rules.txt").write_text(top_rules(model, val_set, ds_cfg, n=12))
    (out_dir / "report.json").write_text(
        json.dumps(
            {"open_loop": result["final"], "label_balance": balance, "config": cfg.to_dict()},
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\nresultados em {out_dir}/")


def _slice(ep, a: int, b: int):
    """Recorta um episódio no tempo, preservando a interface usada pelo dataset."""
    import copy

    new = copy.copy(ep)
    new.series = {k: v[a:b] for k, v in ep.series.items()}
    for attr in ("u_command", "valve_true", "T_true", "setpoint", "fault_active"):
        setattr(new, attr, getattr(ep, attr)[a:b])
    new.band = ep.band[a:b]
    new.advisories = ep.advisories[a:b]
    new.hist = ep.hist
    return new


if __name__ == "__main__":
    main()
