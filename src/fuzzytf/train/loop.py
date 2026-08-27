"""Laço de treino do estágio Crawl.

Deliberadamente simples e sem dependências além de PyTorch: o objetivo do
Crawl é isolar riscos de arquitetura, não otimizar infraestrutura. Escalonar
(AMP, múltiplas GPUs, checkpoint distribuído) é problema do estágio Run.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from ..config import ExperimentConfig
from ..data.dataset import EpisodeDataset, collate
from ..eval.metrics import (
    accuracy,
    band_metrics,
    multilabel_metrics,
    regression_metrics,
)
from .losses import LossTerms, mask_state_tokens, total_loss


def iterate(dataset: EpisodeDataset, batch_size: int, shuffle: bool, seed: int = 0, device: str = "cpu"):
    idx = np.arange(len(dataset))
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    for start in range(0, len(idx) - batch_size + 1, batch_size):
        chunk = idx[start : start + batch_size]
        yield collate([dataset[int(i)] for i in chunk], device=device)


def cosine_lr(step: int, warmup: int, total: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def advisory_pos_weight(dataset: EpisodeDataset, n_advisories: int, device: str = "cpu") -> torch.Tensor:
    """Compensa o desbalanceamento das orientações (a maioria é rara)."""
    pos = np.zeros(n_advisories)
    n = 0
    for ep in dataset.episodes:
        pos += ep.advisories.sum(axis=0)
        n += len(ep)
    neg = np.maximum(n - pos, 1.0)
    return torch.as_tensor(neg / np.maximum(pos, 1.0), dtype=torch.float32, device=device).clamp(
        max=20.0
    )


@torch.no_grad()
def evaluate(model, dataset: EpisodeDataset, cfg: ExperimentConfig) -> Dict[str, float]:
    model.eval()
    dev = cfg.train.device
    preds, targets, bands, band_t = [], [], [], []
    adv_p, adv_t, flt_p, flt_t = [], [], [], []
    for batch in iterate(dataset, cfg.train.batch_size, shuffle=False, device=dev):
        out = model(batch)
        preds.append(out.control.delta.cpu().numpy())
        targets.append(batch["target_delta"].cpu().numpy())
        bands.append(
            np.stack(
                [out.control.band_lo[:, 0].cpu().numpy(), out.control.band_hi[:, 0].cpu().numpy()],
                axis=1,
            )
        )
        band_t.append(batch["target_band"].cpu().numpy())
        if out.advisory is not None:
            adv_p.append(out.advisory.logits.cpu().numpy())
            adv_t.append(batch["target_advisory"].cpu().numpy())
        if out.fault_logits is not None:
            flt_p.append(out.fault_logits.cpu().numpy())
            flt_t.append(batch["target_fault"].cpu().numpy())

    m: Dict[str, float] = {}
    if preds:
        p, t = np.concatenate(preds), np.concatenate(targets)
        m.update({f"delta_{k}": v for k, v in regression_metrics(p, t).items()})
        # referência trivial: prever "não mexer". Um modelo que não bate isso
        # não aprendeu nada, por melhor que o MAE pareça em valor absoluto.
        m["delta_mae_zero_baseline"] = float(np.abs(t).mean())
        m["skill_over_zero"] = 1.0 - m["delta_mae"] / max(m["delta_mae_zero_baseline"], 1e-9)
        b, bt = np.concatenate(bands), np.concatenate(band_t)
        m.update(band_metrics(b[:, 0], b[:, 1], bt))
    if adv_p:
        m.update(multilabel_metrics(np.concatenate(adv_p), np.concatenate(adv_t)))
    if flt_p:
        m["fault_accuracy"] = accuracy(np.concatenate(flt_p), np.concatenate(flt_t))
    model.train()
    return m


def train(
    model,
    train_set: EpisodeDataset,
    val_set: EpisodeDataset,
    cfg: ExperimentConfig,
    out_dir: Optional[str | Path] = None,
    log_every: int = 20,
    verbose: bool = True,
) -> Dict[str, object]:
    tc = cfg.train
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    model.to(tc.device).train()

    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    steps_per_epoch = max(len(train_set) // tc.batch_size, 1)
    total_steps = steps_per_epoch * tc.epochs
    pw = (
        advisory_pos_weight(train_set, model.cfg.n_advisories, tc.device)
        if model.cfg.n_advisories > 0
        else None
    )

    history: List[Dict[str, float]] = []
    step = 0
    t0 = time.time()
    for epoch in range(tc.epochs):
        running: Dict[str, float] = {}
        for batch in iterate(train_set, tc.batch_size, shuffle=True, seed=tc.seed + epoch, device=tc.device):
            for g in opt.param_groups:
                g["lr"] = cosine_lr(step, tc.warmup_steps, total_steps, tc.lr)
            if tc.w_masked_state > 0:
                batch = mask_state_tokens(batch)
            out = model(batch)
            terms: LossTerms = total_loss(model, out, batch, tc, advisory_pos_weight=pw)
            opt.zero_grad(set_to_none=True)
            terms.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            opt.step()
            for k, v in terms.parts.items():
                running[k] = running.get(k, 0.0) + v
            step += 1
            if verbose and step % log_every == 0:
                msg = " ".join(f"{k}={v / log_every:.4f}" for k, v in sorted(running.items()))
                print(f"  passo {step:5d}/{total_steps} {msg}", flush=True)
                running = {}
        metrics = evaluate(model, val_set, cfg)
        metrics.update({"epoch": epoch, "step": step, "elapsed_s": time.time() - t0})
        history.append(metrics)
        if verbose:
            print(
                f"epoch {epoch:3d} | delta_mae={metrics.get('delta_mae', float('nan')):.4f} "
                f"(skill={metrics.get('skill_over_zero', float('nan')):+.3f}) "
                f"| viol={metrics.get('violation_rate', float('nan')):.3f} "
                f"| f1={metrics.get('f1_macro', float('nan')):.3f} "
                f"| falha={metrics.get('fault_accuracy', float('nan')):.3f}",
                flush=True,
            )

    result = {"history": history, "final": history[-1] if history else {}, "config": cfg.to_dict()}
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, out / "checkpoint.pt")
        (out / "history.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result
