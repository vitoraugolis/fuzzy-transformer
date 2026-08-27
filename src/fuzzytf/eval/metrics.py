"""Métricas de avaliação.

Separadas em três famílias, porque o modelo tem três promessas distintas e
falhar em qualquer uma delas invalida a proposta:

1. **Malha aberta** — o quanto a ação prevista se parece com a do professor.
2. **Malha fechada** — o quanto a planta se comporta melhor com o modelo no
   lugar do controlador (é aqui que a arquitetura se justifica ou não).
3. **Orientação e antecipação** — qualidade das orientações e, sobretudo,
   *quantas amostras antes do alarme* a falha foi sinalizada.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


# --- malha aberta ---------------------------------------------------------

def regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred, target = np.asarray(pred, float).ravel(), np.asarray(target, float).ravel()
    err = pred - target
    var = target.var()
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "r2": float(1.0 - (err**2).mean() / var) if var > 1e-12 else float("nan"),
    }


def band_metrics(lo: np.ndarray, hi: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Qualidade do envelope de contingência.

    ``violation_rate`` é a métrica de segurança: fração de amostras em que o
    modelo autorizaria uma ação fora da faixa admissível.
    """
    lo, hi = np.asarray(lo, float).ravel(), np.asarray(hi, float).ravel()
    t_lo, t_hi = np.asarray(target, float)[:, 0], np.asarray(target, float)[:, 1]
    over = (hi > t_hi + 1e-3) | (lo < t_lo - 1e-3)
    width_ratio = (hi - lo) / np.maximum(t_hi - t_lo, 1e-6)
    return {
        "violation_rate": float(over.mean()),
        "mean_width_ratio": float(np.mean(width_ratio)),
        "band_mae": float((np.abs(lo - t_lo) + np.abs(hi - t_hi)).mean() / 2),
    }


# --- orientações ----------------------------------------------------------

def multilabel_metrics(logits: np.ndarray, target: np.ndarray, threshold: float = 0.0) -> Dict[str, float]:
    pred = np.asarray(logits) > threshold
    tgt = np.asarray(target) > 0.5
    tp = np.logical_and(pred, tgt).sum(axis=0)
    fp = np.logical_and(pred, ~tgt).sum(axis=0)
    fn = np.logical_and(~pred, tgt).sum(axis=0)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    support = tgt.sum(axis=0)
    present = support > 0
    return {
        "f1_macro": float(f1[present].mean()) if present.any() else float("nan"),
        "f1_micro": float(
            2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)
        ),
        "exact_match": float((pred == tgt).all(axis=1).mean()),
        "hamming": float((pred != tgt).mean()),
    }


def accuracy(logits: np.ndarray, target: np.ndarray) -> float:
    return float((np.asarray(logits).argmax(axis=-1) == np.asarray(target)).mean())


# --- antecipação ----------------------------------------------------------

def detection_lead_time(
    fault_scores: Sequence[float],
    fault_start: int,
    alarm_flags: Sequence[bool],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Quantas amostras o modelo antecipa em relação ao alarme convencional.

    Esta é *a* métrica que sustenta a motivação do projeto: detectar a válvula
    saturando fora de faixa **antes** do alarme de temperatura.

    Devolve NaN nos campos correspondentes quando não houve detecção ou não
    houve alarme no episódio — a agregação deve usar ``np.nanmean``.
    """
    s = np.asarray(fault_scores, float)
    alarm = np.asarray(alarm_flags, bool)
    det = np.argmax(s > threshold) if (s > threshold).any() else -1
    alm = np.argmax(alarm) if alarm.any() else -1
    out = {
        "detected_at": float(det) if det >= 0 else float("nan"),
        "alarm_at": float(alm) if alm >= 0 else float("nan"),
        "detection_delay": float(det - fault_start) if det >= 0 else float("nan"),
        "lead_over_alarm": float(alm - det) if (det >= 0 and alm >= 0) else float("nan"),
        "false_alarm": float(det >= 0 and det < fault_start),
    }
    return out


# --- malha fechada --------------------------------------------------------

def closed_loop_metrics(
    measurement: np.ndarray,
    setpoint: np.ndarray,
    action: np.ndarray,
    alarm_hi: Optional[float] = None,
    warmup: int = 50,
) -> Dict[str, float]:
    y = np.asarray(measurement, float)[warmup:]
    sp = np.asarray(setpoint, float)[warmup:]
    u = np.asarray(action, float)[warmup:]
    e = y - sp
    m = {
        "iae": float(np.abs(e).mean()),
        "ise": float((e**2).mean()),
        "max_abs_error": float(np.abs(e).max()),
        "control_effort": float(np.abs(np.diff(u)).sum()),
        "reversals": float((np.diff(np.sign(np.diff(u))) != 0).sum()),
    }
    if alarm_hi is not None:
        m["alarm_fraction"] = float((y > alarm_hi).mean())
    return m
