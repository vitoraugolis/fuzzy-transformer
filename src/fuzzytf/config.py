"""Configuração do FT-IC (Fuzzy Transformer for Industrial Control).

Todos os hiperparâmetros ficam em dataclasses serializáveis para JSON, para que
cada experimento seja reproduzível a partir de um único arquivo
(`experiments/<nome>/config.json`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TokenizerConfig:
    """Janela temporal e geometria da tokenização."""

    window: int = 32          # n+1 instantes (k-n ... k)
    top_p: int = 3            # slots fuzzy retidos por dimensão/token
    layout: str = "tag_major"
    use_value_channel: bool = True   # canal numérico residual (QP-3)
    use_lag_embedding: bool = True   # embedding de defasagem temporal


@dataclass
class AnfisConfig:
    """Camada neuro-fuzzy (substituta do MLP)."""

    n_rules: int = 64          # tamanho do banco de regras por cabeça
    n_axes: int = 8            # eixos antecedentes projetados do embedding
    n_mfs: int = 3             # MFs gaussianas por eixo
    n_heads: int = 4           # bancos de regras independentes (multi-head fuzzy)
    consequent: str = "tsk1"   # "tsk0" | "tsk1"
    rank: int = 8              # posto das matrizes de consequente TSK-1
    hard_rules: bool = False   # seleção discreta (Gumbel) em vez de soft
    firing_temperature: float = 1.0
    dropout: float = 0.0


@dataclass
class AttentionConfig:
    n_heads: int = 8
    dropout: float = 0.0
    causal: bool = False       # a janela é observada por inteiro; ver QP-5
    use_tag_bias: bool = True  # viés relacional por par de tags (topologia/P&ID)


@dataclass
class ModelConfig:
    d_model: int = 256
    n_blocks: int = 4
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    anfis: AnfisConfig = field(default_factory=AnfisConfig)
    n_actions: int = 1              # nº de variáveis manipuladas
    action_terms: int = 7           # termos linguísticos da ação (saída fuzzy)
    n_advisories: int = 0           # nº de rótulos de orientação (multi-label)
    n_fault_classes: int = 0        # cabeça auxiliar de diagnóstico (0 = desligada)
    mixer: str = "anfis"            # "anfis" | "mlp" (ablação: MLP no lugar do ANFIS)
    mlp_ratio: float = 4.0          # usado apenas quando mixer == "mlp"
    n_context_slots: int = 0        # tokens de contexto documental (Walk/Run)
    dropout: float = 0.1
    norm_eps: float = 1e-5


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 64
    epochs: int = 20
    warmup_steps: int = 200
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "cpu"
    # pesos das perdas multitarefa
    w_action: float = 1.0
    w_band: float = 0.5          # envelope tem peso próprio (ver QP-10)
    w_advisory: float = 0.5
    w_forecast: float = 0.2      # pré-treino: prever estado fuzzy em k+1
    w_masked_state: float = 0.2  # pré-treino: MLM sobre tokens de estado
    w_rule_entropy: float = 1e-3 # regulariza o banco de regras (esparsidade)


@dataclass
class ExperimentConfig:
    name: str = "crawl-01"
    stage: str = "crawl"          # crawl | walk | run
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    variable_book: Optional[str] = None  # caminho do vocabulário fuzzy

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ExperimentConfig":
        d = dict(d)
        tok = TokenizerConfig(**d.pop("tokenizer", {}))
        mdl = dict(d.pop("model", {}))
        att = AttentionConfig(**mdl.pop("attention", {}))
        anf = AnfisConfig(**mdl.pop("anfis", {}))
        model = ModelConfig(attention=att, anfis=anf, **mdl)
        train = TrainConfig(**d.pop("train", {}))
        return ExperimentConfig(tokenizer=tok, model=model, train=train, **d)

    @staticmethod
    def load(path: str | Path) -> "ExperimentConfig":
        return ExperimentConfig.from_dict(json.loads(Path(path).read_text()))
