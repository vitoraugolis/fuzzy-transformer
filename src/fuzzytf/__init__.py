"""FT-IC — Fuzzy Transformer for Industrial Control.

Arquitetura de transformer com tokenização fuzzy de dados de processo e camadas
neuro-fuzzy (ANFIS) no lugar do MLP, para gerar simultaneamente a ação de
controle e as orientações às equipes.

Ver `docs/01-arquitetura.md` para a especificação e
`docs/02-roadmap-crawl-walk-run.md` para o plano de investigação.
"""

__version__ = "0.1.0"

from .config import (
    AnfisConfig,
    AttentionConfig,
    ExperimentConfig,
    ModelConfig,
    TokenizerConfig,
    TrainConfig,
)
from .fuzzy import Fuzzifier, VariableBook
from .tokenizer import ProcessTokenizer

__all__ = [
    "__version__",
    "AnfisConfig",
    "AttentionConfig",
    "ExperimentConfig",
    "ModelConfig",
    "TokenizerConfig",
    "TrainConfig",
    "Fuzzifier",
    "VariableBook",
    "ProcessTokenizer",
]
