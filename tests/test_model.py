"""Testes do modelo: formas, gradientes, estabilidade e ablação de mixer."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from fuzzytf.config import ModelConfig, TokenizerConfig
from fuzzytf.data import (
    DatasetConfig,
    EpisodeDataset,
    LoopSimulator,
    collate,
    default_variable_book,
    topology_adjacency,
)
from fuzzytf.model import FTIC
from fuzzytf.model.anfis import AnfisLayer
from fuzzytf.train.losses import mask_state_tokens, total_loss


@pytest.fixture(scope="module")
def fixtures():
    book = default_variable_book()
    eps = LoopSimulator(seed=0).dataset(2, 120)
    ds = EpisodeDataset(eps, book, DatasetConfig(window=8, stride=4))
    return book, ds


def build(book, mixer="anfis", **kw):
    cfg = ModelConfig(d_model=32, n_blocks=2, mixer=mixer, n_advisories=8, n_fault_classes=5, **kw)
    cfg.attention.n_heads = 2
    cfg.anfis.n_heads = 2
    cfg.anfis.n_rules = 8
    cfg.anfis.n_axes = 3
    return FTIC(book, cfg, TokenizerConfig(window=8), action_tags=["V-097"])


def test_forward_shapes(fixtures):
    book, ds = fixtures
    m = build(book)
    b = collate([ds[i] for i in range(4)])
    out = m(b, return_trace=True)
    S = b["tag_index"].shape[1]
    assert out.control.delta.shape == (4, 1)
    assert out.control.band_lo.shape == (4, 1)
    assert out.advisory.logits.shape == (4, 8)
    assert out.fault_logits.shape == (4, 5)
    assert out.state_logits.shape == (4, S, book.n_state_slots + 1)
    assert out.hidden.shape[1] == S + 2          # [CLS] + 1 × [ACT]
    assert len(out.traces) == 2


def test_acao_recortada_no_envelope(fixtures):
    book, ds = fixtures
    m = build(book)
    b = collate([ds[i] for i in range(4)])
    out = m(b)
    assert (out.control.delta >= out.control.band_lo - 1e-5).all()
    assert (out.control.delta <= out.control.band_hi + 1e-5).all()
    assert (out.control.band_lo < out.control.band_hi).all()


def test_limites_duros_nao_podem_ser_alargados(fixtures):
    book, ds = fixtures
    m = build(book)
    b = collate([ds[i] for i in range(4)])
    hard = torch.tensor([[[-0.05, 0.05]]]).expand(4, 1, 2)
    out = m(b, hard_limits=hard)
    assert (out.control.band_lo >= -0.05 - 1e-6).all()
    assert (out.control.band_hi <= 0.05 + 1e-6).all()


def test_gradiente_chega_a_todos_os_parametros(fixtures):
    book, ds = fixtures
    m = build(book)
    b = mask_state_tokens(collate([ds[i] for i in range(4)]))
    out = m(b)
    from fuzzytf.config import TrainConfig

    total_loss(m, out, b, TrainConfig()).total.backward()
    sem_grad = [n for n, p in m.named_parameters() if p.grad is None]
    assert not sem_grad, f"sem gradiente: {sem_grad}"
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_forcas_de_disparo_sao_distribuicao():
    layer = AnfisLayer(32, ModelConfig(d_model=32).anfis)
    x = torch.randn(2, 5, 32)
    y, trace = layer(x, return_trace=True)
    assert y.shape == x.shape
    assert torch.allclose(trace.firing.sum(-1), torch.ones_like(trace.firing.sum(-1)), atol=1e-5)
    assert torch.isfinite(trace.log_firing).all()


def test_anfis_estavel_com_muitos_eixos():
    """Muitos eixos multiplicam pertinências pequenas — sem log-espaço, dá NaN."""
    cfg = ModelConfig(d_model=32).anfis
    cfg.n_axes, cfg.n_rules, cfg.n_heads = 16, 8, 2
    layer = AnfisLayer(32, cfg)
    y, trace = layer(torch.randn(2, 4, 32) * 5.0, return_trace=True)
    assert torch.isfinite(y).all() and torch.isfinite(trace.firing).all()
    y.sum().backward()
    assert torch.isfinite(layer.rule_logits.grad).all()


def test_mixer_mlp_tem_mesma_interface(fixtures):
    book, ds = fixtures
    b = collate([ds[i] for i in range(2)])
    out = build(book, mixer="mlp")(b, return_trace=True)
    assert out.control.delta.shape == (2, 1)
    assert out.traces[0].anfis is None


def test_mixer_invalido_falha(fixtures):
    book, _ = fixtures
    with pytest.raises(ValueError):
        build(book, mixer="quantum")


def test_prior_de_topologia(fixtures):
    book, ds = fixtures
    m = build(book)
    m.set_topology_prior(torch.as_tensor(topology_adjacency(book)), strength=0.7)
    bias = m.blocks[0].attn.tag_bias.detach()
    i, j = book.index("T-102"), book.index("V-097")
    assert float(bias[0, i, j]) == pytest.approx(0.7)
    assert float(bias[0, i, len(book)]) == 0.0     # tokens especiais ficam neutros


def test_tokens_invalidos_nao_geram_nan(fixtures):
    book, ds = fixtures
    m = build(book)
    b = collate([ds[i] for i in range(2)])
    b["valid"] = torch.zeros_like(b["valid"])      # todos os instrumentos em falha
    out = m(b)
    assert torch.isfinite(out.control.delta).all()


def test_entropia_de_regras_diminui_com_logits_nitidos(fixtures):
    book, _ = fixtures
    m = build(book)
    antes = float(m.rule_entropy().detach())
    with torch.no_grad():
        for blk in m.blocks:
            blk.anfis.rule_logits.mul_(50.0)
    assert float(m.rule_entropy().detach()) < antes
