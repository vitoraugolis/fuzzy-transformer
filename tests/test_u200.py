"""Testes do adaptador do estudo de caso U-200.

São pulados quando `case_study/` não foi extraído — o acervo é versionado como
zip e o diretório é gerado por `scripts/setup_case_study.py`.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuzzytf.data import u200

ROOT = Path(__file__).resolve().parents[1] / "case_study"
pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "historian.csv").exists(),
    reason="case_study/ não extraído (rode scripts/setup_case_study.py)",
)


def test_conversao_de_unidades_fecha():
    assert float(u200.x2_to_c(u200.c_to_x2(229.23))) == pytest.approx(229.23, abs=1e-6)
    assert float(u200.c_to_x2(u200.x2_to_c(4.7))) == pytest.approx(4.7, abs=1e-9)


def test_vocabulario_ancorado_nos_limites():
    book = u200.variable_book()
    pv = book["TT201_PV_C"].variables["level"]
    # o termo "muito_alto" está centrado no TAH-201
    assert int(np.argmax(pv.memberships(u200.TAH_C))) == len(pv.terms) - 1
    curso = book["ZT201_pct"].variables["level"]
    assert int(np.argmax(curso.memberships(u200.TRAVEL_ZAH * 100))) == len(curso.terms) - 2
    assert "error" in book["TT201_PV_C"].variables


def test_historian_carrega():
    h = u200.load_historian(ROOT)
    assert len(h) > 1000 and h.dt_seconds == 300.0
    assert {"TT201_PV_C", "ZT201_pct", u200.ACTION_TAG} <= set(h.series)
    assert all(np.isfinite(v).all() for v in h.series.values())


def test_episodios_com_incidente_e_baseline():
    eps = u200.episodes_from_case(ROOT)
    assert len(eps) == 2
    incidente, baseline = eps
    assert incidente.fault_active.sum() > 0
    assert baseline.fault_active.sum() == 0
    assert (incidente.band[:, 1] <= 1.0).all()
    # sob curso alto, o envelope se estreita
    alto = incidente.valve_true > u200.TRAVEL_ZAH
    assert alto.any() and incidente.band[alto, 1].max() < 1.0


def test_supervisao_distante_gera_rotulos_balanceados():
    ep = u200.episodes_from_case(ROOT, advisory_window_h=8.0)[0]
    frac = ep.advisories.mean(axis=0)
    assert frac.max() < 1.0, "rótulo saturado: janela Δ grande demais"
    assert (frac > 0).sum() >= 3


def test_achado_wo_anterior_a_janela():
    """As WOs que explicam o incidente precedem o historian (ver docs/08)."""
    h = u200.load_historian(ROOT)
    eventos = u200.load_tables(ROOT)["events"]
    inicio = h.timestamps[0]
    manutencao = [e for e in eventos if "MAINTENANCE" in e.get("event_type", "")]
    assert manutencao
    fv201 = [e for e in manutencao if "FV-201" in e.get("description", "")]
    assert fv201, "esperava ao menos um evento de manutenção da FV-201"
    assert all(u200._parse_ts(e["timestamp"]) < inicio for e in fv201)


def test_contexto_de_equipamento():
    ctx = u200.equipment_context(ROOT, datetime(2026, 8, 12), "FV-201")
    assert ctx["n_intervencoes"] >= 1
    assert 0 < ctx["dias_desde_ultima"] < 60


def test_documentos_por_silo():
    pytest.importorskip("pypdf")
    docs = u200.load_documents(ROOT)
    assert len(docs) >= 30
    silos = {d.meta["silo"] for d in docs}
    assert {"maintenance", "safety", "vendor", "engineering"} <= silos
    assert all(len(d.text) > 100 for d in docs)


def test_dataset_do_caso_real():
    from fuzzytf.data.dataset import DatasetConfig, EpisodeDataset

    book = u200.variable_book()
    eps = u200.episodes_from_case(ROOT)
    ds = EpisodeDataset(eps, book, DatasetConfig(window=16, stride=8), action_tag=u200.ACTION_TAG)
    item = ds[5]
    assert item["tag_index"].shape[0] == len(book) * 16
    assert np.isfinite(item["target_delta"])


def test_controlador_respeita_o_contrato():
    torch = pytest.importorskip("torch")
    from fuzzytf.config import ModelConfig, TokenizerConfig
    from fuzzytf.data.dataset import DatasetConfig
    from fuzzytf.integration import FTICController
    from fuzzytf.model import FTIC

    book = u200.variable_book()
    cfg = ModelConfig(d_model=32, n_blocks=1, n_advisories=len(u200.ADVISORIES_U200))
    cfg.attention.n_heads = 2
    cfg.anfis.n_heads = 2
    cfg.anfis.n_rules = 8
    cfg.anfis.n_axes = 3
    model = FTIC(book, cfg, TokenizerConfig(window=8), action_tags=[u200.ACTION_TAG])
    ctrl = FTICController(model, book, DatasetConfig(window=8, delta_scale=0.05))
    ctrl.reset({"q_bias": 0.32})

    obs = {"SP": 4.705, "PV": 4.700, "ZT201": 0.32, "FT201": 14.3,
           "TT203": 27.0, "TT207": 88.7, "TT204": 133.6, "AT205": 0.764}
    for _ in range(6):
        q = ctrl.update(obs, {"envelope": (224.0, 235.0)}, 300.0)
        assert 0.0 <= q <= 1.0
    assert len(ctrl.log) == 6
    assert set(ctrl.advisories_raised()) == set(u200.ADVISORIES_U200)
