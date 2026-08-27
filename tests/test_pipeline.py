"""Testes do pipeline: simulador, dataset, treino, rollout e interpretação."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch")

from fuzzytf.config import ExperimentConfig
from fuzzytf.data import (
    ADVISORIES,
    DatasetConfig,
    EpisodeDataset,
    FaultSpec,
    KeywordRetriever,
    Document,
    LoopSimulator,
    collate,
    default_variable_book,
    hazop_to_rules,
    split_episodes,
)
from fuzzytf.eval import (
    ModelPolicy,
    PIPolicy,
    band_metrics,
    closed_loop_metrics,
    detection_lead_time,
    regression_metrics,
)
from fuzzytf.interpret import explain_sample, top_rules
from fuzzytf.model import FTIC
from fuzzytf.train import train


def small_cfg():
    cfg = ExperimentConfig(name="teste")
    cfg.tokenizer.window = 8
    cfg.model.d_model = 32
    cfg.model.n_blocks = 1
    cfg.model.attention.n_heads = 2
    cfg.model.anfis.n_heads = 2
    cfg.model.anfis.n_rules = 8
    cfg.model.anfis.n_axes = 3
    cfg.model.n_advisories = len(ADVISORIES)
    cfg.model.n_fault_classes = 5
    cfg.train.epochs = 1
    cfg.train.batch_size = 8
    return cfg


def test_simulador_falha_limita_curso():
    sim = LoopSimulator(seed=1)
    ep = sim.run(300, fault=FaultSpec(kind="valve_travel_limit", start=100, travel_limit=0.4))
    assert ep.series["V-097"][150:].max() <= 0.4 + 1e-6
    assert ep.band[150:, 1].max() < 1.0            # envelope estreitado
    assert ep.advisories[150:].sum() > 0           # orientações emitidas


def test_simulador_sem_falha_segue_setpoint():
    ep = LoopSimulator(seed=2).run(400, fault=FaultSpec(kind="none"), setpoint_changes=False)
    erro = np.abs(ep.series["T-102"] - ep.setpoint)[200:]
    assert erro.mean() < 1.0


def test_politica_externa_fecha_a_malha():
    estado = {"i": 0.0}

    def politica(k, series, sp, u_prev, band):
        e = sp - series["T-102"][-1]
        estado["i"] += e
        return float(np.clip(0.08 * e + 0.01 * estado["i"], band[0], band[1]))

    ep = LoopSimulator(seed=2).rollout(politica, 400, fault=FaultSpec(kind="none"),
                                       setpoint_changes=False)
    assert np.abs(ep.series["T-102"] - ep.setpoint)[200:].mean() < 1.0


def test_split_por_episodio_nao_mistura():
    eps = LoopSimulator(seed=3).dataset(6, 60)
    tr, va = split_episodes(eps, val_fraction=0.34, seed=0)
    assert len(tr) + len(va) == 6
    assert not ({id(e) for e in tr} & {id(e) for e in va})


def test_dataset_alvos_coerentes():
    book = default_variable_book()
    eps = LoopSimulator(seed=4).dataset(1, 80)
    ds = EpisodeDataset(eps, book, DatasetConfig(window=8))
    item = ds[10]
    assert item["target_band"][0] <= item["target_band"][1]
    assert -1.0 <= float(item["target_delta"]) <= 1.0
    assert item["forecast_slots"].shape[0] == len(book)      # um token por tag em k
    assert ds.stats()["samples"] == len(ds)


def test_treino_reduz_a_perda_e_salva(tmp_path):
    book = default_variable_book()
    eps = LoopSimulator(seed=5).dataset(4, 120)
    tr, va = split_episodes(eps, val_fraction=0.5, seed=0)
    cfg = small_cfg()
    ds_cfg = DatasetConfig(window=cfg.tokenizer.window, stride=4)
    train_set = EpisodeDataset(tr, book, ds_cfg)
    val_set = EpisodeDataset(va, book, ds_cfg)
    model = FTIC(book, cfg.model, cfg.tokenizer, action_tags=["V-097"])
    res = train(model, train_set, val_set, cfg, out_dir=tmp_path, verbose=False)
    assert (tmp_path / "checkpoint.pt").exists()
    assert "delta_mae" in res["final"] and "skill_over_zero" in res["final"]
    assert np.isfinite(res["final"]["delta_mae"])


def test_rollout_com_modelo_produz_episodio():
    book = default_variable_book()
    cfg = small_cfg()
    model = FTIC(book, cfg.model, cfg.tokenizer, action_tags=["V-097"])
    policy = ModelPolicy(model, book, DatasetConfig(window=8))
    ep = LoopSimulator(seed=6).rollout(policy, 40, fault=FaultSpec(kind="none"))
    assert len(policy.trace) == 40
    m = closed_loop_metrics(ep.series["T-102"], ep.setpoint, ep.series["V-097"],
                            alarm_hi=108.0, warmup=5)
    assert set(m) >= {"iae", "control_effort", "alarm_fraction"}


def test_interpretacao_produz_texto():
    book = default_variable_book()
    cfg = small_cfg()
    eps = LoopSimulator(seed=7).dataset(1, 80)
    ds = EpisodeDataset(eps, book, DatasetConfig(window=8))
    model = FTIC(book, cfg.model, cfg.tokenizer, action_tags=["V-097"])
    texto = explain_sample(model, book, ds[20], DatasetConfig(window=8))
    assert "Ação de controle" in texto and "Regras dominantes" in texto
    regras = top_rules(model, ds, DatasetConfig(window=8), n=3, n_batches=1)
    assert "regra #" in regras and "dispara sobre" in regras


def test_metricas():
    r = regression_metrics(np.array([1.0, 2.0]), np.array([1.0, 3.0]))
    assert r["mae"] == pytest.approx(0.5)
    b = band_metrics(np.array([-1.0]), np.array([1.0]), np.array([[-0.5, 0.5]]))
    assert b["violation_rate"] == 1.0                     # faixa larga demais
    d = detection_lead_time([0.1, 0.9, 0.9], 1, [False, False, True])
    assert d["lead_over_alarm"] == 1.0 and d["detection_delay"] == 0.0
    vazio = detection_lead_time([0.1, 0.1], 1, [False, False])
    assert np.isnan(vazio["lead_over_alarm"])


def test_hazop_vira_regra():
    linhas = [{"node": "N1", "deviation": "Mais temperatura", "cause": "falha V-097",
               "consequence": "alívio", "recommendation": "manutenção da válvula"}]
    regras = hazop_to_rules(linhas, known_tags=["V-097", "T-102"])
    assert regras and regras[0].antecedent[0][0] == "V-097"
    assert "acionar_manutencao_valvula" in regras[0].consequent


def test_recuperador_prioriza_documento_da_tag():
    docs = [Document("A", "wo", "válvula travando manutenção mecânica", tags=["V-097"]),
            Document("B", "wo", "válvula travando manutenção mecânica", tags=["X-001"])]
    hits = KeywordRetriever(docs).search("válvula manutenção", tags=["V-097"], top_k=2)
    assert hits[0][0].doc_id == "A"
