"""Testes do núcleo fuzzy: partições, fuzzificação e vocabulário."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuzzytf.fuzzy import Fuzzifier, VariableBook
from fuzzytf.fuzzy.membership import Gaussian, MembershipFunction, Trapezoidal, Triangular
from fuzzytf.fuzzy.variables import (
    DEFAULT_LEVEL_TERMS,
    partition_from_breakpoints,
    ruspini_partition,
    standard_tag,
)


def test_membership_round_trip():
    for mf in (Triangular(0, 1, 2), Trapezoidal(0, 1, 2, 3), Gaussian(1.0, 0.5)):
        back = MembershipFunction.from_dict(mf.to_dict())
        assert np.allclose(back(np.linspace(-1, 4, 20)), mf(np.linspace(-1, 4, 20)))


def test_ruspini_partition_soma_um():
    var = ruspini_partition(0.0, 10.0, DEFAULT_LEVEL_TERMS)
    mu = var.memberships(np.linspace(0, 10, 51), normalize=False)
    assert np.allclose(mu.sum(axis=-1), 1.0, atol=1e-6)


def test_partition_por_breakpoints_centra_nos_pontos():
    var = partition_from_breakpoints([94, 98, 100.5, 103.5, 108], DEFAULT_LEVEL_TERMS)
    for i, c in enumerate([94, 98, 100.5, 103.5, 108]):
        assert int(np.argmax(var.memberships(c))) == i


def test_breakpoints_precisam_ser_crescentes():
    with pytest.raises(ValueError):
        partition_from_breakpoints([1, 1, 2], ["a", "b", "c"])


def test_descricao_linguistica():
    u = standard_tag("T-102", "temperature", 80, 120)
    mu = u.variables["level"].memberships(101.563)
    desc = u.variables["level"].describe(mu)
    assert "%" in desc and any(t in desc for t in u.variables["level"].terms)


def test_book_fingerprint_e_serializacao(tmp_path):
    book = VariableBook([standard_tag("T-102", "temperature", 80, 120)])
    path = tmp_path / "book.json"
    book.save(path)
    assert VariableBook.load(path).fingerprint() == book.fingerprint()


def test_book_rejeita_tag_duplicada():
    with pytest.raises(ValueError):
        VariableBook([standard_tag("T", "x", 0, 1), standard_tag("T", "x", 0, 1)])


def test_fuzzificacao_esparsa_normalizada():
    book = VariableBook([standard_tag("T-102", "temperature", 80, 120)])
    fz = Fuzzifier(book, top_p=2)
    fw = fz.transform({"T-102": [100.0, 101.0, 102.0, 103.0]})
    assert len(fw) == 4                       # 1 tag × 4 instantes
    assert (fw.lag == [3, 2, 1, 0]).all()     # lag 0 é o instante k
    for row in range(len(fw)):
        for dim_start in (0, 2):              # level e trend, top_p=2 cada
            w = fw.weights[row, dim_start : dim_start + 2]
            assert abs(w.sum() - 1.0) < 1e-5


def test_nan_marca_invalido_sem_propagar():
    book = VariableBook([standard_tag("T-102", "temperature", 80, 120)])
    fz = Fuzzifier(book)
    fw = fz.transform({"T-102": [100.0, np.nan, 102.0]})
    assert not fw.valid[1] and fw.valid[0] and fw.valid[2]
    assert np.isfinite(fw.weights).all() and np.isfinite(fw.value).all()


def test_layouts_produzem_mesmo_conjunto_de_tokens():
    book = VariableBook(
        [standard_tag("A", "x", 0, 1), standard_tag("B", "x", 0, 1)]
    )
    win = {"A": [0.1, 0.2, 0.3], "B": [0.4, 0.5, 0.6]}
    tag_major = Fuzzifier(book, layout="tag_major").transform(win)
    time_major = Fuzzifier(book, layout="time_major").transform(win)
    assert len(tag_major) == len(time_major) == 6
    assert sorted(zip(tag_major.tags, tag_major.lag)) == sorted(
        zip(time_major.tags, time_major.lag)
    )


def test_dimensao_error_exige_setpoint():
    from fuzzytf.data import default_variable_book

    book = default_variable_book()
    fz = Fuzzifier(book)
    sem = fz.transform({t: [1.0] * 4 for t in book.tags})
    com = fz.transform({t: [1.0] * 4 for t in book.tags}, setpoints={"T-102": 100.0})
    assert com.slot_ids.shape[1] >= sem.slot_ids.shape[1]
