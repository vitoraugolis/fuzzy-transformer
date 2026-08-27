#!/usr/bin/env python3
"""Diagnóstico do case study: o que já está lá e o que falta.

    python scripts/inspect_case_study.py --path ../case_study
    FTIC_CASE_STUDY=/caminho python scripts/inspect_case_study.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fuzzytf.data import case_study as cs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None)
    ap.add_argument("--load", action="store_true", help="também tenta carregar tags e séries")
    args = ap.parse_args()

    report = cs.validate(args.path)
    print(report)
    if not args.load or report.root is None:
        return

    try:
        process = cs.load_process(args.path)
        print(f"\nséries carregadas: {len(process)}")
        for tag, v in list(process.items())[:12]:
            finite = v[~(v != v)]
            print(f"  {tag:12s} n={len(v):7d} min={finite.min():10.3f} max={finite.max():10.3f}")
        book = cs.load_variable_book(args.path, process=process)
        print(f"\nvocabulário: {len(book)} tags, {book.n_state_slots} slots de estado")
        print(f"fingerprint: {book.fingerprint()}")
        events = cs.load_events(args.path)
        for k, v in events.items():
            print(f"  {k}: {len(v)} registros")
    except Exception as exc:  # noqa: BLE001 - diagnóstico deve reportar, não quebrar
        print(f"\nfalha ao carregar: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
