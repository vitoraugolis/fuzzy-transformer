#!/usr/bin/env python3
"""Extrai `case_study.zip` para `case_study/` na raiz do repositório.

O acervo é versionado como zip (11 MB, 223 arquivos, muitos PDFs). O diretório
extraído fica em `.gitignore` — a fonte é sempre o zip.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(ROOT / "case_study.zip"))
    ap.add_argument("--dest", default=str(ROOT))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    zip_path, dest = Path(args.zip), Path(args.dest)
    target = dest / "case_study"
    if not zip_path.exists():
        raise SystemExit(f"não encontrei {zip_path}")
    if target.exists():
        if not args.force:
            print(f"{target} já existe (use --force para reextrair)")
            return
        shutil.rmtree(target)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    files = sum(1 for _ in target.rglob("*") if _.is_file())
    print(f"extraído: {target} ({files} arquivos)")
    for silo in sorted(p.name for p in target.iterdir() if p.is_dir()):
        n = sum(1 for _ in (target / silo).rglob("*") if _.is_file())
        print(f"  {silo:22s} {n:3d} arquivo(s)")


if __name__ == "__main__":
    main()
