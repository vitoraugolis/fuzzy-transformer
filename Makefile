.PHONY: test smoke crawl u200 case clean

test:
	python -m pytest -q

smoke:
	python scripts/run_crawl.py --smoke --out runs/smoke

crawl:
	python scripts/run_crawl.py --episodes 40 --steps 600 --out runs/crawl-01

ablacao-mlp:
	python scripts/run_crawl.py --episodes 40 --steps 600 --mixer mlp --out runs/crawl-01-mlp

case:
	python scripts/setup_case_study.py

u200: case
	python scripts/run_u200.py --path case_study

clean:
	rm -rf runs .pytest_cache **/__pycache__
