PY ?= .venv/bin/python

.PHONY: data test test-slow report queue all

data:        ## regenerate the synthetic task (seeded)
	$(PY) -m basediff.data_gen --out data

test:        ## fast tests
	$(PY) -m pytest -q

test-slow:   ## also check that no row is truncated (~12 min)
	$(PY) -m pytest -q -m slow

report:      ## rebuild results/ from saved predictions (CPU, seconds)
	$(PY) -m basediff.report

queue:       ## full GPU pipeline, resumable (~8 h on a 4 GB GPU)
	bash scripts/overnight.sh

all: data test queue report
