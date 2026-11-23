# Set ABIDE_POSTPROC and ABIDE_PHENO first (see README). N_JOBS sets the number of processes.
PY      := python
export PYTHONPATH := $(CURDIR)/scripts:$(PYTHONPATH)
RES     := $(or $(ABIDE_RESULTS),results)

.PHONY: all static zigzag figures verify clean

all: zigzag static figures

zigzag:
	$(PY) scripts/01_zigzag_surrogates.py
	$(PY) scripts/02_zigzag_tests.py

static:
	$(PY) scripts/03_static_features.py
	$(PY) scripts/04_static_nulls.py
	$(PY) scripts/05_liang_configuration.py
	$(PY) scripts/06_group_contrast.py   # needs $(RES)/m2_full_cohort_raw.csv from 'make zigzag'

figures:
	$(PY) figures/fig2.py
	$(PY) figures/fig5.py
	$(PY) figures/fig6.py

verify:
	$(PY) scripts/verify.py

clean:
	rm -rf $(RES)
