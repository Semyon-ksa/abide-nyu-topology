# What do persistent homology features measure in resting-state fMRI?

Code for the main results of

> S. A. Kurkin, N. M. Smirnov, V. S. Khorev, Z. Dashdorj, S. Chen, A. E. Hramov.
> *What do persistent homology features measure in resting-state fMRI? Recording length,
> surrogate data and null models.* Chaos, Solitons & Fractals (submitted, manuscript CHAOS-D-26-11287).

The repository reproduces the three main results of the paper on 164 participants of the
ABIDE I NYU sample:

| Result | Paper | Script | Output |
|---|---|---|---|
| Zigzag barcodes of real recordings do not differ from multivariate phase-randomised surrogates | Sec. 4.4, Fig. 2 | `01_zigzag_surrogates.py`, `02_zigzag_tests.py` | `m2_full_cohort_*.csv`, `figures/fig2` |
| β0 is reproduced by a degree-preserving null, β1 is about one fifth of it | Sec. 4.5–4.7, Fig. 5 | `03_static_features.py`, `04_static_nulls.py` | `static_*.csv`, `null_density_robustness*.csv`, `figures/fig5` |
| No topological feature separates autism from controls | Sec. 4.8, 4.10, Fig. 6 | `05_liang_configuration.py`, `06_group_contrast.py` | `liang_eps_*.csv`, `group_*.csv`, `figures/fig6` |

Zigzag persistence is computed with [tdadyn](https://github.com/Semyon-ksa/tdadyn), an
open-source package by the same authors, with Dionysus 2 as its backend. Everything else in
`scripts/` (windowed graphs, surrogates, null models, statistics) is self-contained.

## Installation

```bash
conda env create -f environment.yml
conda activate abide-topology
pip install "git+https://github.com/Semyon-ksa/tdadyn.git"
```

Tested with Python 3.12, NumPy 2.5, SciPy 1.18, pandas 3.0, NetworkX 3.6, statsmodels 0.15,
Dionysus 2.2.3, Matplotlib 3.11 and tdadyn 0.1.0.

## Data

The analysis starts from fMRIPrep 25.2.5 derivatives of the ABIDE I NYU resting-state scans.
ABIDE data are available from the [ABIDE / INDI website](https://fcon_1000.projects.nitrc.org/indi/abide/)
after registration; the phenotypic file is distributed with them.

1. Run fMRIPrep on the NYU participants.
2. Run the post-processing script, which regresses nuisance signals (24 motion parameters,
   five aCompCor components and the cosine high-pass regressors in one model), low-pass
   filters at 0.10 Hz, builds the censoring mask from fMRIPrep's motion-outlier flags,
   extracts AAL3 regional time series and computes static Fisher-z matrices, without and
   with global signal regression (it also needs NiBabel and Nilearn; the paper used Nilearn 0.11.1):

   ```bash
   python scripts/00_postprocess_fmriprep.py --derivatives <fmriprep_dir> --out data/postproc_data_zigzag \
          --atlas-dir <aal3_dir> --n-jobs 8
   ```

3. Point the analysis at the outputs:

   ```bash
   export ABIDE_POSTPROC=data/postproc_data_zigzag
   export ABIDE_PHENO=data/phenotypic_NYU.csv      # ABIDE phenotypic file, NYU rows
   ```

Participants are included if their median framewise displacement is at most 0.14 mm and at
least 120 volumes survive censoring (164 participants: 67 autistic, 97 controls).

## Running

```bash
make all          # all analyses and Figs. 2, 5, 6 (results/ and results/figures/)
make static       # static features, null models, Liang configuration, group contrast (~3 min on 10 cores)
make zigzag       # zigzag barcodes, 164 participants x 21 series x 2 rules (~40 CPU hours)
make verify       # compare results/ with the tables behind the paper in published/
```

`N_JOBS` sets the number of processes (default: number of cores minus two). A quick check of
the zigzag pipeline on a few participants:

```bash
cd scripts && python 01_zigzag_surrogates.py --subjects sub-0050952 sub-0050954 --n-surr 20
```

## Reproducibility

`published/` holds the tables from which the numbers and figures in the paper were produced.
`make verify` compares a fresh run with them.

* Static features, null models, the Liang configuration, the group contrast and the zigzag
  statistics of the **real** recordings are deterministic and agree to numerical precision.
* The **surrogate** draws of the published zigzag run were seeded with Python's `hash()` of
  the participant identifier, which is salted per process and cannot be reproduced. This
  repository seeds them with a CRC32 checksum instead, so individual surrogate values differ
  from the published run; the cohort-level comparison of real and surrogate barcodes, which
  is the result of the paper, does not depend on the draws.


## Layout

```
scripts/common.py          paths, inclusion criterion, data loading
scripts/zztools.py         windows, adjacency rules, surrogates, zigzag barcodes, null models
scripts/00_...06_*.py      analysis steps, in order
scripts/verify.py          comparison with published/
figures/fig{2,5,6}.py      figures of the paper
published/                 result tables behind the paper
```

## License

MIT (see `LICENSE`). If you use this code, please cite the paper above and tdadyn.
