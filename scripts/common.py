"""Paths, subject selection and data loading shared by all analysis scripts.

Set two environment variables before running (see README):

    ABIDE_POSTPROC  output directory of 00_postprocess_fmriprep.py
                    (contains qc_postproc.tsv and <variant>/<atlas>/{timeseries,connectivity})
    ABIDE_PHENO     ABIDE phenotypic file restricted to the NYU site (phenotypic_NYU.csv)

Results are written to ./results (override with ABIDE_RESULTS).
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

POSTPROC = Path(os.environ.get("ABIDE_POSTPROC", "data/postproc_data_zigzag"))
PHENO = Path(os.environ.get("ABIDE_PHENO", "data/phenotypic_NYU.csv"))
RESULTS = Path(os.environ.get("ABIDE_RESULTS", "results"))
RESULTS.mkdir(parents=True, exist_ok=True)

VARIANT, ATLAS = "noGSR_bp", "aal3"      # main pipeline: no global signal regression, AAL3
N_JOBS = int(os.environ.get("N_JOBS", max(1, (os.cpu_count() or 2) - 2)))


def qc_table():
    return pd.read_csv(POSTPROC / "qc_postproc.tsv", sep="\t")


def included_subjects():
    """Inclusion criterion of the paper: median FD <= 0.14 mm and >= 120 retained volumes (n = 164)."""
    qc = qc_table()
    return qc.loc[(qc.fd_median <= 0.14) & (qc.n_keep >= 120), "subject"].tolist()


def phenotype():
    ph = pd.read_csv(PHENO)
    ph["subject"] = "sub-" + ph.SUB_ID.astype(str).str.zfill(7)
    return ph


def load_static_z(subject, variant=VARIANT):
    """Static Fisher-z matrix; regions missing in >= half of the rows are dropped."""
    Z = np.load(POSTPROC / variant / ATLAS / "connectivity" / f"{subject}_fisherz.npy").astype(np.float64)
    ok = np.isnan(Z).sum(0) < Z.shape[0] * 0.5
    Zc = np.nan_to_num(Z[np.ix_(ok, ok)])
    Zc = 0.5 * (Zc + Zc.T)
    np.fill_diagonal(Zc, 0.0)
    return Zc


def load_timeseries(subject, variant=VARIANT):
    """Regional time series in real time (censored volumes kept) and the censoring mask."""
    d = POSTPROC / variant / ATLAS / "timeseries"
    ts = np.load(d / f"{subject}_ts_full.npy").astype(np.float64)
    censor = np.load(d / f"{subject}_censor.npy")
    return ts[:, ~np.isnan(ts).any(0)], censor
