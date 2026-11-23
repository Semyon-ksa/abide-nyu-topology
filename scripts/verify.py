"""Compare recomputed tables in results/ with the tables behind the paper in published/.

Deterministic outputs (static features, null models, Liang configuration, group contrast and
the zigzag statistics of the real recordings) should agree to numerical precision. Surrogate
zigzag statistics depend on the random draws and are compared at the level of cohort summaries.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

PUB = Path(__file__).resolve().parents[1] / "published"
KEYS = {"static_tda_features.csv": ["subject", "variant"],
        "static_surrogate_test.csv": ["subject", "null", "stat"],
        "null_density_robustness_summary.csv": ["density", "stat"],
        "liang_eps_replication.csv": ["subject"],
        "group_balance.csv": ["variable"],
        "group_feature_validity.csv": ["feature"],
        "liang_eps_group.csv": ["feature"]}


def compare(name, keys, a, b, rtol=1e-6, atol=1e-9):
    m = a.merge(b, on=keys, suffixes=("_new", "_pub"))
    cols = [c[:-4] for c in m if c.endswith("_new") and pd.api.types.is_numeric_dtype(m[c])
            and c[:-4] + "_pub" in m]
    bad = {c: int((~np.isclose(m[c + "_new"], m[c + "_pub"], rtol=rtol, atol=atol, equal_nan=True)).sum())
           for c in cols}
    bad = {c: k for c, k in bad.items() if k}
    status = "OK " if not bad and len(m) == len(b) else "DIFF"
    print(f"{status} {name:40s} rows {len(m)}/{len(b)} | numeric columns {len(cols)}"
          + (f" | mismatches {bad}" if bad else ""))
    return status == "OK "


ok = True
for name, keys in KEYS.items():
    new = C.RESULTS / name
    if not new.exists():
        print(f"--   {name:40s} not computed")
        continue
    pub = pd.read_csv(PUB / name)
    if name == "group_feature_validity.csv":      # the paper's Fig. 6 uses configurations C and Z
        pub = pub[pub.config.isin(["C", "Z"])]
    if name == "static_surrogate_test.csv":       # nulls reported in Fig. 5
        pub = pub[pub.null.isin(["degree", "erdos_renyi"])]
    ok &= compare(name, keys, pd.read_csv(new), pub)

new = C.RESULTS / "m2_full_cohort_raw.csv"
if new.exists():
    a, b = pd.read_csv(new), pd.read_csv(PUB / "m2_full_cohort_raw.csv")
    b = b[b.subject.isin(a.subject.unique())]
    keys, stats = ["subject", "edge_rule"], ["n_snapshots", "n_bars_h0", "n_bars_h1", "total_length_h0",
                                             "total_length_h1", "invisible_bars", "censored_ends"]
    ok &= compare("zigzag, real recordings", keys, a[a.is_real][keys + stats], b[b.is_real][keys + stats])
    sa = a[~a.is_real].groupby("edge_rule")[["n_bars_h1", "total_length_h1"]].mean()
    sb = b[~b.is_real].groupby("edge_rule")[["n_bars_h1", "total_length_h1"]].mean()
    print("zigzag, surrogate means (recomputed vs published; random draws differ):")
    print(pd.concat({"recomputed": sa, "published": sb}, axis=1).round(2).to_string())
sys.exit(0 if ok else 1)
