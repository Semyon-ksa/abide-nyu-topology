"""Zigzag barcodes of real recordings and of multivariate phase-randomised surrogates
(Sections 3.3-3.4 and 4.4, Fig. 2).

For every included participant: 1 real recording + N_SURR surrogates, each turned into a
sequence of windowed graphs (W = 80 volumes, stride 20, windows with > 20% censored volumes
dropped) under two edge rules (density 0.15; kNN-OR calibrated to the same achieved density),
then into flag complexes joined through their unions, and summarised by its zigzag barcode.

Output: results/m2_full_cohort_raw.csv (one row per participant x series x rule).

Usage:  python scripts/01_zigzag_surrogates.py [--subjects sub-0050952 ...] [--n-surr 20]
Runtime: about 40 CPU hours for the full cohort (about 4 h on 10 cores).
"""
import argparse
import time
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd

import common as C
import zztools as Z

W, S, DENSITY, CENSOR_THRESH = 80, 20, 0.15, 0.20
RULES = ("density", "knn_or")
SEED = 20260920


def adjacencies(ts, censor, rule, kcache):
    starts = Z.zz_window_starts(ts.shape[0], W, S)
    ok = Z.zz_window_censor_fraction(censor, starts, W) <= CENSOR_THRESH
    corrs = Z.zz_window_correlations(ts, starts[ok], W)
    if len(corrs) < 2:
        return None
    if rule == "density":
        return np.stack([Z.zz_adjacency(c, density=DENSITY, rule="density") for c in corrs])
    n = corrs.shape[1]
    if n not in kcache:   # k is calibrated for each series so that its achieved density matches DENSITY
        kcache[n] = Z.zz_calibrate_knn(corrs, DENSITY, rule="knn_or", k_grid=list(range(5, 41)))["k"]
    return np.stack([Z.zz_adjacency(c, rule="knn_or", k=kcache[n]) for c in corrs])


def task(arg):
    subject, series_idx = arg
    ts, censor = C.load_timeseries(subject)
    if series_idx >= 0:
        # Deterministic, process-independent seed (the run reported in the paper used Python's
        # salted hash(); see README, so individual surrogate draws differ from the published run).
        rng = np.random.default_rng(SEED + zlib.crc32(subject.encode()) % 100000 + series_idx)
        ts = Z.zz_phase_randomise(ts, rng)
    out, kcache = [], {}
    for rule in RULES:
        A = adjacencies(ts, censor, rule, kcache)
        if A is None:
            continue
        t0 = time.perf_counter()
        st = Z.zz_barcode_statistics(A, max_homology_dim=1, max_nodes=400)
        out.append(dict(subject=subject, series=series_idx, is_real=series_idx < 0, edge_rule=rule,
                        n_nodes=int(ts.shape[1]), wall_s=time.perf_counter() - t0,
                        **st, **Z.zz_graph_diagnostics(A)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--n-surr", type=int, default=20)
    ap.add_argument("--out", default=str(C.RESULTS / "m2_full_cohort_raw.csv"))
    a = ap.parse_args()
    subs = a.subjects or C.included_subjects()
    tasks = [(s, i) for s in subs for i in [-1] + list(range(a.n_surr))]
    print(f"{len(subs)} participants, {len(tasks)} series, {C.N_JOBS} processes", flush=True)
    rows, t0 = [], time.time()
    with Pool(C.N_JOBS) as pool:
        for k, res in enumerate(pool.imap_unordered(task, tasks, chunksize=2), 1):
            rows.extend(res)
            if k % 50 == 0:
                el = time.time() - t0
                print(f"{k}/{len(tasks)}  {el/60:.1f} min  eta {el/k*(len(tasks)-k)/60:.0f} min", flush=True)
    df = pd.DataFrame(rows).sort_values(["subject", "edge_rule", "series"])
    df.to_csv(a.out, index=False)
    print("written", len(df), "rows to", a.out)


if __name__ == "__main__":
    main()
