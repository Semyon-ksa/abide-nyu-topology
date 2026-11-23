"""Betti numbers of thresholded static graphs against random-graph null models
(Sections 3.5 and 4.7, Fig. 5).

(a) Density 0.10: degree-preserving rewiring (double edge swaps, 5 per edge) and Erdos-Renyi
    graphs with the same number of edges, 20 realisations each per participant.
(b) Robustness: degree-preserving null at densities 0.05, 0.15 and 0.20.

Output: results/static_surrogate_test.csv, results/null_density_robustness.csv,
        results/null_density_robustness_summary.csv
"""
import zlib
from multiprocessing import Pool

import networkx as nx
import numpy as np
import pandas as pd

import common as C
import zztools as Z

N_NULL = 20
DENS_MAIN = 0.10
DENS_ROBUST = (0.05, 0.15, 0.20)


def null_degree(A, rng):
    g = nx.from_numpy_array((A > 0).astype(int))
    e = g.number_of_edges()
    nx.double_edge_swap(g, nswap=5 * e, max_tries=200 * e, seed=int(rng.integers(1e9)))
    return nx.to_numpy_array(g)


def null_er(A, rng):
    n = A.shape[0]
    iu = np.triu_indices(n, 1)
    m = np.zeros(len(iu[0]))
    m[rng.choice(len(iu[0]), int((A > 0).sum() // 2), replace=False)] = 1
    B = np.zeros((n, n))
    B[iu] = m
    return B + B.T


def betti(A):
    b = Z.zz_flag_betti(A)
    return b["beta0"], b["beta1"]


def main_density(args):
    si, s = args
    A = Z.zz_adjacency(C.load_static_z(s), density=DENS_MAIN, rule="density")
    rb0, rb1 = betti(A)
    rng = np.random.default_rng(1000 + si)
    rows = []
    for name, fn in [("degree", null_degree), ("erdos_renyi", null_er)]:
        vals = np.array([betti(fn(A, rng)) for _ in range(N_NULL)])
        for stat, real, v in [("beta0", rb0, vals[:, 0]), ("beta1", rb1, vals[:, 1])]:
            rows.append(dict(subject=s, null=name, stat=stat, **Z.zz_surrogate_test(real, v)))
    return rows


def robustness(s):
    Zc = C.load_static_z(s)
    rng = np.random.default_rng(20261027 + zlib.crc32(s.encode()) % 100000)
    rows = []
    for d in DENS_ROBUST:
        A = Z.zz_adjacency(Zc, density=d, rule="density")
        b = Z.zz_flag_betti(A)
        nb = [Z.zz_flag_betti(Z.zz_degree_null(A, rng)) for _ in range(N_NULL)]
        for st in ("beta0", "beta1"):
            v = np.array([x[st] for x in nb], float)
            sd = v.std(ddof=1)
            rows.append(dict(subject=s, density=d, stat=st, real=b[st], null_mean=v.mean(), null_sd=sd,
                             z=(b[st] - v.mean()) / sd if sd > 0 else np.nan,
                             outside=bool(b[st] < v.min() or b[st] > v.max())))
    return rows


if __name__ == "__main__":
    subs = C.included_subjects()
    with Pool(C.N_JOBS) as pool:
        SS = pd.DataFrame([r for rows in pool.map(main_density, list(enumerate(subs))) for r in rows])
        N = pd.DataFrame([r for rows in pool.map(robustness, subs) for r in rows])
    SS.to_csv(C.RESULTS / "static_surrogate_test.csv", index=False)
    N.to_csv(C.RESULTS / "null_density_robustness.csv", index=False)

    below = ("z", lambda z: float((z < 0).mean()))
    NR = N.groupby(["density", "stat"]).agg(real=("real", "mean"), null=("null_mean", "mean"),
                                           median_z=("z", "median"), frac_outside=("outside", "mean"),
                                           frac_below=below).reset_index()
    d10 = SS[SS.null == "degree"].groupby("stat").agg(real=("real", "mean"), null=("surr_mean", "mean"),
                                                      median_z=("z", "median"), frac_outside=("outside_range", "mean"),
                                                      frac_below=below).reset_index()
    d10.insert(0, "density", DENS_MAIN)
    NR = pd.concat([NR, d10]).sort_values(["stat", "density"])
    NR["ratio"] = NR.real / NR.null
    NR.to_csv(C.RESULTS / "null_density_robustness_summary.csv", index=False)
    print(SS.groupby(["null", "stat"]).agg(real=("real", "mean"), null=("surr_mean", "mean"),
                                           outside=("outside_range", "mean")).round(3))
    print(NR.round(3).to_string(index=False))
