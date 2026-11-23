"""The configuration of Liang et al. (2021): Vietoris-Rips filtration on d = 1 - r with
fixed thresholds eps = 0.7, 0.8, 0.9, i.e. r >= 0.3, 0.2, 0.1 (Section 4.8, Fig. 6).

Features per threshold: edge count and density, beta0 and beta1 alive at eps, and the
cumulative number of H1 bars born at or before eps. Mean and SD of r are kept as controls.

Output: results/liang_eps_replication.csv
"""
from multiprocessing import Pool

import dionysus as dio
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform

import common as C

EPS = (0.7, 0.8, 0.9)


def liang(s):
    r = np.tanh(C.load_static_z(s))
    D = 1.0 - r
    np.fill_diagonal(D, 0.0)
    f = dio.fill_rips(squareform(D, checks=False).astype("float64"), 2, max(EPS))
    dg = dio.init_diagrams(dio.homology_persistence(f), f)
    iu = np.triu_indices(D.shape[0], 1)
    out = dict(subject=s, n_nodes=D.shape[0], mean_r=float(r[iu].mean()), sd_r=float(r[iu].std()))
    for e in EPS:
        out[f"edges_e{e}"] = int((D[iu] <= e).sum())
        out[f"dens_e{e}"] = float((D[iu] <= e).mean())
        out[f"b0_e{e}"] = sum(1 for p in dg[0] if p.birth <= e < p.death)
        out[f"b1_e{e}"] = sum(1 for p in dg[1] if p.birth <= e < p.death)
        out[f"h1born_e{e}"] = sum(1 for p in dg[1] if p.birth <= e)
    return out


if __name__ == "__main__":
    with Pool(C.N_JOBS) as pool:
        L = pd.DataFrame(pool.map(liang, C.included_subjects()))
    L.to_csv(C.RESULTS / "liang_eps_replication.csv", index=False)
    print(L.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
