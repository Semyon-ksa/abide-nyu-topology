"""Static persistent homology of the full-recording connectivity matrices (Sections 3.5, 4.5-4.6).

Density filtration: each pair is assigned the rank of its correlation, so thresholding the
Vietoris-Rips filtration at d is the same as keeping the strongest fraction d of pairs.
Betti numbers are read off the persistence diagram on a density grid 0.02..0.40.
Both preprocessing variants (without and with global signal regression) are computed.

Output: results/static_tda_features.csv
"""
import numpy as np
import pandas as pd
import dionysus as dio
from scipy.spatial.distance import squareform

import common as C

DGRID = np.round(np.arange(0.02, 0.405, 0.02), 3)
DMAX = 0.40


def rank_distance(Zc):
    n = Zc.shape[0]
    iu = np.triu_indices(n, 1)
    order = np.argsort(np.argsort(-Zc[iu]))       # 0 = strongest pair
    D = np.zeros((n, n))
    D[iu] = (order + 1) / len(order)
    return D + D.T


def diagrams(Zc):
    f = dio.fill_rips(squareform(rank_distance(Zc), checks=False).astype("float32"), 2, DMAX)
    dg = dio.init_diagrams(dio.homology_persistence(f), f)
    return {k: (np.array([[p.birth, p.death] for p in dg[k]]) if len(dg[k]) else np.zeros((0, 2)))
            for k in (0, 1)}


rows = []
for variant in ["noGSR_bp", "GSR_bp"]:
    for s in C.included_subjects():
        Zc = C.load_static_z(s, variant)
        bd = diagrams(Zc)
        alive = lambda b, d: int(((b[:, 0] <= d) & (b[:, 1] > d)).sum())
        b0c = np.array([alive(bd[0], d) for d in DGRID])
        b1c = np.array([alive(bd[1], d) for d in DGRID])
        life1 = (np.minimum(bd[1][:, 1], DMAX) - bd[1][:, 0]) if len(bd[1]) else np.array([0.0])
        i10 = int(np.argmin(abs(DGRID - 0.10)))
        rows.append(dict(subject=s, variant=variant, n_nodes=Zc.shape[0],
                         b0_at_010=int(b0c[i10]), b1_at_010=int(b1c[i10]),
                         b1_max=int(b1c.max()), d_at_b1max=float(DGRID[int(np.argmax(b1c))]),
                         n_bars_h1=len(bd[1]), total_life_h1=float(life1.sum()), max_life_h1=float(life1.max()),
                         auc_b0=float(np.trapezoid(b0c, DGRID)), auc_b1=float(np.trapezoid(b1c, DGRID)),
                         percolation_density=float(DGRID[np.argmax(b0c <= 1)]) if (b0c <= 1).any() else np.nan,
                         **{f"b0_d{d:.2f}": int(x) for d, x in zip(DGRID, b0c)},
                         **{f"b1_d{d:.2f}": int(x) for d, x in zip(DGRID, b1c)}))
ST = pd.DataFrame(rows)
ST.to_csv(C.RESULTS / "static_tda_features.csv", index=False)
print(ST.groupby("variant")[["b0_at_010", "b1_at_010", "n_bars_h1", "total_life_h1"]].agg(["mean", "std"]).round(2))
