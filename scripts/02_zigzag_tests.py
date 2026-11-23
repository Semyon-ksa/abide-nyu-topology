"""Real-versus-surrogate tests for the zigzag barcode summaries (Section 4.4, Fig. 2).

Per participant, rule and statistic: z score of the real value against its surrogates, and
whether it lies outside the surrogate range (probability 2/(m+1) under the null, m surrogates).
Cohort level: binomial test of the exceedance fraction against 2/(m+1), t test of mean z.

Input:  results/m2_full_cohort_raw.csv
Output: results/m2_full_cohort_tests.csv, results/m2_full_cohort_summary.csv
"""
import numpy as np
import pandas as pd
from scipy import stats as sps

import common as C
import zztools as Z

STATS = ["n_bars_h0", "n_bars_h1", "total_length_h0", "total_length_h1", "invisible_bars", "censored_ends"]
# Statistics that are constant by construction under kNN-OR (no region can be isolated)
UNINFORMATIVE = {"knn_or": {"invisible_bars", "n_bars_h0", "total_length_h0"}}

MF = pd.read_csv(C.RESULTS / "m2_full_cohort_raw.csv")
rows = []
for (rule, s), g in MF.groupby(["edge_rule", "subject"]):
    real, surr = g[g.is_real], g[~g.is_real]
    if len(real) != 1 or len(surr) < 5:
        continue
    for st in STATS:
        rows.append(dict(edge_rule=rule, subject=s, stat=st,
                         **Z.zz_surrogate_test(float(real[st].iloc[0]), surr[st].to_numpy(float))))
RF = pd.DataFrame(rows)
RF.to_csv(C.RESULTS / "m2_full_cohort_tests.csv", index=False)

m = int(MF[~MF.is_real].groupby(["subject", "edge_rule"]).size().median())
P_CH = 2 / (m + 1)
summ = []
for (rule, st), g in RF.groupby(["edge_rule", "stat"]):
    k, n = int(g.outside_range.sum()), len(g)
    z = g.z.replace([np.inf, -np.inf], np.nan).dropna()
    summ.append(dict(edge_rule=rule, stat=st, n=n, n_outside=k, frac=k / n,
                     p_binom=float(sps.binomtest(k, n, P_CH).pvalue), mean_z=float(z.mean()),
                     p_ttest_z=float(sps.ttest_1samp(z, 0.0).pvalue) if len(z) > 2 and z.std() > 0 else np.nan,
                     informative=st not in UNINFORMATIVE.get(rule, set())))
TF = pd.DataFrame(summ)
TF.to_csv(C.RESULTS / "m2_full_cohort_summary.csv", index=False)

inf = TF[TF.informative]
k, n = int(inf.n_outside.sum()), int(inf.n.sum())
print(f"surrogates per series: {m}; chance exceedance 2/{m+1} = {P_CH:.4f}")
print(TF.round(4).to_string(index=False))
print(f"\nall cells: {int(TF.n_outside.sum())}/{int(TF.n.sum())} = {TF.n_outside.sum()/TF.n.sum():.4f}")
print(f"informative cells: {k}/{n} = {k/n:.4f}, binomial p vs chance = {sps.binomtest(k, n, P_CH).pvalue:.3f}")
print("cells with p_binom < 0.05:", int((TF.p_binom < 0.05).sum()), "of", len(TF))
