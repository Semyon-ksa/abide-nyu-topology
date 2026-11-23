"""Autism versus typically developing controls (Sections 3.6 and 4.10, Fig. 6).

Features and models were fixed before the diagnostic labels were opened. For each feature:
Cohen's d with 95% CI, Mann-Whitney test, and an OLS model adjusted for age, sex and mean FD;
Benjamini-Hochberg correction within each feature family. FIQ is reported in the balance
table but is not used as a covariate (it differs between groups as a correlate of the diagnosis).

Inputs:  results/static_tda_features.csv, results/static_surrogate_test.csv,
         results/m2_full_cohort_raw.csv, results/liang_eps_replication.csv
Outputs: results/group_balance.csv, results/group_feature_validity.csv, results/liang_eps_group.csv
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats as sps
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestIndPower

import common as C

R = C.RESULTS
qc = C.qc_table()
keep = C.included_subjects()
COV = C.phenotype()[["subject", "DX_GROUP", "AGE_AT_SCAN", "SEX", "FIQ"]].merge(qc, on="subject")
COV = COV[COV.subject.isin(keep)].copy()
COV["group"] = np.where(COV.DX_GROUP == 1, "ASD", "TD")

# --- 1. Balance of demographics and data quality -------------------------------------------
bal = []
for v in ["AGE_AT_SCAN", "FIQ", "fd_mean", "fd_median", "fd_max", "n_censored", "dvars_mean", "n_dropped_roi_aal3"]:
    a, b = COV.loc[COV.group == "ASD", v].dropna(), COV.loc[COV.group == "TD", v].dropna()
    u = sps.mannwhitneyu(a, b)
    bal.append(dict(variable=v, ASD_median=a.median(), TD_median=b.median(), n_ASD=len(a), n_TD=len(b),
                    AUC=u.statistic / (len(a) * len(b)), p=u.pvalue))
BAL = pd.DataFrame(bal)
BAL["p_fdr"] = multipletests(BAL.p, method="fdr_bh")[1]
BAL.to_csv(R / "group_balance.csv", index=False)

# --- 2. Recommended configuration (C) and zigzag features (Z) ---------------------------------
ST = pd.read_csv(R / "static_tda_features.csv")
SS = pd.read_csv(R / "static_surrogate_test.csv")
MF = pd.read_csv(R / "m2_full_cohort_raw.csv")
F = COV[["subject", "group", "AGE_AT_SCAN", "SEX", "FIQ", "fd_mean"]].copy()
F["is_asd"] = (F.group == "ASD").astype(int)
s = ST[ST.variant == "noGSR_bp"].set_index("subject")
F["C_b0_d010"] = F.subject.map(s.b0_at_010)
F["C_b1_d010"] = F.subject.map(s.b1_at_010)
F["C_b1_total_life"] = F.subject.map(s.total_life_h1)
F["C_b1_z_vs_null"] = F.subject.map(SS[(SS.null == "degree") & (SS.stat == "beta1")].set_index("subject").z)
for rule in ["density", "knn_or"]:
    r = MF[MF.is_real & (MF.edge_rule == rule)].set_index("subject")
    F[f"Z_{rule}_nbars_h1"] = F.subject.map(r.n_bars_h1)
    F[f"Z_{rule}_totlen_h1"] = F.subject.map(r.total_length_h1)

n1, n2 = int(F.is_asd.sum()), int((1 - F.is_asd).sum())
z95 = sps.norm.ppf(0.975)
rows = []
for cfg in ("C", "Z"):
    for c in [k for k in F if k.startswith(cfg + "_")]:
        d = F[[c, "is_asd", "AGE_AT_SCAN", "SEX", "fd_mean"]].dropna()
        a, b = d.loc[d.is_asd == 1, c], d.loc[d.is_asd == 0, c]
        u = sps.mannwhitneyu(a, b)
        fit = sm.OLS(d[c].astype(float), sm.add_constant(d[["is_asd", "AGE_AT_SCAN", "SEX", "fd_mean"]].astype(float))).fit()
        sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        dd = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
        se = np.sqrt((n1 + n2) / (n1 * n2) + dd ** 2 / (2 * (n1 + n2 - 2)))
        rows.append(dict(config=cfg, feature=c, n=len(d), ASD_mean=a.mean(), TD_mean=b.mean(), cohens_d=dd,
                         d_lo=dd - z95 * se, d_hi=dd + z95 * se, AUC=u.statistic / (len(a) * len(b)),
                         p_raw=u.pvalue, beta_adj=fit.params["is_asd"], p_adj=fit.pvalues["is_asd"]))
G = pd.DataFrame(rows)
for cfg in ("C", "Z"):
    m = G.config == cfg
    G.loc[m, "p_raw_fdr"] = multipletests(G.loc[m, "p_raw"], method="fdr_bh")[1]
    G.loc[m, "p_adj_fdr"] = multipletests(G.loc[m, "p_adj"], method="fdr_bh")[1]
G["mde80"] = TTestIndPower().solve_power(effect_size=None, nobs1=n1, ratio=n2 / n1, alpha=0.05, power=0.80)
G.to_csv(R / "group_feature_validity.csv", index=False)

# --- 3. Published configuration (Liang et al.), with and without the spread of r ---------------
L = pd.read_csv(R / "liang_eps_replication.csv")
D = L.merge(COV[["subject", "DX_GROUP", "AGE_AT_SCAN", "SEX", "fd_mean"]], on="subject")
D["asd"] = (D.DX_GROUP == 1).astype(int)
D["male"] = (D.SEX == 1).astype(int)
rows = []
for f in [f for f in [f"{k}_e{e}" for e in (0.7, 0.8, 0.9) for k in ("b0", "b1", "h1born")] if D[f].std() > 0]:
    a, c = D.loc[D.asd == 1, f], D.loc[D.asd == 0, f]
    sp = np.sqrt(((len(a) - 1) * a.var() + (len(c) - 1) * c.var()) / (len(a) + len(c) - 2))
    dd = (a.mean() - c.mean()) / sp
    se = np.sqrt((len(a) + len(c)) / (len(a) * len(c)) + dd ** 2 / (2 * (len(a) + len(c))))
    dz = D.assign(y=(D[f] - D[f].mean()) / D[f].std(), sdz=(D.sd_r - D.sd_r.mean()) / D.sd_r.std())
    m1 = smf.ols("y ~ asd + AGE_AT_SCAN + male + fd_mean", dz).fit()
    m2 = smf.ols("y ~ asd + AGE_AT_SCAN + male + fd_mean + sdz", dz).fit()
    rows.append(dict(feature=f, mean_asd=a.mean(), mean_td=c.mean(), d=dd, ci_lo=dd - 1.96 * se, ci_hi=dd + 1.96 * se,
                     p_rank=sps.mannwhitneyu(a, c).pvalue, b_adj=m1.params.asd, p_adj=m1.pvalues.asd,
                     b_adj_sd=m2.params.asd, p_adj_sd=m2.pvalues.asd))
LG = pd.DataFrame(rows)
LG["q_rank"] = multipletests(LG.p_rank, method="fdr_bh")[1]
LG["q_adj"] = multipletests(LG.p_adj, method="fdr_bh")[1]
LG.to_csv(R / "liang_eps_group.csv", index=False)

print(f"ASD {n1}, TD {n2}; minimal detectable |d| at 80% power: {G.mde80.iloc[0]:.2f}")
print(BAL.round(3).to_string(index=False))
print(G[["config", "feature", "cohens_d", "d_lo", "d_hi", "p_raw_fdr", "p_adj_fdr"]].round(3).to_string(index=False))
print(LG[["feature", "d", "ci_lo", "ci_hi", "q_rank", "q_adj"]].round(3).to_string(index=False))
