"""Fig. 6: group contrast, 67 autistic vs 97 control participants."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats as sps
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.py")).read())
setup()
GB = pd.read_csv(os.path.join(RES, "group_balance.csv")); GV = pd.read_csv(os.path.join(RES, "group_feature_validity.csv")); LG = pd.read_csv(os.path.join(RES, "liang_eps_group.csv"))
fig, axes = plt.subplots(1, 3, figsize=(W2, 2.9), gridspec_kw=dict(width_ratios=[0.75, 1.2, 0.75]))

ax = axes[0]
VN = {"AGE_AT_SCAN": "age", "FIQ": "FIQ", "fd_mean": "mean FD", "fd_median": "median FD", "fd_max": "max FD",
      "n_censored": "censored volumes", "dvars_mean": "DVARS", "n_dropped_roi_aal3": "dropped regions"}
g = GB.set_index("variable").loc[list(VN)]
y = np.arange(len(g))[::-1]
ax.axvline(0.5, color=GREY, lw=0.6, ls="--")
ax.plot(g.AUC, y, "o", color=BLUE, ms=5)
for yi, (_, r) in zip(y, g.iterrows()):
    ax.text(0.86, yi, f"{r.p_fdr:.3f}" if r.p_fdr >= 0.001 else "<0.001", fontsize=6, va="center", ha="right", clip_on=False)
ax.text(0.86, len(g) - 0.3, "$q$", fontsize=6.5, ha="right", va="bottom", clip_on=False)
ax.set_yticks(y); ax.set_yticklabels([VN[v] for v in g.index]); ax.set_xlim(0.25, 0.72); ax.set_ylim(-0.7, len(g))
ax.set_xlabel("AUC (autism vs control)")
letter(ax, "a", dx=-0.62)

ax = axes[1]
FA = {"b0_e0.7": r"$\beta_0$, $\varepsilon$ = 0.7", "b1_e0.7": r"$\beta_1$, $\varepsilon$ = 0.7",
      "h1born_e0.7": r"$H_1$ born, $\varepsilon\leq$0.7", "b0_e0.8": r"$\beta_0$, $\varepsilon$ = 0.8",
      "b1_e0.8": r"$\beta_1$, $\varepsilon$ = 0.8", "h1born_e0.8": r"$H_1$ born, $\varepsilon\leq$0.8",
      "b1_e0.9": r"$\beta_1$, $\varepsilon$ = 0.9", "h1born_e0.9": r"$H_1$ born, $\varepsilon\leq$0.9"}
FC = {"C_b0_d010": r"$\beta_0$, density 0.10", "C_b1_d010": r"$\beta_1$, density 0.10",
      "C_b1_total_life": r"$H_1$ total lifetime", "C_b1_z_vs_null": r"$\beta_1$ vs degree null ($z$)"}
FZ = {"Z_density_nbars_h1": r"$H_1$ bars, density", "Z_density_totlen_h1": r"$H_1$ length, density",
      "Z_knn_or_nbars_h1": r"$H_1$ bars, kNN-OR", "Z_knn_or_totlen_h1": r"$H_1$ length, kNN-OR"}
rows = [(FA[f], r.d, r.ci_lo, r.ci_hi, RED) for f in FA for _, r in LG[LG.feature == f].iterrows()]
rows += [(FC[f], r.cohens_d, r.d_lo, r.d_hi, BLUE) for f in FC for _, r in GV[GV.feature == f].iterrows()]
rows += [(FZ[f], r.cohens_d, r.d_lo, r.d_hi, GREEN) for f in FZ for _, r in GV[GV.feature == f].iterrows()]
ypos, yy = [], 0
for i, row in enumerate(rows):
    if i in (len(FA), len(FA) + len(FC)):
        yy += 0.8
    ypos.append(yy); yy += 1
ypos = np.max(ypos) - np.array(ypos)
mde = float(GV.mde80.iloc[0])
for s in (-mde, mde):
    ax.axvline(s, color=GREY, lw=0.5, ls=":")
ax.axvline(0, color=GREY, lw=0.6)
for yi, (lab, d, lo, hi, col) in zip(ypos, rows):
    ax.plot([lo, hi], [yi, yi], color=col, lw=1.4, alpha=0.8); ax.plot(d, yi, "o", color=col, ms=4.5)
ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows], fontsize=6)
for t, r in zip(ax.get_yticklabels(), rows):
    t.set_color(r[4])
ax.set_xticks([-0.5, 0, 0.5]); ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.8, ypos.max() + 0.8)
ax.set_xlabel("Cohen's $d$ (autism $-$ control)")
letter(ax, "b", dx=-0.52)

ax = axes[2]
n1, n2 = 67, 97
dd = np.linspace(0, 0.8, 161)
ncp = dd * np.sqrt(n1 * n2 / (n1 + n2)); df = n1 + n2 - 2; tc = sps.t.ppf(0.975, df)
pw = 1 - sps.nct.cdf(tc, df, ncp) + sps.nct.cdf(-tc, df, ncp)
ax.plot(dd, pw, color=BLUE, lw=1.6)
ax.axhline(0.8, color=GREY, lw=0.5, ls=":"); ax.axvline(mde, color=GREY, lw=0.5, ls=":")
ax.text(mde + 0.02, 0.06, f"$d$ = {mde:.2f}", fontsize=6)
ax.set_xlabel("True effect size $|d|$"); ax.set_ylabel("Power ($\\alpha$ = 0.05)")
ax.set_ylim(0, 1.02); ax.set_xlim(0, 0.8)
letter(ax, "c", dx=-0.3)
fig.tight_layout(w_pad=3.0)
save(fig, "fig6")
print("overlaps:", overlaps(fig))
print("power at 0.3:", round(float(np.interp(0.3, dd, pw)), 3), "| at 0.5:", round(float(np.interp(0.5, dd, pw)), 3))
