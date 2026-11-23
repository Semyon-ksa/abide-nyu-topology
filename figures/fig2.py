"""Fig. 2: zigzag barcodes vs surrogates, 164 participants, W=80, S=20, density 0.15."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy import stats as sps
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.py")).read())
setup()
M = pd.read_csv(os.path.join(RES, "m2_full_cohort_raw.csv"))
STATS = ["n_bars_h0", "total_length_h0", "n_bars_h1", "total_length_h1", "invisible_bars", "censored_ends"]
NAME = {"n_bars_h0": "$H_0$ bars", "total_length_h0": "$H_0$ length", "n_bars_h1": "$H_1$ bars",
        "total_length_h1": "$H_1$ length", "invisible_bars": "invisible bars", "censored_ends": "cut-off bars"}
DEGEN = {("knn_or", "n_bars_h0"), ("knn_or", "total_length_h0"), ("knn_or", "invisible_bars")}
rows = []
for (s, rule), g in M.groupby(["subject", "edge_rule"]):
    real, sur = g[g.is_real].iloc[0], g[~g.is_real]
    for st in STATS:
        v = sur[st].to_numpy(float); r = float(real[st]); sd = v.std(ddof=1)
        rows.append(dict(subject=s, rule=rule, stat=st, z=(r - v.mean()) / sd if sd > 0 else np.nan,
                         outside=bool(r < v.min() or r > v.max())))
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "fig2_data.csv"), index=False)

fig, axes = plt.subplots(1, 2, figsize=(W2 * 0.72, 2.2), gridspec_kw=dict(width_ratios=[1, 1.35]))
ax = axes[0]
bins = np.linspace(-4, 4, 33)
for rule, col, lab in [("density", BLUE, "density"), ("knn_or", RED, "kNN-OR")]:
    z = R[(R.rule == rule) & ~R.apply(lambda r: (r.rule, r.stat) in DEGEN, axis=1)].z.dropna()
    ax.hist(z, bins=bins, histtype="stepfilled", color=col, alpha=0.22, lw=0, density=True)
    ax.hist(z, bins=bins, histtype="step", color=col, lw=1.2, density=True, label=f"{lab} ($n$ = {len(z)})")
xx = np.linspace(-4, 4, 200)
ax.plot(xx, sps.norm.pdf(xx), color="#252525", lw=0.9, ls="--", label="standard normal")
ax.set_xlabel("$z$ of recording relative to its surrogates"); ax.set_ylabel("Density")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02)); ax.set_ylim(0, 0.62)
letter(ax, "a", dx=-0.22)

ax = axes[1]
cells = [(r, s) for r in ("density", "knn_or") for s in STATS if (r, s) not in DEGEN]
y = np.arange(len(cells))[::-1]
for yi, (rule, st) in zip(y, cells):
    g = R[(R.rule == rule) & (R.stat == st)]
    k, n = int(g.outside.sum()), len(g)
    lo, hi = sps.binomtest(k, n).proportion_ci(method="wilson")
    col = BLUE if rule == "density" else RED
    ax.plot([lo, hi], [yi, yi], color=col, lw=1.4, alpha=0.8)
    ax.plot(k / n, yi, "o", color=col, ms=5)
ax.axvline(2 / 21, color="k", lw=0.7, ls="--")
ax.text(2 / 21 + 0.004, len(cells) - 0.35, "chance, 2/21", fontsize=6, va="bottom")
ax.set_yticks(y); ax.set_yticklabels([NAME[s] for _, s in cells])
for t, (rule, _) in zip(ax.get_yticklabels(), cells):
    t.set_color(BLUE if rule == "density" else RED)
ax.set_xlim(0, 0.25); ax.set_ylim(-0.7, len(cells) + 0.3)
ax.set_xlabel("Fraction outside the surrogate range")
letter(ax, "b", dx=-0.42)
fig.tight_layout(w_pad=2.0)
save(fig, "fig2")
print("overlaps:", overlaps(fig))
