"""Fig. 5: degree-preserving null; 164 participants, no GSR, 20 random graphs each."""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import os
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.py")).read())
setup()
SS = pd.read_csv(os.path.join(RES, "static_surrogate_test.csv"))
NR = pd.read_csv(os.path.join(RES, "null_density_robustness_summary.csv"))
fig, axes = plt.subplots(1, 4, figsize=(W2, 2.05), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.9]))
for ax, st, lab, let in [(axes[0], "beta0", r"$\beta_0$", "a"), (axes[1], "beta1", r"$\beta_1$", "b")]:
    g = SS[(SS.null == "degree") & (SS.stat == st)]
    hi = max(g.real.max(), g.surr_mean.max()) * 1.05
    if st == "beta0":
        ax.plot([0, hi], [0, hi], color=GREY, lw=0.6, ls="--")
    ax.plot(g.real.clip(lower=1), g.surr_mean, "o", ms=3.5, color=BLUE if st == "beta0" else RED, mew=0, alpha=0.65)
    if st == "beta1":
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.8, 400); ax.set_ylim(0.8, 400)
        ax.plot([0.8, 400], [0.8, 400], color=GREY, lw=0.6, ls="--")
    else:
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(f"Observed {lab}"); ax.set_ylabel(f"{lab} of degree-preserving null")
    letter(ax, let, dx=-0.42, dy=1.05)
ax = axes[2]
for st, col, mk, lab in [("beta0", BLUE, "o", r"$\beta_0$"), ("beta1", RED, "s", r"$\beta_1$")]:
    g = NR[NR.stat == st].sort_values("density")
    ax.plot(g.density, g.ratio, mk + "-", color=col, ms=5, lw=1.3, label=lab)
ax.axhline(1, color=GREY, lw=0.6, ls="--")
ax.set_xticks([0.05, 0.10, 0.15, 0.20]); ax.set_ylim(0, 1.2)
ax.set_xlabel("Edge density"); ax.set_ylabel("Observed / null mean")
ax.legend(loc="lower right")
letter(ax, "c", dx=-0.3)
ax = axes[3]
nulls = [("degree", "degree-\npreserving"), ("erdos_renyi", "Erd\u0151s\u2013\nR\u00e9nyi")]
x = np.arange(len(nulls))
for k, (st, col, mk, lab) in enumerate([("beta0", BLUE, "o", r"$\beta_0$"), ("beta1", RED, "s", r"$\beta_1$")]):
    v = [SS[(SS.null == n) & (SS.stat == st)].outside_range.mean() for n, _ in nulls]
    ax.plot(x + (k - 0.5) * 0.18, v, mk, color=col, ms=6, ls="none", label=lab)
ax.axhline(2 / 21, color=GREY, lw=0.6, ls="--")
ax.set_xticks(x); ax.set_xticklabels([l for _, l in nulls]); ax.set_xlim(-0.5, 1.5); ax.set_ylim(0, 1.05)
ax.set_ylabel("Fraction outside null range")
letter(ax, "d", dx=-0.36)
fig.tight_layout(w_pad=1.4)
save(fig, "fig5")
print("overlaps:", overlaps(fig))
print(SS[SS.null.isin(["degree", "erdos_renyi"])].groupby(["null", "stat"]).agg(out=("outside_range", "mean"), mz=("z", "median")).round(3))
