"""Shared plotting style for the figures (Elsevier double-column layout)."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import os

MM = 1 / 25.4
W1, W2 = 90 * MM, 190 * MM          # single / double column widths, inches


BLUE, RED, GREEN, GREY, LIGHT = "#08519c", "#e6550d", "#31a354", "#737373", "#9ecae1"
LBLUE, LORANGE = "#6baed6", "#fdae6b"
CMAP = "viridis"


def setup():
    mpl.rcdefaults()
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.direction": "out",
        "ytick.direction": "out", "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.3, "lines.markersize": 4, "legend.frameon": False,
        "legend.handlelength": 1.6, "legend.borderaxespad": 0.2, "mathtext.fontset": "dejavusans",
        "savefig.dpi": 600, "figure.dpi": 150, "axes.unicode_minus": True,
        "pdf.fonttype": 42,
    })


def letter(ax, s, dx=-0.16, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def overlaps(fig):
    """List pairs of overlapping visible text objects (labels, ticks, annotations)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tx = [t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text().strip()]
    bb = [(t, t.get_window_extent(r)) for t in tx]
    out = []
    for i, (a, ba) in enumerate(bb):
        for b, bbb in bb[i + 1:]:
            if ba.overlaps(bbb):
                out.append((a.get_text()[:20], b.get_text()[:20]))
    return out


RES = os.environ.get("ABIDE_RESULTS", "results")
OUT = os.path.join(RES, "figures")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=600)
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
