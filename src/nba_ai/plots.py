"""One-figure report of the untouched test season."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from .evaluate import TEST_SEASON, metrics

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
# categorical slots in fixed order (validated reference palette): blue, orange, aqua, yellow
COLOR = {"model, day before": "#2a78d6", "model, at tip-off": "#eb6834", "Elo": "#1baf7a", "market": "#eda100",
         "model, pre-game": "#6250d6"}


def _style(ax: Axes, title: str, subtitle: str) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def _dots(ax: Axes, y: np.ndarray, probs: dict[str, np.ndarray], title: str, subtitle: str) -> None:
    names = list(probs)
    ms = [metrics(y, probs[n]) for n in names]
    lls = [m["logloss"] for m in ms]
    pos = np.arange(len(names))[::-1]
    left = min(lls) - 0.012
    for p, n, m in zip(pos, names, ms, strict=True):
        ax.plot([left, m["logloss"]], [p, p], color=GRID, lw=2, zorder=1)
        ax.scatter([m["logloss"]], [p], s=90, color=COLOR[n], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(m["logloss"] + 0.0015, p, f"{m['logloss']:.4f}  ·  {m['acc']:.1%} right",
                va="center", fontsize=9, color=INK2)
    ax.set_yticks(pos, names)
    ax.set_xlim(left, max(lls) + 0.03)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("log loss (lower is better)")
    ax.set_ylim(-0.6, len(names) - 0.4)
    _style(ax, title, subtitle)


def report(pred: pd.DataFrame, out: str) -> None:
    y = pred.home_win.to_numpy()
    allp = {"Elo": pred.elo_prob.to_numpy(), "model, day before": pred.p_daybefore.to_numpy(),
            "model, pre-game": pred.p_pregame.to_numpy(), "model, at tip-off": pred.p_tipoff.to_numpy()}
    has = pred.p_market.notna().to_numpy()
    mk = {k: v[has] for k, v in allp.items()} | {"market": pred.p_market.to_numpy()[has]}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6), dpi=150, facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.15, 1.15, 1]})
    _dots(axes[0], y, allp, "Every game of the test season", f"{len(y):,} games, never used for training or tuning")
    _dots(axes[1], y[has], mk, "Head-to-head with the betting market", f"{has.sum():,} games with a DraftKings line on ESPN")

    ax = axes[2]
    edges = np.linspace(0, 1, 11)
    for name in ("model, day before", "model, pre-game", "model, at tip-off"):
        p = allp[name]
        b = np.clip(np.digitize(p, edges) - 1, 0, 9)
        xs = [p[b == i].mean() for i in range(10) if (b == i).sum() >= 15]
        ys = [y[b == i].mean() for i in range(10) if (b == i).sum() >= 15]
        ax.plot(xs, ys, color=COLOR[name], lw=2, marker="o", ms=7, mec=SURFACE, mew=2)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.4, ls="--")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_xlabel("predicted home win probability")
    ax.set_ylabel("actual home win rate")
    _style(ax, "Are the probabilities honest?", "Binned predictions vs outcomes; dashed = perfect calibration")
    ax.legend(handles=[Line2D([], [], color=COLOR[n], lw=2, marker="o", label=n)
                       for n in ("model, day before", "model, pre-game", "model, at tip-off")],
              frameon=False, fontsize=9, labelcolor=INK2, loc="upper left")

    fig.suptitle(f"Who wins? NBA game predictions, {TEST_SEASON} test season", x=0.01, y=0.99,
                 ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Day before = previous evening's official injury report. Pre-game = last report before tip. "
             "At tip-off = which rotation players are active. Data: stats.nba.com, NBA injury reports, ESPN odds.",
             fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
