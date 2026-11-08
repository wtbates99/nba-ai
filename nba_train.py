"""Predict a player's points in their next game from their last 10 games.

Two models, trained side by side, so you can see how a training loop works:
  * MLP          - the simplest neural net (flatten the last 10 games, a few layers)
  * Transformer  - one attention block: the upcoming game "looks back" at the last 10

    python nba_train.py                                  # synthetic data, works offline
    python nba_train.py --source nba --season 2024-25    # real games via nba_api (needs internet)

Needs: pip install torch pandas matplotlib nba_api
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import matplotlib

matplotlib.use("Agg")  # draw straight to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from torch import Tensor, nn

F32 = NDArray[np.float32]
Tensors = tuple[Tensor, Tensor, Tensor]  # (history, context, target)

STAT_COLS: list[str] = ["MIN", "FGA", "FG3A", "FTA", "REB", "AST", "TOV", "PTS"]
PTS_IDX: int = STAT_COLS.index("PTS")
CTX_COLS: list[str] = ["IS_HOME", "REST_DAYS"]
SEQ_LEN: int = 10  # how many previous games the model gets to see


# ============================================================ 1. DATA
def load_nba(season: str) -> pd.DataFrame:
    """One request returns every player-game of a season (needs internet)."""
    from nba_api.stats.endpoints import leaguegamelog

    raw: pd.DataFrame = leaguegamelog.LeagueGameLog(
        season=season, player_or_team_abbreviation="P"
    ).get_data_frames()[0]
    need = ["PLAYER_ID", "GAME_DATE", "MATCHUP", *STAT_COLS]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise ValueError(f"nba_api result lacks columns {missing}; it has {list(raw.columns)}")
    return raw[need].dropna().copy()


def make_synthetic(n_players: int = 300, n_games: int = 70, seed: int = 0) -> pd.DataFrame:
    """Fake game logs with the same columns as the real thing.

    Built-in structure: each player has a minutes level and a points-per-minute skill,
    a hot/cold streak that drifts slowly, and a small home-court bonus.
    """
    rng = np.random.default_rng(seed)
    season_start = pd.Timestamp("2024-10-22")
    rows: list[dict[str, object]] = []
    for pid in range(n_players):
        base_min = rng.uniform(12.0, 36.0)
        ppm = rng.uniform(0.30, 0.65)
        form = 0.0
        for day in np.sort(rng.choice(165, size=n_games, replace=False)):
            form = 0.85 * form + rng.normal(0.0, 0.12)
            home = bool(rng.random() < 0.5)
            minutes = float(np.clip(base_min + rng.normal(0.0, 4.0), 4.0, 42.0))
            pts = max(0.0, minutes * ppm * (1.0 + form) + (1.0 if home else 0.0) + rng.normal(0.0, 3.0))
            rows.append(
                {
                    "PLAYER_ID": pid,
                    "GAME_DATE": season_start + pd.Timedelta(days=int(day)),
                    "MATCHUP": "AAA vs. BBB" if home else "AAA @ BBB",
                    "MIN": minutes,
                    "PTS": round(pts),
                    "FGA": max(0.0, round(pts * 0.8 + rng.normal(0, 1.5))),
                    "FG3A": max(0.0, round(pts * 0.3 + rng.normal(0, 1.0))),
                    "FTA": max(0.0, round(pts * 0.25 + rng.normal(0, 1.0))),
                    "REB": max(0.0, round(minutes * 0.2 + rng.normal(0, 2.0))),
                    "AST": max(0.0, round(minutes * 0.12 + rng.normal(0, 1.5))),
                    "TOV": max(0.0, round(minutes * 0.07 + rng.normal(0, 1.0))),
                }
            )
    return pd.DataFrame(rows)


@dataclass
class Samples:
    hist: F32  # (N, SEQ_LEN, n_stats)  the player's previous games
    ctx: F32  # (N, 2)                  known before tip-off: home?, rest days
    y: F32  # (N,)                      points actually scored in the game
    date: NDArray[np.datetime64]  # (N,)


def build_samples(games: pd.DataFrame, seq_len: int = SEQ_LEN) -> Samples:
    df = games.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["IS_HOME"] = (~df["MATCHUP"].str.contains("@")).astype(float)
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    df["REST_DAYS"] = df.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days.fillna(7).clip(upper=7)

    hist: list[F32] = []
    ctx: list[F32] = []
    y: list[float] = []
    date: list[np.datetime64] = []
    for _, g in df.groupby("PLAYER_ID"):
        stats = g[STAT_COLS].to_numpy(dtype=np.float32)
        context = g[CTX_COLS].to_numpy(dtype=np.float32)
        when = g["GAME_DATE"].to_numpy()
        for t in range(seq_len, len(g)):
            hist.append(stats[t - seq_len : t])  # only games BEFORE game t -> no peeking
            ctx.append(context[t])
            y.append(float(stats[t, PTS_IDX]))
            date.append(when[t])
    if not hist:
        raise ValueError("No player has enough games to build a history window.")
    order = np.argsort(np.array(date))  # oldest -> newest
    return Samples(
        np.stack(hist)[order],
        np.stack(ctx)[order],
        np.array(y, dtype=np.float32)[order],
        np.array(date)[order],
    )


def take(s: Samples, sl: slice) -> Samples:
    return Samples(s.hist[sl], s.ctx[sl], s.y[sl], s.date[sl])


def split_by_time(s: Samples, test_frac: float = 0.2) -> tuple[Samples, Samples]:
    """Train on the past, test on the future (never shuffle time series)."""
    cut = int(len(s.y) * (1.0 - test_frac))
    return take(s, slice(0, cut)), take(s, slice(cut, None))


@dataclass
class Scaler:
    h_mean: F32
    h_std: F32
    c_mean: F32
    c_std: F32
    y_mean: float
    y_std: float


def fit_scaler(train: Samples) -> Scaler:
    """Networks train much better on inputs with mean 0, std 1. Fit on TRAIN only."""
    flat = train.hist.reshape(-1, train.hist.shape[-1])
    return Scaler(
        flat.mean(axis=0),
        flat.std(axis=0) + 1e-6,
        train.ctx.mean(axis=0),
        train.ctx.std(axis=0) + 1e-6,
        float(train.y.mean()),
        float(train.y.std()),
    )


def to_tensors(s: Samples, sc: Scaler) -> Tensors:
    hist = torch.from_numpy(((s.hist - sc.h_mean) / sc.h_std).astype(np.float32))
    ctx = torch.from_numpy(((s.ctx - sc.c_mean) / sc.c_std).astype(np.float32))
    y = torch.from_numpy(((s.y - sc.y_mean) / sc.y_std).astype(np.float32))
    return hist, ctx, y


# ============================================================ 2. MODELS
class MLP(nn.Module):
    """Flatten the last 10 games into one long vector and push it through two hidden layers."""

    def __init__(
        self, seq_len: int, n_feat: int, n_ctx: int, hidden: int = 64, dropout: float = 0.3
    ) -> None:
        super().__init__()
        # Dropout randomly zeroes some units while training, so the net can't just
        # memorize the training games (this is what fights overfitting).
        self.net = nn.Sequential(
            nn.Linear(seq_len * n_feat + n_ctx, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, hist: Tensor, ctx: Tensor) -> Tensor:
        x = torch.cat([hist.flatten(start_dim=1), ctx], dim=1)  # (B, seq*feat + ctx)
        out: Tensor = self.net(x)
        return out.reshape(-1)  # (B,)


class SeqTransformer(nn.Module):
    """One attention block (the same Q/K/V math from the NumPy version).

    The upcoming game (its home/rest context) is the QUERY. The last 10 games are the
    KEYS and VALUES. Attention decides how much each past game matters for this prediction.
    """

    def __init__(
        self, n_feat: int, n_ctx: int, seq_len: int, d_model: int = 32, n_heads: int = 4
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(n_feat, d_model)  # each past game -> a vector
        self.pos = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)  # learned "how long ago"
        self.query = nn.Linear(n_ctx, d_model)  # the upcoming game -> a vector
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def _run(self, hist: Tensor, ctx: Tensor) -> tuple[Tensor, Tensor]:
        kv = self.norm_kv(self.embed(hist) + self.pos)  # (B, S, D)
        q = self.query(ctx).unsqueeze(1)  # (B, 1, D)
        attended, weights = self.attn(q, kv, kv, need_weights=True, average_attn_weights=True)
        x = q + attended  # residual connection
        x = x + self.ff(self.norm_ff(x))  # residual connection
        pred: Tensor = self.head(self.norm_out(x)).reshape(-1)  # (B,)
        return pred, weights.squeeze(1)  # weights: (B, S)

    def forward(self, hist: Tensor, ctx: Tensor) -> Tensor:
        return self._run(hist, ctx)[0]

    def attention(self, hist: Tensor, ctx: Tensor) -> F32:
        """How much each of the last 10 games was looked at, per sample: (N, SEQ_LEN)."""
        self.eval()
        with torch.no_grad():
            _, weights = self._run(hist, ctx)
        return weights.numpy().astype(np.float32)


# ============================================================ 3. TRAINING
@dataclass
class Curve:
    train_mae: list[float] = field(default_factory=list)
    test_mae: list[float] = field(default_factory=list)


def mae_points(model: nn.Module, data: Tensors, y_std: float) -> float:
    """Mean absolute error in real points (undo the standardizing)."""
    hist, ctx, y = data
    model.eval()
    with torch.no_grad():
        pred: Tensor = model(hist, ctx)
    return float((pred - y).abs().mean().item() * y_std)


def fit(
    model: nn.Module,
    train: Tensors,
    test: Tensors,
    y_std: float,
    epochs: int,
    lr: float = 2e-3,
    weight_decay: float = 1e-2,
    batch_size: int = 256,
    seed: int = 0,
) -> Curve:
    """The whole of 'training a model' is this loop."""
    gen = torch.Generator().manual_seed(seed)
    # weight_decay gently pulls weights toward zero: another guard against overfitting
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    hist, ctx, y = train
    n = len(y)
    curve = Curve()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)  # new shuffled order each epoch
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            pred: Tensor = model(hist[idx], ctx[idx])  # 1. forward: make predictions
            loss = nn.functional.mse_loss(pred, y[idx])  # 2. how wrong were they?
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]  # 3. backward: which weights caused the error?
            opt.step()  # 4. nudge every weight a little to reduce it
        curve.train_mae.append(mae_points(model, train, y_std))
        curve.test_mae.append(mae_points(model, test, y_std))
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f"  epoch {epoch + 1:3d}  train MAE {curve.train_mae[-1]:.3f}"
                f"  test MAE {curve.test_mae[-1]:.3f}"
            )
    return curve


# ============================================================ 4. PLOTS
INK, INK2 = "#0b0b0b", "#52514e"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def style_axes(ax: Axes, title: str, subtitle: str, grid: bool = True) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=24)
    ax.text(0, 1.03, subtitle, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def make_figure(
    curves: dict[str, Curve],
    baseline: float,
    actual: F32,
    predicted: F32,
    attention: F32,
    data_label: str,
    out: str,
) -> None:
    seq_len = attention.shape[1]
    games_ago = [str(seq_len - j) for j in range(seq_len)]  # oldest on the left
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.2), dpi=150, facecolor=SURFACE)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # (a) learning curves
    epochs = np.arange(1, len(curves["MLP"].test_mae) + 1)
    for name, color in (("MLP", BLUE), ("Transformer", ORANGE)):
        ax_a.plot(epochs, curves[name].test_mae, color=color, lw=2)
        ax_a.plot(epochs, curves[name].train_mae, color=color, lw=1.4, ls="--")
    ax_a.axhline(baseline, color=GRAY, lw=1.6, ls=":")
    style_axes(ax_a, "Error falls as the model trains", "Mean absolute error in points, per epoch (lower is better)")
    ax_a.set_xlabel("epoch (one full pass over the training games)")
    ax_a.set_ylabel("points off, on average")
    y_lo, y_hi = ax_a.get_ylim()
    ax_a.set_ylim(y_lo, baseline + 0.30 * (baseline - y_lo))  # headroom so the legend can't cover the lines
    ax_a.legend(
        handles=[
            Line2D([], [], color=BLUE, lw=2, label="MLP"),
            Line2D([], [], color=ORANGE, lw=2, label="Transformer"),
            Line2D([], [], color=GRAY, lw=1.6, ls=":", label="baseline: avg of last 10 games"),
            Line2D([], [], color=INK2, lw=1.4, ls="--", label="dashed = train, solid = test"),
        ],
        frameon=False,
        fontsize=9,
        labelcolor=INK2,
        loc="upper left",
        ncol=2,
    )

    # (b) predicted vs actual
    rng = np.random.default_rng(0)
    pick = rng.choice(len(actual), size=min(2500, len(actual)), replace=False)
    lo = min(0.0, float(predicted.min()))
    hi = max(float(actual.max()), float(predicted.max())) * 1.05
    ax_b.scatter(actual[pick], predicted[pick], s=12, alpha=0.25, color=ORANGE, linewidths=0)
    ax_b.plot([lo, hi], [lo, hi], color=GRAY, lw=1.6, ls="--")
    ax_b.set_xlim(lo, hi)
    ax_b.set_ylim(lo, hi)
    style_axes(ax_b, "Transformer: predicted vs actual points", "Unseen future games; dots on the dashed line = perfect")
    ax_b.set_xlabel("actual points")
    ax_b.set_ylabel("predicted points")

    # (c) mean attention by position
    mean_att = attention.mean(axis=0)
    ax_c.bar(range(seq_len), mean_att, color=ORANGE, width=0.62)
    ax_c.axhline(1.0 / seq_len, color=GRAY, lw=1.6, ls="--")
    ax_c.text(-0.45, 1.0 / seq_len, "even split (0.10)", ha="left", va="bottom", fontsize=9, color=INK2)
    ax_c.set_xticks(range(seq_len), games_ago)
    style_axes(ax_c, "Which past games does it look at?", "Average attention weight across all test games")
    ax_c.set_xlabel("games ago (1 = the most recent game)")
    ax_c.set_ylabel("attention weight")

    # (d) heatmap of individual games
    rows = np.sort(rng.choice(len(attention), size=min(24, len(attention)), replace=False))
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    im = ax_d.imshow(attention[rows], aspect="auto", cmap=cmap, vmin=0.0)
    ax_d.set_xticks(range(seq_len), games_ago)
    ax_d.set_yticks([])
    style_axes(ax_d, "Attention for 24 individual predictions", "Each row is one game being predicted; darker = looked at more", grid=False)
    for side in ("left", "bottom"):
        ax_d.spines[side].set_visible(False)
    ax_d.set_xlabel("games ago (1 = the most recent game)")
    cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(colors=INK2, labelsize=9)
    cbar.outline.set_visible(False)

    fig.suptitle(
        "Predicting a player's next-game points from his last 10 games",
        x=0.012, y=0.99, ha="left", fontsize=15, fontweight="bold", color=INK,
    )
    fig.text(0.012, 0.012, data_label, fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


# ============================================================ 5. RUN IT
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "nba"], default="synthetic")
    ap.add_argument("--season", default="2024-25", help="NBA season, e.g. 2024-25")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="nba_training.png")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.source == "nba":
        games = load_nba(args.season)
        label = f"Data: real NBA player game logs, {args.season} season (stats.nba.com via nba_api)"
    else:
        games = make_synthetic(seed=args.seed)
        label = "Data: SYNTHETIC game logs (made up, for demonstration), not real NBA games"

    samples = build_samples(games)
    train, test = split_by_time(samples)
    print(f"{len(train.y):,} training games, {len(test.y):,} test games (the most recent 20%)")

    # Baseline to beat: predict the average of the player's last 10 games.
    baseline = float(np.abs(test.hist[:, :, PTS_IDX].mean(axis=1) - test.y).mean())

    scaler = fit_scaler(train)
    train_t, test_t = to_tensors(train, scaler), to_tensors(test, scaler)
    n_feat, n_ctx = len(STAT_COLS), len(CTX_COLS)

    mlp = MLP(SEQ_LEN, n_feat, n_ctx)
    tfm = SeqTransformer(n_feat, n_ctx, SEQ_LEN)
    curves: dict[str, Curve] = {}
    for name, model in (("MLP", mlp), ("Transformer", tfm)):
        n_params = sum(prm.numel() for prm in model.parameters())
        print(f"\n== {name} ({n_params:,} parameters)")
        curves[name] = fit(model, train_t, test_t, scaler.y_std, args.epochs, seed=args.seed)

    print("\nTest error in points (lower is better):")
    print(f"  baseline (avg of last 10 games): {baseline:.3f}")
    for name, curve in curves.items():
        print(f"  {name:<32s} {curve.test_mae[-1]:.3f}")

    tfm.eval()
    with torch.no_grad():
        pred_std: Tensor = tfm(test_t[0], test_t[1])
    predicted = (pred_std.numpy() * scaler.y_std + scaler.y_mean).astype(np.float32)
    attention = tfm.attention(test_t[0], test_t[1])
    make_figure(curves, baseline, test.y, predicted, attention, label, args.out)
    print(f"\nSaved plot -> {args.out}")


if __name__ == "__main__":
    main()
