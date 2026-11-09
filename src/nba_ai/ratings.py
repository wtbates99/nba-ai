"""Team strength ratings computed walk-forward: every value uses only games before that date."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .types import F64


@dataclass(frozen=True)
class EloParams:
    k: float = 15.0
    home: float = 70.0  # Elo points of home-court advantage (tuned on 2008-09..2020-21)
    carry: float = 0.65  # share of rating kept between seasons (rest reverts to the mean)
    mov_exp: float = 0.9


def elo(games: pd.DataFrame, p: EloParams = EloParams()) -> pd.DataFrame:
    """MOV-adjusted Elo. `games` needs GAME_ID, SEASON, home_id, away_id, margin (home pts - away pts),
    sorted by date. Returns pre-game home/away Elo and the Elo win probability."""
    rating: dict[int, float] = {}
    season_of: dict[int, str] = {}
    h_pre = np.empty(len(games))
    a_pre = np.empty(len(games))
    for i, (season, h, a, margin) in enumerate(
        games[["SEASON", "home_id", "away_id", "margin"]].itertuples(index=False)
    ):
        for t in (h, a):
            if t not in rating:
                rating[t] = 1500.0
            elif season_of[t] != season:  # first game of a new season: regress to the mean
                rating[t] = p.carry * rating[t] + (1 - p.carry) * 1505.0
            season_of[t] = season
        rh, ra = rating[h], rating[a]
        h_pre[i], a_pre[i] = rh, ra
        diff = rh + p.home - ra
        exp_h = 1.0 / (1.0 + 10 ** (-diff / 400.0))
        if np.isnan(margin):  # unplayed game (prediction time): no update
            continue
        win = 1.0 if margin > 0 else 0.0
        fav_diff = diff if margin > 0 else -diff  # winner's rating edge
        mult = (abs(margin) + 3.0) ** p.mov_exp / (7.5 + 0.006 * fav_diff)
        delta = p.k * mult * (win - exp_h)
        rating[h] = rh + delta
        rating[a] = ra - delta
    out = pd.DataFrame({"elo_home": h_pre, "elo_away": a_pre}, index=games.index)
    out["elo_prob"] = 1.0 / (1.0 + 10 ** (-(out.elo_home + p.home - out.elo_away) / 400.0))
    return out


def srs(games: pd.DataFrame, halflife_days: float = 45.0, ridge: float = 4.0) -> pd.DataFrame:
    """Opponent-adjusted offense/defense ratings (points per game vs average), refit each game day
    on time-weighted prior games:  pts_for = mu + hca*home + off[team] - def[opp].

    Ratings carry across seasons through the time decay (summer break = big decay)."""
    team_ids = sorted(set(games.home_id) | set(games.away_id))
    idx = {t: i for i, t in enumerate(team_ids)}
    n = len(team_ids)
    dates = games.GAME_DATE.to_numpy()
    hi = games.home_id.map(idx).to_numpy()
    ai = games.away_id.map(idx).to_numpy()
    hp = games.home_pts.to_numpy(dtype=float)
    ap = games.away_pts.to_numpy(dtype=float)

    # design: params = [mu, hca, off_0..off_n-1, def_0..def_n-1]; two rows per game
    n_par = 2 + 2 * n
    out = np.full((len(games), 4), np.nan)  # off_h, def_h, off_a, def_a
    played = ~np.isnan(hp)
    uniq = np.unique(dates)
    for d in uniq:
        prior = played & (dates < d)
        today = np.where(dates == d)[0]
        if prior.sum() < 150:
            continue
        age = (d - dates[prior]).astype("timedelta64[D]").astype(float)
        w = 0.5 ** (age / halflife_days)
        keep = w > 1e-3
        w = w[keep]
        ph, pa = hi[prior][keep], ai[prior][keep]
        m = len(w)
        X = np.zeros((2 * m, n_par))
        r = np.arange(m)
        X[:, 0] = 1.0
        X[r, 1] = 1.0  # home row gets HCA
        X[r, 2 + ph] = 1.0
        X[r, 2 + n + pa] = -1.0
        X[m + r, 2 + pa] = 1.0
        X[m + r, 2 + n + ph] = -1.0
        y = np.concatenate([hp[prior][keep], ap[prior][keep]])
        ww = np.concatenate([w, w])
        reg = np.full(n_par, ridge)
        reg[:2] = 1e-6
        A = X.T @ (X * ww[:, None]) + np.diag(reg) * ww.sum() / len(ww)
        b = X.T @ (ww * y)
        beta: F64 = np.linalg.solve(A, b)
        off, de = beta[2 : 2 + n], beta[2 + n :]
        out[today, 0] = off[hi[today]]
        out[today, 1] = de[hi[today]]
        out[today, 2] = off[ai[today]]
        out[today, 3] = de[ai[today]]
    res = pd.DataFrame(out, columns=["srs_off_h", "srs_def_h", "srs_off_a", "srs_def_a"], index=games.index)
    res["srs_net_diff"] = (res.srs_off_h + res.srs_def_h) - (res.srs_off_a + res.srs_def_a)
    return res


ApmGame = tuple[float, float, list[tuple[int, float]], list[tuple[int, float]]]  # day, margin, home, away


def fit_apm(past: list[ApmGame], today: float, halflife_days: float, window_days: float,
            lam: float, prior: dict[int, float]) -> dict[int, float]:
    """Game-level adjusted plus-minus. margin = hca + sum_home r*min/48 - sum_away r*min/48,
    ridge-shrunk toward `prior` (a box-score estimate), time-weighted. Returns rating per 48 min."""
    from scipy import sparse

    rows = [g for g in past if today - g[0] <= window_days]
    if len(rows) < 300:
        return {}
    pids = sorted({pid for g in rows for side in (g[2], g[3]) for pid, _ in side})
    col = {pid: i + 1 for i, pid in enumerate(pids)}  # col 0 = home-court intercept
    r_idx, c_idx, vals = [], [], []
    y = np.empty(len(rows))
    w = np.empty(len(rows))
    for i, (day, margin, home, away) in enumerate(rows):
        r_idx.append(i); c_idx.append(0); vals.append(1.0)
        for pid, mn in home:
            r_idx.append(i); c_idx.append(col[pid]); vals.append(mn / 48.0)
        for pid, mn in away:
            r_idx.append(i); c_idx.append(col[pid]); vals.append(-mn / 48.0)
        y[i] = margin
        w[i] = 0.5 ** ((today - day) / halflife_days)
    X = sparse.csr_matrix((vals, (r_idx, c_idx)), shape=(len(rows), len(pids) + 1))
    p0 = np.array([0.0] + [prior.get(pid, 0.0) for pid in pids])
    resid = y - X @ p0
    Xw = X.multiply(w[:, None]).tocsr()
    A = (X.T @ Xw).toarray()
    reg = np.full(len(pids) + 1, lam)
    reg[0] = 1e-3
    A[np.diag_indices_from(A)] += reg
    b = Xw.T @ resid
    beta = p0 + np.linalg.solve(A, b)
    return {pid: float(beta[col[pid]]) for pid in pids}

