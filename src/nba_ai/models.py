"""Feature sets and models. Every model maps a feature frame to P(home win)."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FORM_HL, FORM_STATS
from .types import F64

RATING = ["d_elo", "elo_prob", "srs_net_diff", "srs_off_h", "srs_def_h", "srs_off_a", "srs_def_a"]
FORM = [f"d_{c}_ew{hl}" for c in FORM_STATS for hl in FORM_HL]
SEASON = ["d_season_win", "d_season_margin", "home_season_gp", "away_season_gp"]
SCHED = ["home_rest", "away_rest", "home_b2b", "away_b2b", "home_games_last7", "away_games_last7",
         "away_travel_km", "home_travel_km", "home_alt", "month", "lg_home_margin", "lg_home_home_win"]
ROSTER_DAYB = ["home_r_usual", "away_r_usual", "d_r_usual", "home_r_stale_missing", "away_r_stale_missing",
               "d_r_stale_missing", "home_a_usual", "away_a_usual", "d_a_usual", "d_a_stale"]
ROSTER_TIP = ["home_r_tip", "away_r_tip", "d_r_tip", "home_r_tip_missing", "away_r_tip_missing",
              "d_r_tip_missing", "home_r_top3_out", "away_r_top3_out", "d_r_tip_pm",
              "home_a_tip", "away_a_tip", "d_a_tip", "d_a_tip_missing"]
MARKET = ["mkt_spread", "mkt_prob", "mkt_total"]


def injury_cols(snap: str) -> list[str]:
    """Official injury report features for snapshot 'dayb' (previous day 5 PM) or 'pre' (game day)."""
    return [f"home_i_{snap}", f"away_i_{snap}", f"d_i_{snap}", f"home_il_{snap}", f"away_il_{snap}",
            f"d_il_{snap}", f"home_i_has_{snap}"]


def feature_set(tipoff: bool, market: bool, injury: str | None = None) -> list[str]:
    cols = RATING + FORM + SEASON + SCHED + ROSTER_DAYB
    if injury:
        cols += injury_cols(injury)
    if tipoff:
        cols += ROSTER_TIP
    if market:
        cols += MARKET
    return cols


def logistic(train: pd.DataFrame, cols: list[str], C: float = 0.05) -> object:
    X = train[cols].fillna(train[cols].median())
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    model.fit(X, train.home_win)
    model.medians_ = train[cols].median()
    return model


def predict_logistic(model: object, df: pd.DataFrame, cols: list[str]) -> F64:
    X = df[cols].fillna(model.medians_)  # type: ignore[attr-defined]
    p: F64 = model.predict_proba(X)[:, 1]  # type: ignore[attr-defined]
    return p


GBM_PARAMS: dict[str, object] = {
    "learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 200, "feature_fraction": 0.5,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "seed": 0,
    "num_threads": 8,
}


def gbm(train: pd.DataFrame, val: pd.DataFrame, cols: list[str], objective: str = "binary",
        params: dict[str, object] | None = None) -> lgb.Booster:
    target = "home_win" if objective == "binary" else "margin"
    p = {**GBM_PARAMS, **(params or {}), "objective": objective}
    dtr = lgb.Dataset(train[cols], train[target])
    dva = lgb.Dataset(val[cols], val[target], reference=dtr)
    return lgb.train(p, dtr, num_boost_round=5000, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(200, verbose=False)])


def margin_to_prob(margin: F64, sigma: float) -> F64:
    out: F64 = norm.cdf(margin / sigma)
    return out


def fit_sigma(pred_margin: F64, y: F64) -> float:
    """Pick the Normal sd that minimizes log loss of P(win) = Phi(margin / sigma)."""
    best, best_ll = 12.0, np.inf
    for s in np.arange(8.0, 18.0, 0.1):
        p = np.clip(margin_to_prob(pred_margin, s), 1e-6, 1 - 1e-6)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if ll < best_ll:
            best, best_ll = float(s), ll
    return best


def mlp(train: pd.DataFrame, stop: pd.DataFrame, pred: pd.DataFrame, cols: list[str],
        seeds: int = 5, hidden: int = 64, dropout: float = 0.3, wd: float = 1e-2, lr: float = 1e-3) -> F64:
    """Small PyTorch MLP (the same training loop as nba_train.py), early-stopped on `stop`,
    averaged over seeds in logit space. Returns P(home win) for `pred`."""
    import torch
    from torch import nn

    med = train[cols].median()
    mu, sd = train[cols].fillna(med).mean(), train[cols].fillna(med).std() + 1e-9

    def tens(d: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(((d[cols].fillna(med) - mu) / sd).to_numpy(np.float32))

    Xtr, Xst, Xpr = tens(train), tens(stop), tens(pred)
    ytr = torch.tensor(train.home_win.to_numpy(np.float32))
    yst = torch.tensor(stop.home_win.to_numpy(np.float32))
    logits = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(len(cols), hidden), nn.GELU(), nn.Dropout(dropout),
                            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        best, best_state, bad = np.inf, None, 0
        for _epoch in range(200):
            net.train()
            perm = torch.randperm(len(ytr))
            for i in range(0, len(ytr), 512):
                idx = perm[i : i + 512]
                loss = nn.functional.binary_cross_entropy_with_logits(net(Xtr[idx]).squeeze(1), ytr[idx])
                opt.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]
                opt.step()
            net.eval()
            with torch.no_grad():
                vl = float(nn.functional.binary_cross_entropy_with_logits(net(Xst).squeeze(1), yst))
            if vl < best - 1e-5:
                best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= 8:
                    break
        assert best_state is not None
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            logits.append(net(Xpr).squeeze(1).numpy())
    out: F64 = 1.0 / (1.0 + np.exp(-np.mean(logits, axis=0)))
    return out


def gbm_margin_predict(train: pd.DataFrame, pred: pd.DataFrame, cols: list[str]) -> F64:
    """LightGBM on point margin -> P(win) = Phi(margin / sigma). Rounds and sigma come from holding
    out the last training season; then refit on everything."""
    params: dict[str, object] = {"num_leaves": 7, "min_data_in_leaf": 400, "learning_rate": 0.01}
    last = train.SEASON.max()
    inner, stop = train[train.SEASON < last], train[train.SEASON == last]
    b = gbm(inner, stop, cols, "regression", params)
    sigma = fit_sigma(np.asarray(b.predict(stop[cols]), dtype=float), stop.home_win.to_numpy())
    full = lgb.train({**GBM_PARAMS, **params, "objective": "regression"},
                     lgb.Dataset(train[cols], train.margin), num_boost_round=int(b.best_iteration * 1.05))
    return margin_to_prob(np.asarray(full.predict(pred[cols]), dtype=float), sigma)


def final_predict(train: pd.DataFrame, pred: pd.DataFrame, cols: list[str]) -> F64:
    """The production model, chosen by walk-forward CV (see results.csv): an equal-weight
    logit-space average of logistic regression, the MLP and the margin GBM."""
    last = train.SEASON.max()
    parts = [
        predict_logistic(logistic(train, cols, C=0.003), pred, cols),
        mlp(train[train.SEASON < last], train[train.SEASON == last], pred, cols),
        gbm_margin_predict(train, pred, cols),
    ]
    z = np.mean([np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))) for p in parts], axis=0)
    out: F64 = 1.0 / (1.0 + np.exp(-z))
    return out


def market_prob(train: pd.DataFrame, pred: pd.DataFrame) -> F64:
    """Market-implied P(home win): vig-free moneyline where available, else spread mapped through a
    logistic fit on training games. NaN when ESPN has no line."""
    tr = train[train.mkt_spread.notna()]
    out = np.full(len(pred), np.nan)
    has = pred.mkt_spread.notna().to_numpy()
    if len(tr) and has.any():
        lr = LogisticRegression(C=10).fit(tr[["mkt_spread"]], tr.home_win)
        out[has] = lr.predict_proba(pred.loc[has, ["mkt_spread"]])[:, 1]
    ml = pred.mkt_prob.to_numpy(dtype=float)
    out = np.where(~np.isnan(ml), ml, out)
    return out
