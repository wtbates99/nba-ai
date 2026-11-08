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


def feature_set(tipoff: bool, market: bool) -> list[str]:
    cols = RATING + FORM + SEASON + SCHED + ROSTER_DAYB
    if tipoff:
        cols += ROSTER_TIP
    if market:
        cols += MARKET
    return cols


def logistic(train: pd.DataFrame, cols: list[str], C: float = 0.05) -> object:
    X = train[cols].fillna(train[cols].median())
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    model.fit(X, train.home_win)
    model.medians_ = train[cols].median()  # type: ignore[attr-defined]
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
