"""Leakage guard: features for games on or before date D must not change when every game after D is
removed. If any did, the model would be peeking at the future."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nba_ai import data, features, ratings

SEASONS = ["2022-23", "2023-24"]
CUTOFF = pd.Timestamp("2024-01-15")


@pytest.fixture(scope="module")
def logs() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not all(data.season_path(s, k).exists() for s in SEASONS for k in ("T", "P")):
        pytest.skip("run `nba-ai fetch` first")
    return data.load("T", SEASONS), data.load("P", SEASONS)


def test_no_future_leakage(logs: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    teams, players = logs
    full = features.build(features.build_games(teams), teams, players)
    cut_t, cut_p = teams[teams.GAME_DATE <= CUTOFF], players[players.GAME_DATE <= CUTOFF]
    cut = features.build(features.build_games(cut_t), cut_t, cut_p)

    a = full[full.GAME_DATE <= CUTOFF].set_index("GAME_ID").sort_index()
    b = cut.set_index("GAME_ID").sort_index()
    assert list(a.index) == list(b.index)
    num = [c for c in b.columns if pd.api.types.is_numeric_dtype(b[c])]
    diff = (a[num] - b[num]).abs().max()
    leaking = diff[diff > 1e-6]
    assert leaking.empty, f"features changed when the future was removed: {leaking.to_dict()}"


def test_elo_is_zero_sum_and_pregame() -> None:
    g = pd.DataFrame({
        "SEASON": ["s"] * 3, "home_id": [1, 2, 1], "away_id": [2, 1, 2], "margin": [10.0, 5.0, np.nan],
    })
    e = ratings.elo(g)
    assert e.elo_home.iloc[0] == e.elo_away.iloc[0] == 1500.0  # first game sees no result
    assert np.isclose(e.elo_home.iloc[1] + e.elo_away.iloc[1], 3000.0)  # zero-sum update
    assert e.elo_away.iloc[1] > 1500.0  # team 1 won game 0, so it is rated higher going into game 1


def test_garbage_time_players_do_not_change_tipoff_features(logs: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    """Who checks in late in a blowout is decided by the score. Adding a deep-bench player with no
    rotation history to a game's box must not move any tip-off feature for that game."""
    teams, players = logs
    games = features.build_games(teams)
    target = games[games.GAME_DATE > CUTOFF].iloc[0]
    extra = players[players.GAME_ID == target.GAME_ID].iloc[[0]].copy()
    extra["PLAYER_ID"] = 999_999_999  # never seen before -> no rotation minutes
    extra["MIN"] = 3
    base = features.roster_features(games, players).set_index(["GAME_ID", "team_id"])
    more = features.roster_features(games, pd.concat([players, extra])).set_index(["GAME_ID", "team_id"])
    key = (target.GAME_ID, int(extra.TEAM_ID.iloc[0]))
    tip_cols = ["r_tip", "r_top3_out", "r_tip_pm", "a_tip", "r_tip_missing", "a_tip_missing"]
    pd.testing.assert_series_equal(base.loc[key, tip_cols], more.loc[key, tip_cols])
