"""Fetch and cache NBA regular-season game logs (players and teams) per season via nba_api."""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
FIRST_SEASON, LAST_SEASON = 2004, 2026  # 2004-05 .. 2026-27
CURRENT_SEASON = f"{LAST_SEASON}-{str(LAST_SEASON + 1)[-2:]}"


def season_name(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


SEASONS: list[str] = [season_name(y) for y in range(FIRST_SEASON, LAST_SEASON + 1)]

PLAYER_COLS = [
    "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GAME_ID", "GAME_DATE", "MATCHUP", "WL",
    "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS",
]
TEAM_COLS = [
    "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MATCHUP", "WL",
    "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS",
]


def _fetch(season: str, kind: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog

    for attempt in range(6):
        try:
            df: pd.DataFrame = leaguegamelog.LeagueGameLog(
                season=season,
                player_or_team_abbreviation=kind,
                season_type_all_star="Regular Season",
                timeout=60,
            ).get_data_frames()[0]
            return df
        except Exception as exc:  # stats.nba.com is flaky
            print(f"  retry {attempt + 1} for {season} {kind}: {exc}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {season} {kind}")


def season_path(season: str, kind: str) -> Path:
    return DATA_DIR / f"{'players' if kind == 'P' else 'teams'}_{season}.parquet"


def fetch_logs(seasons: list[str] = SEASONS, refresh_current: bool = True) -> None:
    """Cache every season; the in-progress season is always refetched."""
    DATA_DIR.mkdir(exist_ok=True)
    for season in seasons:
        for kind, cols in (("P", PLAYER_COLS), ("T", TEAM_COLS)):
            path = season_path(season, kind)
            if path.exists() and not (refresh_current and season == CURRENT_SEASON):
                if set(cols) <= set(pd.read_parquet(path).columns):
                    continue
            df = _fetch(season, kind)[cols].copy()
            df["SEASON"] = season
            df.to_parquet(path)
            print(f"  {path}: {len(df):,} rows")
            time.sleep(0.8)


def load(kind: str, seasons: list[str] = SEASONS) -> pd.DataFrame:
    frames = [pd.read_parquet(season_path(s, kind)) for s in seasons if season_path(s, kind).exists()]
    df = pd.concat(frames, ignore_index=True)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df
