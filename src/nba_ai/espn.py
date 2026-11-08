"""ESPN public API: schedules (scoreboard) and pregame market odds (summary pickcenter)."""
from __future__ import annotations

import datetime as dt
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import requests

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
_session = requests.Session()
_session.headers["User-Agent"] = "nba-ai/0.1"  # ESPN 403s a UA that contains a URL

# ESPN abbreviations -> nba_api TEAM_ABBREVIATION
ESPN_TO_NBA = {
    "GS": "GSW", "NY": "NYK", "SA": "SAS", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS",
    "PHO": "PHX", "BKN": "BKN", "NJ": "NJN", "NOH": "NOH", "NOK": "NOK", "SEA": "SEA",
}


def get(url: str, params: dict[str, str] | None = None, tries: int = 5) -> dict[str, Any]:
    for attempt in range(tries):
        try:
            r = _session.get(url, params=params, timeout=20)
            if r.status_code == 200:
                out: dict[str, Any] = r.json()
                return out
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"ESPN request failed: {url} {params}")


def nba_abbr(espn_abbr: str) -> str:
    return ESPN_TO_NBA.get(espn_abbr, espn_abbr)


def scoreboard(day: dt.date) -> list[dict[str, Any]]:
    """Regular-season games on a date: ESPN id, home/away (nba_api abbreviations), status."""
    payload = get(f"{SITE}/scoreboard", {"dates": day.strftime("%Y%m%d"), "seasontype": "2"})
    games = []
    for ev in payload.get("events", []):
        comp = ev["competitions"][0]
        side = {c["homeAway"]: c for c in comp["competitors"]}
        games.append(
            {
                "espn_id": ev["id"],
                "date": pd.Timestamp(day),
                "tipoff": ev.get("date"),
                "home": nba_abbr(side["home"]["team"]["abbreviation"]),
                "away": nba_abbr(side["away"]["team"]["abbreviation"]),
                "completed": bool(comp["status"]["type"].get("completed")),
            }
        )
    return games


def _american_to_prob(ml: float | None) -> float | None:
    if ml is None or abs(ml) < 100:  # ESPN sometimes reports 0 / junk
        return None
    return 100.0 / (ml + 100.0) if ml > 0 else -ml / (-ml + 100.0)


def odds(espn_id: str) -> dict[str, Any]:
    """Market line for one game: home spread, total, vig-free home win probability."""
    payload = get(f"{SITE}/summary", {"event": espn_id})
    row: dict[str, Any] = {"espn_id": espn_id}
    picks = payload.get("pickcenter") or []
    if not picks:
        return row
    pc = picks[0]
    home_ml = (pc.get("homeTeamOdds") or {}).get("moneyLine")
    away_ml = (pc.get("awayTeamOdds") or {}).get("moneyLine")
    spread = pc.get("spread")
    # ESPN's `spread` is from the home team's perspective (negative = home favored)
    row["mkt_provider"] = (pc.get("provider") or {}).get("name")
    row["mkt_spread"] = float(spread) if spread is not None else None
    row["mkt_total"] = float(pc["overUnder"]) if pc.get("overUnder") is not None else None
    ph, pa = _american_to_prob(home_ml), _american_to_prob(away_ml)
    if ph is not None and pa is not None:
        row["mkt_prob"] = ph / (ph + pa)
    return row


def fetch_season_odds(dates: list[dt.date], out: Path, workers: int = 8) -> pd.DataFrame:
    """Schedule + odds for every date in a season, cached to parquet."""
    if out.exists():
        return pd.read_parquet(out)
    with ThreadPoolExecutor(workers) as pool:
        games = [g for day in pool.map(scoreboard, dates) for g in day]
        lines = list(pool.map(odds, [g["espn_id"] for g in games]))
    df = pd.DataFrame(games).merge(pd.DataFrame(lines), on="espn_id", how="left") if games else pd.DataFrame()
    df.to_parquet(out)
    return df
