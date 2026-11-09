"""uv run nba-ai fetch | train | report | predict [--date YYYY-MM-DD]"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, espn, evaluate, features, models

FEATURES = data.DATA_DIR / "features.parquet"
PRED = data.DATA_DIR / "test_predictions.parquet"


def cmd_fetch(_: argparse.Namespace) -> None:
    data.fetch_logs()
    teams = data.load("T")
    for season in data.SEASONS:
        if season < "2012-13":
            continue
        out = data.DATA_DIR / f"odds_{season}.parquet"
        if season == data.CURRENT_SEASON:
            out.unlink(missing_ok=True)  # in progress: always refresh
        dates = sorted(teams[teams.SEASON == season].GAME_DATE.dt.date.unique())
        espn.fetch_season_odds(dates, out)
    print("fetched")


def build_table(future: pd.DataFrame | None = None) -> pd.DataFrame:
    teams, players = data.load("T"), data.load("P")
    games = features.build_games(teams)
    if future is not None and len(future):
        games = features.append_future(games, future)
    return features.build(games, teams, players)


def cmd_train(_: argparse.Namespace) -> None:
    """Fit on everything before the test season, score the untouched test season once."""
    df = build_table()
    df.to_parquet(FEATURES)
    played = df[df.home_win.notna()]
    train = played[played.SEASON < evaluate.TEST_SEASON]
    test = played[played.SEASON == evaluate.TEST_SEASON].copy()
    y = test.home_win.to_numpy()
    out = test[["GAME_ID", "GAME_DATE", "home_abbr", "away_abbr", "home_win", "margin",
                "elo_prob", "mkt_prob", "mkt_spread"]].copy()
    for variant, tip in (("daybefore", False), ("tipoff", True)):
        cols = models.feature_set(tip, False)
        out[f"p_{variant}"] = models.final_predict(train, test, cols)
    out["p_market"] = models.market_prob(train, test)
    out.to_parquet(PRED)

    rows = {"elo": out.elo_prob.to_numpy(), "model, day before": out.p_daybefore.to_numpy(),
            "model, at tip-off": out.p_tipoff.to_numpy()}
    print(f"\nTEST {evaluate.TEST_SEASON}: {len(test)} games")
    for name, p in rows.items():
        m = evaluate.metrics(y, p)
        evaluate.log(name, "final", "test", m)
        print(f"  {name:24s} logloss {m['logloss']:.4f}  acc {m['acc']:.1%}  brier {m['brier']:.4f}")
    has = out.p_market.notna().to_numpy()
    print(f"\n  head-to-head on the {has.sum()} games with a market line:")
    for name, p in {**rows, "market": out.p_market.to_numpy()}.items():
        m = evaluate.metrics(y[has], p[has])
        evaluate.log(name, "final", "test-mkt", m)
        print(f"  {name:24s} logloss {m['logloss']:.4f}  acc {m['acc']:.1%}")


def cmd_report(args: argparse.Namespace) -> None:
    from . import plots

    plots.report(pd.read_parquet(PRED), args.out)
    print(f"saved {args.out}")


def cmd_predict(args: argparse.Namespace) -> None:
    day = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    sched = [g for g in espn.scoreboard(day) if not g["completed"]]
    if not sched:
        print(f"no upcoming games on {day}")
        return
    teams = data.load("T", [data.CURRENT_SEASON])
    ids = dict(zip(teams.TEAM_ABBREVIATION, teams.TEAM_ID, strict=False))
    fut = pd.DataFrame(
        {
            "GAME_DATE": pd.Timestamp(day), "SEASON": data.CURRENT_SEASON,
            "home_id": [ids[g["home"]] for g in sched], "away_id": [ids[g["away"]] for g in sched],
            "home_abbr": [g["home"] for g in sched], "away_abbr": [g["away"] for g in sched],
            "GAME_ID": [f"F{g['espn_id']}" for g in sched],
        }
    )
    df = build_table(fut)
    played = df[df.home_win.notna()]
    upcoming = df[df.GAME_ID.isin(fut.GAME_ID)].copy()
    cols = models.feature_set(False, False)
    upcoming["p"] = models.final_predict(played, upcoming, cols)
    lines = {g["espn_id"]: espn.odds(g["espn_id"]) for g in sched}
    print(f"\n{day}  (model = day-before: no lineup info)\n")
    for _, r in upcoming.iterrows():
        fav, p = (r.home_abbr, r.p) if r.p >= 0.5 else (r.away_abbr, 1 - r.p)
        mk = lines[r.GAME_ID[1:]].get("mkt_prob")
        mtxt = f"   market {r.home_abbr} {mk:.0%}" if mk is not None and not np.isnan(mk) else ""
        print(f"  {r.away_abbr:>3} @ {r.home_abbr:<3}  {fav} {p:.0%}   (home {r.p:.0%}){mtxt}")
    if args.json:
        Path(args.json).write_text(json.dumps(upcoming[["home_abbr", "away_abbr", "p"]].to_dict("records"), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(prog="nba-ai", description="NBA game-winner model")
    sub = ap.add_subparsers(required=True)
    sub.add_parser("fetch", help="download/refresh box scores and ESPN odds").set_defaults(fn=cmd_fetch)
    sub.add_parser("train", help="build features, fit, score the test season").set_defaults(fn=cmd_train)
    rp = sub.add_parser("report", help="chart of test-season results")
    rp.add_argument("--out", default="report.png")
    rp.set_defaults(fn=cmd_report)
    pp = sub.add_parser("predict", help="win probabilities for a day's upcoming games")
    pp.add_argument("--date", help="YYYY-MM-DD (default today)")
    pp.add_argument("--json", help="also write predictions to this file")
    pp.set_defaults(fn=cmd_predict)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
