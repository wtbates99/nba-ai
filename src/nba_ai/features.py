"""Build one row per game with strictly pre-game features (home and away sides plus diffs)."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from . import ratings

CANON = {"NJN": "BKN", "NOH": "NOP", "NOK": "NOP", "SEA": "OKC"}

# arena (lat, lon) by abbreviation as it appears in that season's logs
ARENA = {
    "ATL": (33.757, -84.396), "BOS": (42.366, -71.062), "BKN": (40.683, -73.976), "NJN": (40.734, -74.171),
    "CHA": (35.225, -80.839), "CHI": (41.881, -87.674), "CLE": (41.497, -81.688), "DAL": (32.790, -96.810),
    "DEN": (39.749, -105.008), "DET": (42.341, -83.055), "GSW": (37.768, -122.388), "HOU": (29.751, -95.362),
    "IND": (39.764, -86.156), "LAC": (34.043, -118.267), "LAL": (34.043, -118.267), "MEM": (35.138, -90.051),
    "MIA": (25.781, -80.188), "MIL": (43.045, -87.917), "MIN": (44.979, -93.276), "NOP": (29.949, -90.082),
    "NOH": (29.949, -90.082), "NOK": (35.463, -97.515), "NYK": (40.751, -73.994), "OKC": (35.463, -97.515),
    "SEA": (47.622, -122.354), "ORL": (28.539, -81.384), "PHI": (39.901, -75.172), "PHX": (33.446, -112.071),
    "POR": (45.532, -122.667), "SAC": (38.580, -121.500), "SAS": (29.427, -98.438), "TOR": (43.643, -79.379),
    "UTA": (40.768, -111.901), "WAS": (38.898, -77.021),
}
ALTITUDE_TEAMS = {"DEN", "UTA"}

FORM_STATS = ["net", "ortg", "drtg", "pace", "efg", "oefg", "tov", "otov", "orb", "drb",
              "ftr", "oftr", "par3", "opar3", "margin", "win"]
FORM_HL = [5, 15, 40]


# ------------------------------------------------------------------ games
def build_games(teams: pd.DataFrame) -> pd.DataFrame:
    """Pair the two team rows of each game into one row (home vs away)."""
    t = teams.copy()
    t["is_home"] = t.MATCHUP.str.contains(" vs. ")
    home = t[t.is_home].set_index("GAME_ID")
    away = t[~t.is_home].set_index("GAME_ID")
    both = home.index.intersection(away.index)
    g = pd.DataFrame(
        {
            "GAME_ID": both,
            "GAME_DATE": home.loc[both, "GAME_DATE"].to_numpy(),
            "SEASON": home.loc[both, "SEASON"].to_numpy(),
            "home_id": home.loc[both, "TEAM_ID"].to_numpy(),
            "away_id": away.loc[both, "TEAM_ID"].to_numpy(),
            "home_abbr": home.loc[both, "TEAM_ABBREVIATION"].to_numpy(),
            "away_abbr": away.loc[both, "TEAM_ABBREVIATION"].to_numpy(),
            "home_pts": home.loc[both, "PTS"].to_numpy(dtype=float),
            "away_pts": away.loc[both, "PTS"].to_numpy(dtype=float),
        }
    )
    g["margin"] = g.home_pts - g.away_pts
    g["home_win"] = (g.margin > 0).astype(float)
    return g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def append_future(games: pd.DataFrame, future: pd.DataFrame) -> pd.DataFrame:
    """Add unplayed games (GAME_DATE, SEASON, home_id, away_id, home_abbr, away_abbr) with NaN results."""
    f = future.copy()
    for c in ("home_pts", "away_pts", "margin", "home_win"):
        f[c] = np.nan
    if "GAME_ID" not in f:
        f["GAME_ID"] = [f"F{i}" for i in range(len(f))]
    return pd.concat([games, f], ignore_index=True).sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


# ------------------------------------------------------------------ team form
def team_long(games: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with box-derived per-game stats (NaN for unplayed games)."""
    box = teams.set_index(["GAME_ID", "TEAM_ID"])
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        r = games[["GAME_ID", "GAME_DATE", "SEASON"]].copy()
        r["team_id"] = games[f"{side}_id"]
        r["opp_id"] = games[f"{opp}_id"]
        r["abbr"] = games[f"{side}_abbr"]
        r["venue"] = games["home_abbr"]
        r["is_home"] = side == "home"
        rows.append(r)
    tl = pd.concat(rows, ignore_index=True)
    cols = ["FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "TOV", "PTS", "MIN"]
    me = box[cols].reindex(pd.MultiIndex.from_arrays([tl.GAME_ID, tl.team_id])).reset_index(drop=True)
    op = box[cols].reindex(pd.MultiIndex.from_arrays([tl.GAME_ID, tl.opp_id])).reset_index(drop=True)
    p_me = me.FGA + 0.44 * me.FTA - me.OREB + me.TOV
    p_op = op.FGA + 0.44 * op.FTA - op.OREB + op.TOV
    poss = 0.5 * (p_me + p_op)
    tl["ortg"] = 100 * me.PTS / poss
    tl["drtg"] = 100 * op.PTS / poss
    tl["net"] = tl.ortg - tl.drtg
    tl["pace"] = poss * 240.0 / me.MIN.clip(lower=240)
    tl["efg"] = (me.FGM + 0.5 * me.FG3M) / me.FGA
    tl["oefg"] = (op.FGM + 0.5 * op.FG3M) / op.FGA
    tl["tov"] = me.TOV / p_me
    tl["otov"] = op.TOV / p_op
    tl["orb"] = me.OREB / (me.OREB + op.DREB)
    tl["drb"] = me.DREB / (me.DREB + op.OREB)
    tl["ftr"] = me.FTM / me.FGA
    tl["oftr"] = op.FTM / op.FGA
    tl["par3"] = me.FG3A / me.FGA
    tl["opar3"] = op.FG3A / op.FGA
    tl["margin"] = me.PTS - op.PTS
    tl["win"] = (tl.margin > 0).astype(float).where(tl.margin.notna())
    return tl.sort_values(["team_id", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def _haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(6371 * 2 * np.arcsin(np.sqrt(h)))


def team_features(tl: pd.DataFrame) -> pd.DataFrame:
    """Pre-game rolling form + schedule features, one row per (GAME_ID, team_id)."""
    out = tl[["GAME_ID", "team_id"]].copy()
    g = tl.groupby("team_id", sort=False)
    for hl in FORM_HL:
        ew = g[FORM_STATS].transform(lambda s, hl=hl: s.ewm(halflife=hl, ignore_na=True).mean().shift(1))
        for c in FORM_STATS:
            out[f"{c}_ew{hl}"] = ew[c]
    gs = tl.groupby(["team_id", "SEASON"], sort=False)
    out["season_gp"] = gs.cumcount()
    out["season_win"] = gs["win"].transform(lambda s: s.shift(1).expanding().mean())
    out["season_margin"] = gs["margin"].transform(lambda s: s.shift(1).expanding().mean())

    # schedule
    prev_date = g["GAME_DATE"].shift(1)
    rest = (tl.GAME_DATE - prev_date).dt.days
    new_season = tl.SEASON != g["SEASON"].shift(1)
    out["rest"] = rest.where(~new_season, 10).clip(upper=10).fillna(10)
    out["b2b"] = (out.rest == 1).astype(float)
    dates = tl.GAME_DATE.to_numpy()
    teams_arr = tl.team_id.to_numpy()
    last7 = np.zeros(len(tl))
    travel = np.zeros(len(tl))
    venues = tl.venue.to_numpy()
    start = 0
    for i in range(len(tl)):
        if i == 0 or teams_arr[i] != teams_arr[i - 1]:
            start = i
        j = i - 1
        cnt = 0
        while j >= start and (dates[i] - dates[j]).astype("timedelta64[D]").astype(int) <= 7:
            cnt += 1
            j -= 1
        last7[i] = cnt
        if i > start and not new_season.iloc[i]:
            travel[i] = _haversine(ARENA[venues[i - 1]], ARENA[venues[i]])
    out["games_last7"] = last7
    out["travel_km"] = travel
    return out


# ------------------------------------------------------------------ roster strength
def roster_features(games: pd.DataFrame, players: pd.DataFrame, hl: float = 15.0,
                    apm_lam: float = 10.0, apm_hl_days: float = 400.0, apm_window_days: float = 1100.0,
                    refit_days: int = 7) -> pd.DataFrame:
    """Lineup strength from player history, walk-forward.

    val(p)  = EWMA of the player's Game Score per game before today (rookies = 3.0)
    apm(p)  = game-level adjusted plus-minus per 48 (ridge toward a box prior), refit weekly
    mins(p) = EWMA of minutes, used to weight apm into a team rating (scaled to 240 team minutes)
    usual   = team's last-10-game regulars weighted by appearance rate   (known the day before)
    stale   = players who played the team's previous game                 (known the day before)
    tip     = players who actually played this game                       (known at tip-off)
    """
    p = players.copy()
    p["gs"] = (p.PTS + 0.4 * p.FGM - 0.7 * p.FGA - 0.4 * (p.FTA - p.FTM) + 0.7 * p.OREB
               + 0.3 * p.DREB + p.STL + 0.7 * p.AST + 0.7 * p.BLK - 0.4 * p.PF - p.TOV)
    p = p.sort_values(["GAME_DATE", "GAME_ID"])
    alpha = 1 - 0.5 ** (1 / hl)
    by_game: dict[tuple[str, int], list[tuple[int, float, float, float]]] = {}
    for gid, tid, pid, gsv, mn, pm in p[["GAME_ID", "TEAM_ID", "PLAYER_ID", "gs", "MIN", "PLUS_MINUS"]].itertuples(index=False):
        by_game.setdefault((gid, tid), []).append((pid, float(gsv), float(mn), float(pm)))

    val: dict[int, float] = {}
    pmv: dict[int, float] = {}
    mins: dict[int, float] = {}
    last_team: dict[int, int] = {}
    recent: dict[int, deque[frozenset[int]]] = {}
    apm: dict[int, float] = {}
    past: list[ratings.ApmGame] = []
    ROOKIE_GS, ROOKIE_MIN, ROOKIE_APM = 3.0, 12.0, -2.0
    ROT_MIN = 12.0
    next_refit = -1e9

    def v(pid: int) -> float:
        return val.get(pid, ROOKIE_GS)

    def box_prior(pid: int) -> float:  # per-48 rating guess from Game Score per minute
        if pid not in val:
            return ROOKIE_APM
        per48 = 48.0 * val[pid] / max(mins.get(pid, ROOKIE_MIN), 6.0)
        return float(np.clip(0.55 * (per48 - 17.0), -8.0, 8.0))

    def team_apm(weights: dict[int, float]) -> float:
        tot = sum(weights.values())
        if tot <= 0:
            return np.nan
        return sum(apm.get(pid, box_prior(pid)) * w for pid, w in weights.items()) * (240.0 / tot) / 48.0

    feats: dict[tuple[str, int], tuple[float, ...]] = {}
    for date, day in games.groupby("GAME_DATE", sort=True):
        dnum = float(pd.Timestamp(str(date)).toordinal())
        if dnum >= next_refit:
            prior = {pid: box_prior(pid) for pid in val}
            apm = ratings.fit_apm(past, dnum, apm_hl_days, apm_window_days, apm_lam, prior)
            next_refit = dnum + refit_days
        updates: list[tuple[int, int, float, float, float]] = []
        for gid, h, a, margin in day[["GAME_ID", "home_id", "away_id", "margin"]].itertuples(index=False):
            sides: list[list[tuple[int, float]]] = []
            for tid in (h, a):
                hist = recent.setdefault(tid, deque(maxlen=10))
                counts: dict[int, int] = {}
                for s in hist:
                    for pid in s:
                        counts[pid] = counts.get(pid, 0) + 1
                regs = {pid: c / max(len(hist), 1) for pid, c in counts.items() if last_team.get(pid) == tid}
                usual = sum(v(pid) * r for pid, r in regs.items())
                a_usual = team_apm({pid: r * mins.get(pid, ROOKIE_MIN) for pid, r in regs.items()})
                top3 = sorted(regs, key=v, reverse=True)[:3]
                stale_set = [pid for pid in (hist[-1] if hist else frozenset()) if last_team.get(pid) == tid]
                stale = sum(v(pid) for pid in stale_set)
                a_stale = team_apm({pid: mins.get(pid, ROOKIE_MIN) for pid in stale_set})
                played = by_game.get((gid, tid))
                if played:
                    # Only ROTATION players (pre-game EWMA minutes >= ROT_MIN) count as "known at tip-off".
                    # Who else got in is post-game info: blowouts empty the bench, which leaks the margin.
                    ids = {x[0] for x in played if mins.get(x[0], 0.0) >= ROT_MIN}
                    tip = sum(v(pid) for pid in ids)
                    top3_out = float(sum(1 for pid in top3 if pid not in ids))
                    tip_pm = sum(pmv.get(pid, 0.0) for pid in ids)
                    a_tip = team_apm({pid: mins.get(pid, ROOKIE_MIN) for pid in ids})
                    for pid, gsv, mn, pm in played:
                        updates.append((pid, tid, gsv, mn, pm))
                    sides.append([(pid, mn) for pid, _, mn, _ in played])
                else:
                    tip = top3_out = tip_pm = a_tip = np.nan
                feats[(gid, tid)] = (usual, stale, tip, top3_out, tip_pm, len(hist), a_usual, a_stale, a_tip)
                if played:
                    recent[tid].append(frozenset(x[0] for x in played))
            if len(sides) == 2 and not np.isnan(margin):
                past.append((dnum, float(margin), sides[0], sides[1]))
        for pid, tid, gsv, mn, pm in updates:  # update after the whole day: no same-day leakage
            val[pid] = gsv if pid not in val else (1 - alpha) * val[pid] + alpha * gsv
            pmv[pid] = pm if pid not in pmv else (1 - alpha) * pmv[pid] + alpha * pm
            mins[pid] = mn if pid not in mins else (1 - alpha) * mins[pid] + alpha * mn
            last_team[pid] = tid
    idx = pd.MultiIndex.from_tuples(list(feats.keys()), names=["GAME_ID", "team_id"])
    df = pd.DataFrame(list(feats.values()), index=idx,
                      columns=["r_usual", "r_stale", "r_tip", "r_top3_out", "r_tip_pm", "r_hist",
                               "a_usual", "a_stale", "a_tip"])
    df["r_stale_missing"] = df.r_usual - df.r_stale
    df["r_tip_missing"] = df.r_usual - df.r_tip
    df["a_tip_missing"] = df.a_usual - df.a_tip
    return df.reset_index()


# ------------------------------------------------------------------ market
def load_market(data_dir: Path = Path("data")) -> pd.DataFrame:
    frames = [pd.read_parquet(f) for f in sorted(data_dir.glob("odds_*.parquet"))]
    if not frames:
        return pd.DataFrame(columns=["GAME_DATE", "home_c", "away_c", "mkt_spread", "mkt_prob", "mkt_total"])
    m = pd.concat(frames, ignore_index=True)
    m["GAME_DATE"] = pd.to_datetime(m["date"])
    m["home_c"] = m.home.map(lambda x: CANON.get(x, x))
    m["away_c"] = m.away.map(lambda x: CANON.get(x, x))
    for c in ("mkt_spread", "mkt_prob", "mkt_total"):
        if c not in m:
            m[c] = np.nan
    m = m.drop_duplicates(["GAME_DATE", "home_c", "away_c"])
    return m[["GAME_DATE", "home_c", "away_c", "mkt_spread", "mkt_prob", "mkt_total"]]


# ------------------------------------------------------------------ assemble
def build(games: pd.DataFrame, teams: pd.DataFrame, players: pd.DataFrame,
          elo_params: ratings.EloParams = ratings.EloParams()) -> pd.DataFrame:
    tl = team_long(games, teams)
    tf = team_features(tl)
    rf = roster_features(games, players)
    df = games.copy()
    for side in ("home", "away"):
        s = tf.merge(rf, on=["GAME_ID", "team_id"], how="left")
        s = s.rename(columns={c: f"{side}_{c}" for c in s.columns if c not in ("GAME_ID", "team_id")})
        df = df.merge(s, left_on=["GAME_ID", f"{side}_id"], right_on=["GAME_ID", "team_id"], how="left").drop(columns="team_id")
    df = pd.concat([df, ratings.elo(df, elo_params), ratings.srs(df)], axis=1)

    # diffs (home - away) for every per-side feature
    side_cols = [c[5:] for c in df.columns if c.startswith("home_") and f"away_{c[5:]}" in df.columns
                 and c[5:] not in ("id", "abbr", "pts", "win")]
    diffs = pd.DataFrame({f"d_{c}": df[f"home_{c}"] - df[f"away_{c}"] for c in side_cols}, index=df.index)
    df = pd.concat([df, diffs], axis=1).copy()
    df["d_elo"] = df.elo_home - df.elo_away
    df["home_alt"] = df.home_abbr.isin(ALTITUDE_TEAMS).astype(float)
    df["month"] = df.GAME_DATE.dt.month
    df["home_c"] = df.home_abbr.map(lambda x: CANON.get(x, x))
    df["away_c"] = df.away_abbr.map(lambda x: CANON.get(x, x))
    df = df.merge(load_market(), on=["GAME_DATE", "home_c", "away_c"], how="left")
    # league-wide home advantage drifts over the years; use only earlier game days
    for c, hl in (("margin", 600), ("home_win", 600)):
        ew = df[c].ewm(halflife=hl, ignore_na=True).mean()
        by_day = ew.groupby(df.GAME_DATE).last().shift(1)
        df[f"lg_home_{c}"] = df.GAME_DATE.map(by_day)
    return df
