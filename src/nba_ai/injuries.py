"""NBA official injury reports (PDFs the league publishes several times a day since 2018-19).

Each game gets two snapshots, both strictly before tip-off:
  * day before: the 5 PM ET report the previous day (it already lists the next day's games)
  * pre-game:   the game-day 1 PM or 5 PM ET report, whichever is latest and >= 30 min before tip
"""
from __future__ import annotations

import datetime as dt
import io
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import requests

URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{hh}{suffix}.pdf"
RAW = Path("data/injury_pdfs")
FIRST_DATE = dt.date(2018, 12, 1)
EDITIONS = {"01PM": 13, "05PM": 17}  # file tag -> hour ET
STATUSES = {"Out", "Doubtful", "Questionable", "Probable", "Available"}
_session = requests.Session()
_session.headers["User-Agent"] = "nba-ai/0.1"


def _download(args: tuple[dt.date, str]) -> Path | None:
    day, tag = args
    out = RAW / f"{day}_{tag}.pdf"
    miss = RAW / f"{day}_{tag}.missing"
    if out.exists():
        return out
    if miss.exists():
        return None
    hh, ampm = tag[:2], tag[2:]
    for suffix in (ampm, f"_00{ampm}"):  # naming switched to 05_00PM in early 2026
        url = URL.format(date=day.isoformat(), hh=hh, suffix=suffix)
        for attempt in range(3):
            try:
                r = _session.get(url, timeout=30)
            except requests.RequestException:
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                out.write_bytes(r.content)
                return out
            break
    miss.touch()
    return None


def _blocked() -> bool:
    """The CDN 403s everything for a while after too many requests; probe a file known to exist."""
    r = _session.get(URL.format(date="2023-12-15", hh="05", suffix="PM"), timeout=30)
    return r.status_code != 200


def download(game_dates: list[dt.date], pause: float = 1.5) -> None:
    """Sequential and slow on purpose: parallel requests get the client rate-limited (HTTP 403)."""
    RAW.mkdir(parents=True, exist_ok=True)
    days = sorted({d for d in game_dates if d >= FIRST_DATE})
    jobs = [(d, "01PM") for d in days] + [(d, "05PM") for d in sorted(set(days) | {d - dt.timedelta(1) for d in days})]
    jobs = [j for j in jobs if not (RAW / f"{j[0]}_{j[1]}.pdf").exists() and not (RAW / f"{j[0]}_{j[1]}.missing").exists()]
    for i, job in enumerate(jobs):
        if i % 50 == 0:
            while _blocked():
                print("  rate-limited; waiting 5 min", flush=True)
                time.sleep(300)
        if _download(job) is None and _blocked():  # a 403 caused by the block, not a missing file
            (RAW / f"{job[0]}_{job[1]}.missing").unlink(missing_ok=True)
            time.sleep(300)
        time.sleep(pause)
    got = len(list(RAW.glob("*.pdf")))
    print(f"  injury reports on disk: {got}")


# ------------------------------------------------------------------ parsing
HEADER_WORDS = {"Matchup": "matchup", "Team": "team", "Player": "player", "Category": "category",
                "Reason": "reason", "Current": "status", "Previous": "previous"}


def _columns(words: list[dict[str, object]]) -> list[tuple[str, float]] | None:
    """Column start x-positions from the page's header row (the layout changed over the years)."""
    tops = [float(str(w["top"])) for w in words if str(w["text"]) == "Matchup"]
    if not tops:
        return None
    row = sorted((w for w in words if abs(float(str(w["top"])) - tops[0]) < 4), key=lambda w: float(str(w["x0"])))
    cols: list[tuple[str, float]] = []
    games_seen = 0
    for w in row:
        t = str(w["text"])
        if t == "Game":
            cols.append(("game_date" if games_seen == 0 else "game_time", float(str(w["x0"]))))
            games_seen += 1
        elif t in HEADER_WORDS:
            cols.append((HEADER_WORDS[t], float(str(w["x0"]))))
    names = {c for c, _ in cols}
    return cols if {"game_date", "matchup", "team", "player", "status"} <= names else None


def _col(x: float, cols: list[tuple[str, float]]) -> str:
    name = cols[0][0]
    for n, x0 in cols:
        if x >= x0 - 3:
            name = n
    return name


def parse(path: Path) -> pd.DataFrame:
    """One row per listed player: game_date, game_time, matchup, team, player ('Last, First'), status."""
    import pdfplumber

    rows: list[dict[str, str]] = []
    cur = {"game_date": "", "game_time": "", "matchup": "", "team": ""}
    cols: list[tuple[str, float]] | None = None
    with pdfplumber.open(io.BytesIO(path.read_bytes())) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=1.5, y_tolerance=2)
            cols = _columns(words) or cols  # pages without a header reuse the last layout
            if cols is None:
                continue
            lines: dict[int, list[dict[str, object]]] = {}
            for w in words:
                lines.setdefault(round(float(str(w["top"])) / 3), []).append(w)
            for _, ws in sorted(lines.items()):
                raw = " ".join(str(w["text"]) for w in ws)
                if raw.startswith(("Game Date", "Injury Report", "Page")) or "Injury Report:" in raw:
                    continue
                cells: dict[str, list[str]] = {}
                for w in sorted(ws, key=lambda w: float(str(w["x0"]))):
                    cells.setdefault(_col(float(str(w["x0"])), cols), []).append(str(w["text"]))
                txt = {k: " ".join(v) for k, v in cells.items()}
                if re.fullmatch(r"\d\d/\d\d/\d{4}", txt.get("game_date", "")):
                    cur["game_date"] = txt["game_date"]
                if re.match(r"\d\d:\d\d", txt.get("game_time", "")):
                    cur["game_time"] = txt["game_time"]
                if "@" in txt.get("matchup", ""):
                    cur["matchup"] = txt["matchup"]
                    cur["team"] = ""
                if txt.get("team") and not txt["team"].upper().startswith("NOT YET"):
                    cur["team"] = txt["team"]
                status = txt.get("status", "").split(" ")[0]
                if status in STATUSES and txt.get("player"):
                    rows.append({**cur, "player": txt["player"], "status": status})
    return pd.DataFrame(rows)


def _parse_one(path: Path) -> pd.DataFrame:
    try:
        df = parse(path)
    except Exception as exc:  # a few PDFs are malformed
        print(f"  could not parse {path.name}: {exc}")
        return pd.DataFrame()
    day, tag = path.stem.rsplit("_", 1)
    df["report_day"] = day
    df["report_hour"] = EDITIONS[tag]
    return df


def parse_all(out: Path = Path("data/injuries.parquet"), workers: int = 8) -> pd.DataFrame:
    paths = sorted(RAW.glob("*.pdf"))
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame()
    seen = set(zip(done.report_day, done.report_hour, strict=False)) if len(done) else set()
    todo = [p for p in paths if (p.stem.rsplit("_", 1)[0], EDITIONS[p.stem.rsplit("_", 1)[1]]) not in seen]
    with ProcessPoolExecutor(workers) as pool:
        frames = list(pool.map(_parse_one, todo, chunksize=8))
    df = pd.concat([done, *frames], ignore_index=True) if frames else done
    df.to_parquet(out)
    print(f"  parsed {len(todo)} new reports; {len(df):,} player-status rows total")
    return df


# ------------------------------------------------------------------ matching to games and players
def _norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", name.lower().replace("jr.", "").replace("sr.", "")
                  .replace(" iii", "").replace(" ii", "").replace(" iv", ""))


def _first_last(report_name: str) -> str:
    """'Bagley III, Marvin' -> 'Marvin Bagley III'."""
    if "," in report_name:
        last, first = report_name.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return report_name


def team_lookup() -> dict[str, int]:
    """Full team name (as printed on the report) -> TEAM_ID."""
    from nba_api.stats.static import teams

    out: dict[str, int] = {}
    for t in teams.get_teams():
        out[_norm(t["full_name"])] = int(t["id"])
        out[_norm(f"{t['city']} {t['nickname']}")] = int(t["id"])
    out[_norm("LA Clippers")] = out[_norm("Los Angeles Clippers")]
    return out


def statuses(inj: pd.DataFrame, games: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Per (GAME_ID, team_id, PLAYER_ID): status in the day-before and pre-game snapshots."""
    if inj.empty:
        return pd.DataFrame(columns=["GAME_ID", "team_id", "PLAYER_ID", "status_dayb", "status_pre"])
    inj = inj.copy()
    tl = team_lookup()
    from nba_api.stats.static import teams as static_teams

    abbr = {t["abbreviation"]: t for t in static_teams.get_teams()}

    def resolve(team: str, matchup: str) -> int | None:
        if _norm(team) in tl:
            return tl[_norm(team)]
        # long names wrap onto two lines ("Minnesota" / "Timberwolves"): match the fragment
        # against the two teams in the matchup code
        hits = [int(abbr[a]["id"]) for a in matchup.split("@") if a in abbr
                and _norm(team) and _norm(team) in _norm(abbr[a]["full_name"])]
        return hits[0] if len(hits) == 1 else None

    inj["team_id"] = [resolve(t, m) for t, m in zip(inj.team, inj.matchup, strict=True)]
    inj["gdate"] = pd.to_datetime(inj.game_date, format="%m/%d/%Y", errors="coerce")
    inj["rday"] = pd.to_datetime(inj.report_day)
    hh = inj.game_time.str.extract(r"(\d\d):(\d\d)").astype(float)
    inj["tip_hour"] = (hh[0] % 12 + 12 + hh[1] / 60).where(hh[0].notna())  # all tips are PM ET
    inj = inj.dropna(subset=["team_id", "gdate"])

    # player name -> PLAYER_ID, per team (fallback: league-wide unique name)
    names = players[["PLAYER_ID", "PLAYER_NAME", "TEAM_ID"]].drop_duplicates()
    names = names.assign(key=names.PLAYER_NAME.map(_norm))
    by_team = {(int(t), k): int(p) for p, k, t in names[["PLAYER_ID", "key", "TEAM_ID"]].itertuples(index=False)}
    uniq = names.drop_duplicates("PLAYER_ID").groupby("key").PLAYER_ID.agg(list)
    by_name = {k: int(v[0]) for k, v in uniq.items() if len(v) == 1}
    inj["key"] = inj.player.map(lambda s: _norm(_first_last(s)))
    inj["PLAYER_ID"] = [by_team.get((int(t), k), by_name.get(k)) for t, k in zip(inj.team_id, inj.key, strict=True)]
    inj = inj.dropna(subset=["PLAYER_ID"])
    inj["PLAYER_ID"] = inj.PLAYER_ID.astype(int)
    inj["team_id"] = inj.team_id.astype(int)

    dayb = inj[(inj.rday == inj.gdate - pd.Timedelta(days=1)) & (inj.report_hour == 17)]
    pre = inj[(inj.rday == inj.gdate) & (inj.report_hour + 0.5 <= inj.tip_hour)]
    pre = pre.sort_values("report_hour").drop_duplicates(["gdate", "team_id", "PLAYER_ID"], keep="last")
    dayb = dayb.drop_duplicates(["gdate", "team_id", "PLAYER_ID"], keep="last")

    side = pd.concat([
        games[["GAME_ID", "GAME_DATE", "home_id"]].rename(columns={"home_id": "team_id"}),
        games[["GAME_ID", "GAME_DATE", "away_id"]].rename(columns={"away_id": "team_id"}),
    ])
    out = []
    for snap, name in ((dayb, "status_dayb"), (pre, "status_pre")):
        m = side.merge(snap[["gdate", "team_id", "PLAYER_ID", "status"]],
                       left_on=["GAME_DATE", "team_id"], right_on=["gdate", "team_id"])
        out.append(m[["GAME_ID", "team_id", "PLAYER_ID", "status"]].rename(columns={"status": name}))
    res = out[0].merge(out[1], on=["GAME_ID", "team_id", "PLAYER_ID"], how="outer")
    return res


def coverage(games: pd.DataFrame, st: pd.DataFrame) -> pd.Series:
    """Share of games per season with any pre-game report entry."""
    have = set(st.loc[st.status_pre.notna(), "GAME_ID"])
    return games.assign(has=games.GAME_ID.isin(have)).groupby("SEASON").has.mean()
