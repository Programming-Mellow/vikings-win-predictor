"""Collect each team's weekly news into data/news/{TEAM}.json.

Runs at the end of the daily prediction job. Feeds and nflverse tables are each
fetched once and then filtered per team, so a run makes a handful of requests in
total, not one per team. Any source that fails is skipped; the rest still save.

Usage:
    python -m vikpred.news.collect
"""

import json
import logging
from datetime import datetime, timezone

import httpx
import nflreadpy as nfl

from vikpred.config import DATA_DIR, current_season
from vikpred.data import load
from vikpred.data.ingest import normalize_teams
from vikpred.news.facts import team_facts
from vikpred.news.feeds import fetch, team_headlines
from vikpred.news.sources import SOURCES, USER_AGENT
from vikpred.teams import TEAM_NAMES

log = logging.getLogger(__name__)

NEWS_DIR = DATA_DIR / "news"


def _nflverse_tables(season: int):
    """Rosters and trades from nflverse; empty frames if a download fails."""
    import pandas as pd

    tables = {}
    for name, loader in [("rosters", lambda: nfl.load_rosters_weekly(season)), ("trades", nfl.load_trades)]:
        try:
            df = loader()
            df = normalize_teams(df, ["team", "gave", "received"])
            tables[name] = df.to_pandas()
        except Exception as exc:
            log.warning("nflverse %s unavailable: %s", name, exc)
            tables[name] = pd.DataFrame(columns=["team", "season", "week", "status", "gsis_id", "full_name",
                                                 "position", "trade_id", "trade_date", "gave", "received",
                                                 "pfr_name", "pick_season", "pick_round"])
    return tables["rosters"], tables["trades"]


def run(season: int | None = None, teams: list[str] | None = None) -> dict[str, dict]:
    season = season or current_season()
    teams = teams or list(TEAM_NAMES)
    now = datetime.now(timezone.utc)

    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=15, follow_redirects=True) as client:
        feeds = {source: fetch(source, client) for source in SOURCES}
    rosters, trades = _nflverse_tables(season)
    injuries, schedules = load.injuries(), load.schedules()

    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for team in teams:
        payload = {
            "team": team,
            "season": season,
            "generated_at": now.isoformat(timespec="seconds"),
            "facts": team_facts(team, season, injuries, rosters, trades, schedules, now),
            "headlines": team_headlines(feeds, team, now=now),
            "sources": [{"name": s.name, "url": s.url, "terms": s.terms} for s in SOURCES
                        if s.team in (None, team)] + [{"name": "nflverse", "url": "https://github.com/nflverse",
                                                       "terms": "Open NFL data; credited in the README."}],
        }
        path = NEWS_DIR / f"{team}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        tmp.replace(path)
        out[team] = payload
    log.info("news: %d teams, %d feed items fetched from %d sources", len(out),
             sum(len(v) for v in feeds.values()), len(SOURCES))
    return out


def load_team_news(team: str) -> dict | None:
    path = NEWS_DIR / f"{team}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    news = run()
    print(json.dumps(news.get("MIN"), indent=1)[:3000])
