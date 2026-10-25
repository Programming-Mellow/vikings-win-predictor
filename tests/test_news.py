"""Team news: feed parsing, team filtering, nflverse facts, Haiku tags, API, MCP (all offline)."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from test_insights import client_returning
from vikpred.api import main as api
from vikpred.llm.insights import tag_facts
from vikpred.mcp_server import server
from vikpred.news import collect, facts, feeds
from vikpred.news.sources import Source

NOW = datetime(2026, 10, 25, 22, 0, tzinfo=timezone.utc)

RSS = b"""<?xml version="1.0"?><rss version="2.0"><channel>
<item><title>Vikings place Darrisaw on IR</title><link>https://a.example/1</link>
  <pubDate>Sat, 24 Oct 2026 15:00:00 GMT</pubDate><description>Left tackle out</description></item>
<item><title>Packers sign kicker</title><link>https://a.example/2</link><pubDate>Sat, 24 Oct 2026 16:00:00 GMT</pubDate></item>
<item><title>How to bet Vikings-Lions: odds and picks</title><link>https://a.example/3</link>
  <pubDate>Sat, 24 Oct 2026 17:00:00 GMT</pubDate></item>
<item><title>Old Vikings story</title><link>https://a.example/4</link><pubDate>Mon, 01 Sep 2026 12:00:00 GMT</pubDate></item>
<item><title>Bad link Vikings</title><link>javascript:alert(1)</link><pubDate>Sat, 24 Oct 2026 12:00:00 GMT</pubDate></item>
<item><title>Vikings&amp;#39; defense dominates</title><link>https://a.example/5</link>
  <pubDate>Sun, 25 Oct 2026 20:00:00 GMT</pubDate></item>
</channel></rss>"""

ATOM = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<entry><title>Defense Rules Again As Vikings Move To 6-0</title><link href="https://team.example/x"/>
  <updated>2026-10-25T19:59:07Z</updated><summary>Recap</summary></entry>
</feed>"""


# --- feeds -----------------------------------------------------------------------------

def test_parse_rss_and_atom_keep_titles_and_drop_unsafe_links():
    rss = [h for h, _ in feeds.parse_feed(RSS, "National")]
    assert [h.link for h in rss] == ["https://a.example/1", "https://a.example/2", "https://a.example/3",
                                     "https://a.example/4", "https://a.example/5"]
    assert rss[-1].title == "Vikings' defense dominates"  # double-encoded entity rendered
    atom = feeds.parse_feed(ATOM, "Team")
    assert atom[0][0].title.startswith("Defense Rules") and atom[0][0].published.startswith("2026-10-25T19:59")


@pytest.mark.parametrize("text, category", [
    ("Vikings place Darrisaw on IR", "Injury"),
    ("Vikings fire defensive coordinator", "Coaching"),
    ("Player suspended two games after injury", "Suspension"),
    ("Vikings acquire edge rusher in trade", "Trade"),
    ("Vikings sign veteran to practice squad", "Roster move"),
    ("Their aim is high", None),  # lowercase "ir" inside words isn't injured reserve
])
def test_categorize(text, category):
    assert feeds.categorize(text) == category


def test_team_headlines_filter_dedupe_recent_and_no_betting():
    national = Source("National", "https://n.example/rss", None, "")
    team = Source("Team", "https://t.example/rss", "MIN", "")
    syndicated = feeds.parse_feed(ATOM.replace(b"https://team.example/x", b"https://n.example/copy"), "National")
    got = feeds.team_headlines({national: feeds.parse_feed(RSS, "National") + syndicated,
                                team: feeds.parse_feed(ATOM, "Team")}, "MIN", now=NOW)
    titles = [h["title"] for h in got]
    assert titles == ["Vikings' defense dominates", "Defense Rules Again As Vikings Move To 6-0",
                      "Vikings place Darrisaw on IR"]
    assert got[1]["source"] == "Team"  # the syndicated copy is dropped, original source kept
    assert got[2]["category"] == "Injury"


def test_team_feed_not_shown_for_other_teams():
    team = Source("Team", "https://t.example/rss", "MIN", "")
    assert feeds.team_headlines({team: feeds.parse_feed(ATOM, "Team")}, "GB", now=NOW) == []


# --- nflverse facts ------------------------------------------------------------------------

def injuries_frame():
    rows = [("MIN", 7, "p1", "Justin Jefferson", "WR", "Out", "Hamstring"),
            ("MIN", 7, "p2", "Will Reichard", "K", "Out", "Back"),
            ("MIN", 7, "p3", "Jordan Mason", "RB", "Questionable", "Thumb"),
            ("MIN", 7, "p4", "Healthy Guy", "TE", None, None),
            ("MIN", 6, "p5", "Old Report", "QB", "Out", "Knee")]
    return pd.DataFrame(rows, columns=["team", "week", "gsis_id", "full_name", "position", "report_status",
                                       "report_primary_injury"]).assign(season=2026)


def test_injury_facts_only_for_upcoming_game_and_in_model_for_key_outs():
    got = facts.injury_facts(injuries_frame(), "MIN", 2026, upcoming_week=7)
    assert [f["text"] for f in got] == ["Justin Jefferson (WR) is Out (Hamstring) for Week 7",
                                        "Will Reichard (K) is Out (Back) for Week 7",
                                        "Jordan Mason (RB) is Questionable (Thumb) for Week 7"]
    assert [f["in_model"] for f in got] == [True, False, False]  # kickers aren't key starters; Questionable isn't counted
    assert facts.injury_facts(injuries_frame(), "MIN", 2026, upcoming_week=8) == []  # Week 7 report is stale


def test_roster_moves_ignore_gameday_inactives():
    def roster(week, rows):
        return pd.DataFrame(rows, columns=["gsis_id", "full_name", "position", "status"]).assign(
            team="MIN", season=2026, week=week)
    rosters = pd.concat([
        roster(5, [("a", "Starter", "WR", "ACT"), ("b", "Backup", "DB", "INA"), ("c", "Hurt", "RB", "RES"),
                   ("d", "Leaving", "OL", "ACT"), ("e", "PS Guy", "WR", "DEV")]),
        roster(7, [("a", "Starter", "WR", "INA"), ("b", "Backup", "DB", "ACT"), ("c", "Hurt", "RB", "ACT"),
                   ("d", "Leaving", "OL", "CUT"), ("e", "PS Guy", "WR", "ACT"), ("f", "New Guy", "LB", "DEV")]),
    ])
    texts = [f["text"].split(" (Week")[0] for f in facts.roster_facts(rosters, "MIN", 2026)]
    assert set(texts) == {"Activated Hurt (RB) from the reserve list", "Released Leaving (OL)",
                          "Promoted PS Guy (WR) from the practice squad", "Added New Guy (LB) to the practice squad"}
    assert texts[-1] == "Added New Guy (LB) to the practice squad"  # non-active moves sort last


def test_trade_and_coach_facts():
    trades = pd.DataFrame([
        (1, 2026, "2026-10-20", "DET", "MIN", "Edge Rusher", None, None),
        (1, 2026, "2026-10-20", "MIN", "DET", "", 2027, 3),
        (2, 2026, "2026-08-01", "MIN", "GB", "Old Trade", None, None),
    ], columns=["trade_id", "season", "trade_date", "gave", "received", "pfr_name", "pick_season", "pick_round"])
    (fact,) = facts.trade_facts(trades, "MIN", 2026, now=NOW)
    assert fact["text"] == ("Trade with the Detroit Lions on 2026-10-20: received Edge Rusher; "
                            "sent 2027 round 3 pick")

    sched = pd.DataFrame([
        (2026, 1, "2026-09-13", "MIN", "GB", 1.0, "Coach A", "X"),
        (2026, 2, "2026-09-20", "CHI", "MIN", -3.0, "Y", "Coach A"),
        (2026, 3, "2026-09-27", "MIN", "TB", 7.0, "Coach B", "Z"),
        (2026, 4, "2026-10-04", "MIN", "MIA", None, None, None),
    ], columns=["season", "week", "gameday", "home_team", "away_team", "result", "home_coach", "away_coach"])
    assert facts.coach_facts(sched, "MIN", 2026)[0]["text"].startswith("Head coach change: Coach A → Coach B")
    assert facts.upcoming_week(sched, "MIN", 2026) == 4


# --- Haiku tags -------------------------------------------------------------------------------

FACTS = [{"id": "injury:p1:7", "kind": "Injury", "text": "Justin Jefferson (WR) is Out", "in_model": True},
         {"id": "roster:7:0", "kind": "Roster move", "text": "Released Leaving (OL)", "in_model": False}]


def test_tag_facts_keeps_only_known_ids_and_valid_impacts():
    payload = {"tags": [{"id": "injury:p1:7", "impact": "hurts", "note": "Top receiver out. "},
                        {"id": "made-up", "impact": "helps", "note": "invented"},
                        {"id": "roster:7:0", "impact": "huge", "note": "bad enum"}]}
    client = client_returning([NS(type="text", text=json.dumps(payload))])
    tags = tag_facts(client, "claude-haiku-5-5", "MIN", "DET", FACTS)
    assert tags == {"injury:p1:7": {"impact": "hurts", "note": "Top receiver out."}}
    sent = client.messages.calls[0]["messages"][0]["content"]
    assert "Minnesota Vikings" in sent and "Detroit Lions" in sent and "headline" not in sent.lower()


def test_tag_facts_skips_call_without_facts():
    client = client_returning([])
    assert tag_facts(client, "m", "MIN", "DET", []) == {} and client.messages.calls == []


# --- API + MCP -----------------------------------------------------------------------------------

@pytest.fixture
def news_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(collect, "NEWS_DIR", tmp_path)
    (tmp_path / "MIN.json").write_text(json.dumps({
        "team": "MIN", "season": 2026, "generated_at": "2026-10-25T22:00:00+00:00", "facts": FACTS,
        "headlines": [{"title": "Vikings place Darrisaw on IR", "link": "https://a.example/1", "source": "ESPN",
                       "published": "2026-10-24T15:00:00+00:00", "category": "Injury", "team": "MIN"}],
        "sources": [{"name": "ESPN", "url": "https://e.example", "terms": "t"}]}))
    return tmp_path


def test_news_endpoint(news_dir):
    client = TestClient(api.app)
    body = client.get("/teams/min/news").json()
    assert body["facts"][0]["in_model"] is True and body["headlines"][0]["category"] == "Injury"
    assert client.get("/teams/GB/news").status_code == 404   # not collected
    assert client.get("/teams/XYZ/news").status_code == 404  # unknown team


def test_mcp_news_tool_never_returns_headlines(news_dir, monkeypatch):
    monkeypatch.setattr(server, "_client", lambda: TestClient(api.app))
    result = server.get_team_news("MIN")
    assert result["facts"] == [{**f, "date": None} for f in FACTS]
    assert "headlines" not in result and "Darrisaw" not in json.dumps(result)
    assert "1 recent headlines" in result["note"]
