"""News feeds the app reads, checked on 2026-10-25 for availability and terms.

Every feed is displayed as published (headline + link + source credit), fetched
at most once a day, and never sent to an AI model: ESPN's (Disney) and PMC's
terms prohibit using their content with AI tools, so none of the feeds are.
Only nflverse data (open, already used for the models) goes to Claude.

Excluded after review:
- ProFootballTalk / NBC Sports: terms prohibit robots that aggregate content.
- CBS Sports NFL: feed carries sportsbook promotions (this app isn't for betting).
- ESPN's per-team feed (`team=min`): returns general headlines, not the team's.
- NFL.com: no RSS feed.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    team: str | None  # set for team-specific feeds; None for national feeds filtered by team name
    terms: str        # what the publisher's terms require, in short


SOURCES = [
    Source("Vikings.com", "https://www.vikings.com/rss/news", "MIN",
           "Official team feed ('Modular Content'): show unedited, keep Vikings branding, "
           "remove within 24h if asked."),
    Source("Daily Norseman", "https://www.dailynorseman.com/rss/index.xml", "MIN",
           "PMC content feed: use as provided, don't modify; no AI use."),
    Source("ESPN", "https://www.espn.com/espn/rss/nfl/news", None,
           "Show feed content unmodified, credit ESPN, link to the full article, no ads; no AI use."),
    Source("Yahoo Sports", "https://sports.yahoo.com/nfl/rss/", None,
           "Show feed content unmodified with credit and a link to the full article, no ads."),
]

USER_AGENT = "vikings-win-predictor/0.1 (personal, non-commercial RSS reader; daily)"
