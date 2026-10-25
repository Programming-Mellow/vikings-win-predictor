"""Read RSS/Atom feeds and pick out a team's headlines.

Headlines are kept exactly as published. Categories come from plain keyword
rules (not AI), so they're predictable and can be shown next to the headline
without touching it.
"""

import html
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

import httpx
from defusedxml import ElementTree as ET  # feeds are untrusted XML

from vikpred.news.sources import USER_AGENT, Source
from vikpred.teams import TEAM_NAMES, full_name

log = logging.getLogger(__name__)

ATOM = "{http://www.w3.org/2005/Atom}"
DC_DATE = "{http://purl.org/dc/elements/1.1/}date"

# Order matters: the first matching category wins (a "suspended after injury" story is a suspension).
CATEGORIES = [
    ("Suspension", re.compile(r"\bsuspen\w*|\breinstat\w*", re.I)),
    ("Coaching", re.compile(r"\bfire[sd]\b|\bdismiss\w*|\bcoordinator\b|\bhead coach\b|\binterim\b|\bhire[sd]?\b", re.I)),
    ("Trade", re.compile(r"\btrade[sd]?\b|\btrading\b|\bacquir\w*", re.I)),
    ("Injury", re.compile(r"\binjur\w*|\bruled out\b|\bquestionable\b|\bdoubtful\b|\bconcussion\b|\btorn\b|"
                          r"\bsprain\w*|\bhamstring\b|\bACL\b|\bsurgery\b|\bout for\b|\bweek-to-week\b", re.I)),
    ("Roster move", re.compile(r"\bsign(s|ed|ing)?\b|\breleas\w*|\bwaive[sd]?\b|\bcuts?\b|\bclaim(s|ed)?\b|"
                               r"\bpractice squad\b|\bactivat\w*|\bpromot\w*|\bextension\b", re.I)),
]
INJURED_RESERVE = re.compile(r"\bIR\b")  # case-sensitive, so "ir" inside other words never matches
# This app isn't for betting, so betting and sportsbook items are dropped.
BETTING = re.compile(r"\bbet(s|ting)?\b|\bodds\b|\bpicks?\b|\bparlay|\bsportsbook|\bpromo code|"
                     r"\bDraftKings\b|\bFanDuel\b|\bBetMGM\b|\bprops?\b|\bspread\b", re.I)
TEAM_NICKNAMES = {code: name.rsplit(" ", 1)[-1] for code, name in TEAM_NAMES.items()}


@dataclass
class Headline:
    title: str
    link: str
    source: str
    published: str | None  # ISO 8601, UTC
    category: str | None
    team: str | None = None


def _parse_date(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        d = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        try:
            d = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def categorize(text: str) -> str | None:
    """Keyword category for a headline, or None for general news."""
    for name, pattern in CATEGORIES:
        if pattern.search(text) or (name == "Injury" and INJURED_RESERVE.search(text)):
            return name
    return None


def parse_feed(xml: bytes, source: str) -> list[tuple[Headline, str]]:
    """Return (headline, searchable text) pairs from RSS 2.0 or Atom."""
    root = ET.fromstring(xml)
    out = []
    for item in root.findall("./channel/item"):
        # Some feeds double-encode entities (&#39;); decoding renders the title, it doesn't change it.
        title = html.unescape(item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        date = _parse_date(item.findtext("pubDate") or item.findtext(DC_DATE))
        text = f"{title} {item.findtext('description') or ''}"
        out.append((Headline(title, link, source, date.isoformat() if date else None, None), text))
    for entry in root.findall(f"{ATOM}entry"):
        title = html.unescape(entry.findtext(f"{ATOM}title") or "").strip()
        link_el = entry.find(f"{ATOM}link")
        link = (link_el.get("href") if link_el is not None else "") or ""
        date = _parse_date(entry.findtext(f"{ATOM}published") or entry.findtext(f"{ATOM}updated"))
        text = f"{title} {entry.findtext(f'{ATOM}summary') or ''}"
        out.append((Headline(title, link, source, date.isoformat() if date else None, None), text))
    return [(h, t) for h, t in out if h.title and h.link.startswith(("https://", "http://"))]


def fetch(source: Source, client: httpx.Client | None = None) -> list[tuple[Headline, str]]:
    """Download one feed. A broken feed returns [] so one source never breaks the rest."""
    try:
        own = client is None
        client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=15, follow_redirects=True)
        try:
            resp = client.get(source.url)
        finally:
            if own:
                client.close()
        resp.raise_for_status()
        return parse_feed(resp.content, source.name)
    except Exception as exc:  # network errors, bad XML, unexpected formats
        log.warning("feed %s failed: %s", source.name, exc)
        return []


def mentions_team(text: str, team: str) -> bool:
    """True if the text names the team (full name or nickname, e.g. 'Vikings')."""
    nickname = TEAM_NICKNAMES.get(team, team)
    return bool(re.search(rf"\b({re.escape(full_name(team))}|{re.escape(nickname)})\b", text, re.IGNORECASE))


def team_headlines(feeds: dict[Source, list[tuple[Headline, str]]], team: str, days: int = 7,
                   now: datetime | None = None, limit: int = 10) -> list[dict]:
    """Recent headlines for one team: everything from its own feeds, plus national items naming it."""
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    picked, seen = [], set()
    # Team-specific feeds first, so a story syndicated to a national feed keeps its original source.
    for source, items in sorted(feeds.items(), key=lambda kv: kv[0].team is None):
        if source.team not in (None, team):
            continue
        for headline, text in items:
            published = _parse_date(headline.published)
            # The same story often appears in several feeds under different links.
            title_key = re.sub(r"[^a-z0-9]", "", headline.title.lower())
            if published is None or published < cutoff or headline.link in seen or title_key in seen:
                continue
            if source.team is None and not mentions_team(text, team):
                continue
            if BETTING.search(headline.title):
                continue
            seen.update({headline.link, title_key})
            picked.append(Headline(**{**asdict(headline), "category": categorize(text), "team": team}))
    picked.sort(key=lambda h: h.published or "", reverse=True)
    return [asdict(h) for h in picked[:limit]]
