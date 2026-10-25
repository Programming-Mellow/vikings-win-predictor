"""This week's team facts from nflverse: injuries, roster moves, trades, coach changes.

`in_model` is computed here, not by an LLM: the models' features count key
starters and QBs ruled Out on the injury report, and nothing else in this list.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from vikpred.features.build import KEY_POSITIONS
from vikpred.teams import full_name

REPORT_STATUSES = ["Out", "Doubtful", "Questionable"]
STATUS_LABELS = {
    "ACT": "active roster", "DEV": "practice squad", "RES": "reserve list (IR, PUP or suspension)",
    "INA": "inactive list", "CUT": "released", "RET": "retired", "TRT": "transferred",
}


def _fact(id_: str, kind: str, text: str, in_model: bool, date: str | None = None) -> dict:
    return {"id": id_, "kind": kind, "text": text, "in_model": in_model, "date": date}


def injury_facts(injuries: pd.DataFrame, team: str, season: int, upcoming_week: int | None) -> list[dict]:
    """Players with a game designation on the injury report for the team's upcoming game.

    Reports for games already played are stale, so nothing shows between the
    final whistle and the next week's first report (usually Wednesday).
    """
    df = injuries[(injuries.team == team) & (injuries.season == season) & injuries.report_status.isin(REPORT_STATUSES)]
    if df.empty or upcoming_week is None or int(df.week.max()) < upcoming_week:
        return []
    week = int(df.week.max())
    df = df[df.week == week].copy()
    df["order"] = df.report_status.map({s: i for i, s in enumerate(REPORT_STATUSES)})
    facts = []
    for r in df.sort_values(["order", "full_name"]).itertuples():
        injury = f" ({r.report_primary_injury})" if isinstance(r.report_primary_injury, str) else ""
        counted = r.report_status == "Out" and r.position in KEY_POSITIONS
        facts.append(_fact(f"injury:{r.gsis_id}:{week}", "Injury",
                           f"{r.full_name} ({r.position}) is {r.report_status}{injury} for Week {week}", counted))
    return facts


def roster_facts(rosters: pd.DataFrame, team: str, season: int, limit: int = 8) -> list[dict]:
    """Moves between the team's two most recent weekly rosters."""
    df = rosters[(rosters.team == team) & (rosters.season == season)]
    weeks = sorted(df.week.unique())
    if len(weeks) < 2:
        return []
    prev_w, last_w = weeks[-2], weeks[-1]
    # Game-day inactives are still on the 53-man roster, so INA counts as ACT here;
    # otherwise every week's inactive list would look like a pile of roster moves.
    df = df.assign(status=df.status.replace({"INA": "ACT"}))
    prev = df[df.week == prev_w].set_index("gsis_id")
    last = df[df.week == last_w].set_index("gsis_id")
    facts = []
    for pid, r in last.iterrows():
        name = f"{r.full_name} ({r.position})"
        if pid not in prev.index:
            if r.status != "CUT":
                facts.append((r.status == "ACT", f"Added {name} to the {STATUS_LABELS.get(r.status, r.status)}"))
            continue
        before = prev.loc[pid].status
        if before == r.status:
            continue
        if r.status == "CUT":
            text = f"Released {name}"
        elif before == "DEV" and r.status == "ACT":
            text = f"Promoted {name} from the practice squad"
        elif before == "RES" and r.status == "ACT":
            text = f"Activated {name} from the reserve list"
        else:
            text = f"Moved {name} from the {STATUS_LABELS.get(before, before)} to the {STATUS_LABELS.get(r.status, r.status)}"
        facts.append(("ACT" in (before, r.status), text))
    for pid, r in prev.iterrows():
        if pid not in last.index and r.status != "CUT":
            facts.append((r.status == "ACT", f"{r.full_name} ({r.position}) is no longer on the roster"))
    # Moves touching the active roster matter most for a game, so they come first.
    facts.sort(key=lambda f: not f[0])
    return [_fact(f"roster:{last_w}:{i}", "Roster move", f"{text} (Week {prev_w} → {last_w} rosters)", False)
            for i, (_, text) in enumerate(facts[:limit])]


def trade_facts(trades: pd.DataFrame, team: str, season: int, now: datetime | None = None,
                days: int = 21) -> list[dict]:
    """Trades involving the team in the last few weeks."""
    now = now or datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=days)).date().isoformat()
    df = trades[(trades.season == season) & ((trades.gave == team) | (trades.received == team))
                & (trades.trade_date.astype(str) >= cutoff)]
    facts = []
    for trade_id, g in df.groupby("trade_id"):
        other = next(t for t in pd.concat([g.gave, g.received]) if t != team)

        def assets(rows):
            names = [r.pfr_name if isinstance(r.pfr_name, str) and r.pfr_name.strip()
                     else f"{int(r.pick_season)} round {int(r.pick_round)} pick" if pd.notna(r.pick_round)
                     else "an asset" for r in rows.itertuples()]
            return ", ".join(names) or "nothing listed"

        got, sent = assets(g[g.received == team]), assets(g[g.gave == team])
        date = str(g.trade_date.iloc[0])
        facts.append(_fact(f"trade:{trade_id}", "Trade",
                           f"Trade with the {full_name(other)} on {date}: received {got}; sent {sent}", False, date))
    return facts


def coach_facts(schedules: pd.DataFrame, team: str, season: int) -> list[dict]:
    """A head-coach change between the team's last two played games."""
    df = schedules[(schedules.season == season) & schedules.result.notna()
                   & ((schedules.home_team == team) | (schedules.away_team == team))].sort_values("gameday")
    coaches = [r.home_coach if r.home_team == team else r.away_coach for r in df.itertuples()]
    coaches = [c for c in coaches if isinstance(c, str)]
    if len(coaches) >= 2 and coaches[-1] != coaches[-2]:
        return [_fact(f"coach:{coaches[-1]}", "Coaching",
                      f"Head coach change: {coaches[-2]} → {coaches[-1]} (per the last two games)", False)]
    return []


def upcoming_week(schedules: pd.DataFrame, team: str, season: int) -> int | None:
    df = schedules[(schedules.season == season) & schedules.result.isna()
                   & ((schedules.home_team == team) | (schedules.away_team == team))]
    return int(df.week.min()) if not df.empty else None


def team_facts(team: str, season: int, injuries: pd.DataFrame, rosters: pd.DataFrame,
               trades: pd.DataFrame, schedules: pd.DataFrame, now: datetime | None = None) -> list[dict]:
    return (injury_facts(injuries, team, season, upcoming_week(schedules, team, season))
            + coach_facts(schedules, team, season)
            + trade_facts(trades, team, season, now) + roster_facts(rosters, team, season))
