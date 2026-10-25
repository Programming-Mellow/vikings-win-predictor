"""Team display names and the plain-English prediction headline."""

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def full_name(team: str) -> str:
    return TEAM_NAMES.get(team, team)


def prediction_headline(team: str, opponent: str, win_prob: float) -> tuple[str, str]:
    """One-sentence verdict from `team`'s side, plus its tone: "win", "loss", or "tossup".

    The percentage is always `team`'s own win chance, so the number means the
    same thing whichever way the verdict goes.
    """
    pct = round(win_prob * 100)
    us, them = full_name(team), f"the {full_name(opponent)}"
    if pct == 50:
        return f"The {us} vs. {them} is a toss-up: a 50% chance either way.", "tossup"
    if pct > 50:
        return f"The {us} are most likely going to win against {them}, with a {pct}% chance.", "win"
    return f"The {us} are most likely not going to win against {them}, with only a {pct}% chance.", "loss"
