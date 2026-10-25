"""Build the pre-game feature table: one row per game, home team's perspective.

Every feature must be knowable before kickoff. Rolling team stats are
exponentially weighted over each team's previous games (across seasons, so
week 1 inherits last year's form) and shifted one game so a game's own stats
never feed its prediction.

Usage:
    python -m vikpred.features.build      # writes data/processed/features.parquet
"""

import logging

import numpy as np
import pandas as pd

from vikpred.config import PROCESSED_DIR
from vikpred.data import load

log = logging.getLogger(__name__)

EWM_SPAN = 8  # ~half a season; weights recent games more heavily

ROLLING_STATS = [
    "off_epa", "def_epa", "off_success", "def_success",
    "off_pass_epa", "def_pass_epa", "off_rush_epa", "def_rush_epa", "point_diff",
]

# Offensive starters plus the defensive front/secondary. Kickers etc. excluded.
KEY_POSITIONS = {"QB", "RB", "WR", "TE", "T", "G", "C", "DE", "DT", "LB", "CB", "S"}

TEAM_FEATURES = [f"{s}_ewm" for s in ROLLING_STATS] + ["qb_changed", "qb_out", "key_players_out"]
GAME_FEATURES = ["rest_diff", "div_game", "indoors", "temp", "wind"]
MARKET_FEATURES = ["spread_line", "total_line", "market_home_prob"]
DIFF_FEATURES = [f"diff_{f}" for f in TEAM_FEATURES]


def team_game_stats(sched: pd.DataFrame, epa: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with offense, defense, and scoring results."""
    off = epa.rename(columns={
        "epa_per_play": "off_epa", "success_rate": "off_success",
        "pass_epa_per_play": "off_pass_epa", "rush_epa_per_play": "off_rush_epa",
    })[["game_id", "team", "off_epa", "off_success", "off_pass_epa", "off_rush_epa"]]
    # A team's defense is its opponent's offense in the same game.
    dfn = epa.rename(columns={
        "opponent": "team_d", "epa_per_play": "def_epa", "success_rate": "def_success",
        "pass_epa_per_play": "def_pass_epa", "rush_epa_per_play": "def_rush_epa",
    })[["game_id", "team_d", "def_epa", "def_success", "def_pass_epa", "def_rush_epa"]]
    dfn = dfn.rename(columns={"team_d": "team"})

    cols = ["game_id", "season", "week", "gameday"]
    home = sched[cols + ["home_team", "result", "home_qb_id"]].rename(
        columns={"home_team": "team", "home_qb_id": "qb_id"}).assign(point_diff=lambda d: d.result)
    away = sched[cols + ["away_team", "result", "away_qb_id"]].rename(
        columns={"away_team": "team", "away_qb_id": "qb_id"}).assign(point_diff=lambda d: -d.result)
    games = pd.concat([home, away]).drop(columns="result")
    return (games.merge(off, on=["game_id", "team"], how="left")
                 .merge(dfn, on=["game_id", "team"], how="left")
                 .sort_values(["team", "gameday", "game_id"])
                 .reset_index(drop=True))


def add_rolling(tg: pd.DataFrame) -> pd.DataFrame:
    """Pre-game EWM of each stat plus a QB-change flag, per team."""
    tg = tg.copy()
    by_team = tg.groupby("team", group_keys=False)
    for stat in ROLLING_STATS:
        # shift(1): the value for game N uses games 1..N-1 only.
        tg[f"{stat}_ewm"] = by_team[stat].transform(
            lambda s: s.shift(1).ewm(span=EWM_SPAN, min_periods=1, ignore_na=True).mean())
    prev_qb = by_team["qb_id"].shift(1)
    tg["qb_changed"] = (prev_qb.notna() & (tg["qb_id"] != prev_qb)).astype(int)
    return tg


def injury_counts(inj: pd.DataFrame) -> pd.DataFrame:
    """Per team-week: was the QB ruled Out, and how many key starters are Out."""
    out = inj[(inj["report_status"] == "Out") & inj["position"].isin(KEY_POSITIONS)]
    return (out.assign(is_qb=out["position"] == "QB")
               .groupby(["season", "week", "team"])
               .agg(key_players_out=("gsis_id", "nunique"), qb_out=("is_qb", "max"))
               .astype(int)
               .reset_index())


def devig_home_prob(home_ml: pd.Series, away_ml: pd.Series) -> pd.Series:
    """Market win probability for the home team with the bookmaker margin removed."""
    def implied(ml):
        return np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))
    h, a = implied(home_ml), implied(away_ml)
    return pd.Series(h / (h + a), index=home_ml.index)


def build_features(sched: pd.DataFrame, epa: pd.DataFrame, inj: pd.DataFrame) -> pd.DataFrame:
    tg = add_rolling(team_game_stats(sched, epa))
    tg = tg.merge(injury_counts(inj), on=["season", "week", "team"], how="left")
    tg[["key_players_out", "qb_out"]] = tg[["key_players_out", "qb_out"]].fillna(0).astype(int)
    team_cols = ["game_id", "team"] + TEAM_FEATURES

    g = sched.copy()
    home = tg[team_cols].rename(columns={c: f"home_{c}" for c in TEAM_FEATURES} | {"team": "home_team"})
    away = tg[team_cols].rename(columns={c: f"away_{c}" for c in TEAM_FEATURES} | {"team": "away_team"})
    g = g.merge(home, on=["game_id", "home_team"]).merge(away, on=["game_id", "away_team"])

    for f in TEAM_FEATURES:
        g[f"diff_{f}"] = g[f"home_{f}"] - g[f"away_{f}"]
    g["rest_diff"] = g["home_rest"] - g["away_rest"]
    g["indoors"] = g["roof"].isin(["dome", "closed"]).astype(int)
    # Weather is missing for indoor games; use neutral values there and the
    # median elsewhere (a handful of outdoor games are also missing).
    g["temp"] = g["temp"].where(g["indoors"] == 0, 70).fillna(g["temp"].median())
    g["wind"] = g["wind"].where(g["indoors"] == 0, 0).fillna(g["wind"].median())
    g["market_home_prob"] = devig_home_prob(g["home_moneyline"], g["away_moneyline"])

    keep = (["game_id", "season", "week", "game_type", "gameday", "home_team", "away_team",
             "home_score", "away_score", "result", "home_win"]
            + DIFF_FEATURES + GAME_FEATURES + MARKET_FEATURES)
    return g[keep].sort_values(["gameday", "game_id"]).reset_index(drop=True)


def run() -> pd.DataFrame:
    feats = build_features(load.schedules(), load.team_game_epa(), load.injuries())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
    log.info("features: %d games x %d columns", *feats.shape)
    return feats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
