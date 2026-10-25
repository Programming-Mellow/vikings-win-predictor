"""Pull NFL data from nflverse and save it as Parquet under data/raw/.

Outputs:
    schedules.parquet      one row per game (scores, Vegas lines, rest, weather)
    team_game_epa.parquet  one row per team per game (offensive EPA/success rates)
    injuries.parquet       weekly injury reports

Play-by-play is ~100 MB per season in memory, so each season is aggregated to
team-game level as soon as it's downloaded and the raw plays are not kept.

Usage:
    python -m vikpred.data.ingest                 # FIRST_SEASON..current season
    python -m vikpred.data.ingest --start 2018    # a narrower range
"""

import argparse
import logging

import nflreadpy as nfl
import polars as pl

from vikpred.config import FIRST_SEASON, RAW_DIR, TEAM_ALIASES, current_season

log = logging.getLogger(__name__)

SCHEDULE_COLUMNS = [
    "game_id", "season", "game_type", "week", "gameday", "weekday", "gametime",
    "away_team", "away_score", "home_team", "home_score", "location", "result",
    "total", "overtime", "away_rest", "home_rest", "away_moneyline",
    "home_moneyline", "spread_line", "total_line", "div_game", "roof",
    "surface", "temp", "wind", "away_qb_id", "home_qb_id", "away_qb_name",
    "home_qb_name", "away_coach", "home_coach", "stadium_id",
]


def normalize_teams(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Map relocated franchises to their current abbreviation."""
    return df.with_columns(
        pl.col(c).replace(TEAM_ALIASES) for c in columns if c in df.columns
    )


def clean_schedules(raw: pl.DataFrame, seasons: list[int]) -> pl.DataFrame:
    """Keep the useful columns, current team codes, and a home_win label."""
    df = raw.filter(pl.col("season").is_in(seasons)).select(SCHEDULE_COLUMNS)
    df = normalize_teams(df, ["home_team", "away_team"])
    return df.with_columns(
        pl.col("gameday").str.to_date(),
        # Null until the game is played; ties count as 0.5.
        pl.when(pl.col("result").is_null()).then(None)
        .when(pl.col("result") > 0).then(1.0)
        .when(pl.col("result") < 0).then(0.0)
        .otherwise(0.5)
        .alias("home_win"),
    ).sort(["season", "week", "game_id"])


def aggregate_pbp(pbp: pl.DataFrame) -> pl.DataFrame:
    """Collapse play-by-play to one row per offense per game.

    Only dropbacks and designed runs count (no kicks, penalties, or kneels
    without a play), matching the usual EPA/play convention.
    """
    plays = pbp.filter(
        ((pl.col("pass") == 1) | (pl.col("rush") == 1))
        & pl.col("epa").is_not_null()
        & pl.col("posteam").is_not_null()
    )
    plays = normalize_teams(plays, ["posteam", "defteam"])
    is_pass = pl.col("pass") == 1
    is_rush = pl.col("rush") == 1
    return (
        plays.group_by(["game_id", "season", "week", "season_type", "posteam", "defteam"])
        .agg(
            pl.len().alias("plays"),
            pl.col("epa").mean().alias("epa_per_play"),
            pl.col("success").mean().alias("success_rate"),
            is_pass.sum().alias("pass_plays"),
            pl.col("epa").filter(is_pass).mean().alias("pass_epa_per_play"),
            pl.col("epa").filter(is_rush).mean().alias("rush_epa_per_play"),
            pl.col("success").filter(is_pass).mean().alias("pass_success_rate"),
            pl.col("success").filter(is_rush).mean().alias("rush_success_rate"),
        )
        .rename({"posteam": "team", "defteam": "opponent"})
        .sort(["season", "week", "game_id", "team"])
    )


def ingest_schedules(seasons: list[int]) -> pl.DataFrame:
    df = clean_schedules(nfl.load_schedules(seasons), seasons)
    df.write_parquet(RAW_DIR / "schedules.parquet")
    log.info("schedules: %d games", df.height)
    return df


def ingest_team_game_epa(seasons: list[int], refresh: bool = False) -> pl.DataFrame:
    """Aggregate play-by-play one season at a time, caching finished seasons."""
    cache_dir = RAW_DIR / "epa_by_season"
    cache_dir.mkdir(parents=True, exist_ok=True)
    latest = current_season()
    frames = []
    for season in seasons:
        path = cache_dir / f"{season}.parquet"
        # The in-progress season gets new games every week, so always re-pull it.
        if path.exists() and not refresh and season != latest:
            frames.append(pl.read_parquet(path))
            continue
        log.info("pbp %d: downloading", season)
        agg = aggregate_pbp(nfl.load_pbp(season))
        agg.write_parquet(path)
        frames.append(agg)
    df = pl.concat(frames)
    df.write_parquet(RAW_DIR / "team_game_epa.parquet")
    log.info("team_game_epa: %d team-games", df.height)
    return df


def ingest_injuries(seasons: list[int]) -> pl.DataFrame | None:
    # Injury reports for the current season are sometimes published late.
    frames = []
    for season in seasons:
        try:
            frames.append(nfl.load_injuries(season))
        except Exception as exc:
            log.warning("injuries %d unavailable: %s", season, exc)
    if not frames:
        return None
    df = normalize_teams(pl.concat(frames, how="diagonal_relaxed"), ["team"])
    df = df.with_columns(pl.col("season", "week").cast(pl.Int32))
    df.write_parquet(RAW_DIR / "injuries.parquet")
    log.info("injuries: %d rows", df.height)
    return df


def run(start: int = FIRST_SEASON, end: int | None = None, refresh: bool = False) -> None:
    end = end or current_season()
    seasons = list(range(start, end + 1))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    log.info("ingesting seasons %d-%d into %s", start, end, RAW_DIR)
    ingest_schedules(seasons)
    ingest_team_game_epa(seasons, refresh=refresh)
    ingest_injuries(seasons)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=int, default=FIRST_SEASON)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--refresh", action="store_true", help="re-download cached seasons")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.start, args.end, args.refresh)


if __name__ == "__main__":
    main()
