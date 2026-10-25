"""Project-wide settings: paths, seasons, and the team we care about."""

import os
from pathlib import Path

import nflreadpy as nfl

# In a source checkout the project root is two levels above this package. An
# installed package (e.g. in Docker) can't find it that way, so VIKPRED_HOME overrides it.
PROJECT_ROOT = Path(os.environ.get("VIKPRED_HOME") or Path(__file__).resolve().parents[2])
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

TEAM = "MIN"
FIRST_SEASON = 2010


def current_season() -> int:
    return nfl.get_current_season()


# Schedules keep pre-relocation codes; play-by-play uses the current ones.
TEAM_ALIASES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}
