"""Where weekly predictions live.

The API and the job only talk to the `PredictionStore` interface, so moving
from local Parquet to S3/DynamoDB later (Phase 7) means adding one class.
"""

from pathlib import Path
from typing import Protocol

import pandas as pd

from vikpred.config import DATA_DIR

PREDICTIONS_DIR = DATA_DIR / "predictions"


class PredictionStore(Protocol):
    def load(self, season: int) -> pd.DataFrame: ...
    def save(self, season: int, df: pd.DataFrame) -> None: ...
    def seasons(self) -> list[int]: ...


class LocalParquetStore:
    """One Parquet file per season under data/predictions/."""

    def __init__(self, root: Path = PREDICTIONS_DIR):
        self.root = Path(root)

    def _path(self, season: int) -> Path:
        return self.root / f"{season}.parquet"

    def load(self, season: int) -> pd.DataFrame:
        path = self._path(season)
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    def save(self, season: int, df: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        # Write to a temp file first so a crash never leaves a half-written season.
        tmp = self._path(season).with_suffix(".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(self._path(season))

    def seasons(self) -> list[int]:
        return sorted(int(p.stem) for p in self.root.glob("*.parquet"))
