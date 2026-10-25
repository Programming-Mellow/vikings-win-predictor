"""Season tracker: how each pre-game prediction compared with what actually happened."""

import pandas as pd


def tracker_frame(games: list[dict], live_only: bool = False) -> pd.DataFrame:
    """Played games with both the model's and Vegas's call, plus running Brier scores.

    `games` is the /teams/{team}/games payload. Ties and games without a
    stats + Vegas prediction are left out, so model and Vegas are scored on the
    same games. With `live_only`, backfilled predictions are excluded too.
    """
    df = pd.DataFrame(games)
    cols = ["week", "opponent", "venue", "result", "score", "model_prob", "vegas_prob", "won", "live",
            "model_brier", "vegas_brier", "model_correct", "vegas_correct",
            "cum_model_brier", "cum_vegas_brier"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df = df[df.result.isin(["W", "L"]) & df.win_prob.notna() & df.vegas_prob.notna()].copy()
    if live_only:
        df = df[~df.backfilled.astype(bool)]
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = df.sort_values("week").rename(columns={"win_prob": "model_prob"})
    df["won"] = (df.result == "W").astype(int)
    df["live"] = ~df.backfilled.astype(bool)
    for who in ("model", "vegas"):
        p = df[f"{who}_prob"]
        df[f"{who}_brier"] = (p - df.won) ** 2
        df[f"{who}_correct"] = (p > 0.5) == (df.won == 1)
        df[f"cum_{who}_brier"] = df[f"{who}_brier"].expanding().mean()
    return df[cols].reset_index(drop=True)


def summary(frame: pd.DataFrame) -> dict:
    """Headline numbers for the tracker."""
    if frame.empty:
        return {"games": 0}
    return {
        "games": len(frame),
        "model_correct": int(frame.model_correct.sum()),
        "vegas_correct": int(frame.vegas_correct.sum()),
        "model_brier": round(float(frame.model_brier.mean()), 4),
        "vegas_brier": round(float(frame.vegas_brier.mean()), 4),
    }
