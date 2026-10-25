"""Feature tests on a tiny synthetic league (no network, no data files)."""

import numpy as np
import pandas as pd
import pytest

from vikpred.features import build


def synthetic_league(n_weeks: int = 6) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """MIN and GB play each other every week, alternating home field."""
    rng = np.random.default_rng(0)
    sched, epa = [], []
    for w in range(1, n_weeks + 1):
        home, away = ("MIN", "GB") if w % 2 else ("GB", "MIN")
        gid = f"2024_{w:02d}_{away}_{home}"
        result = int(rng.integers(-14, 15)) or 3
        sched.append({
            "game_id": gid, "season": 2024, "week": w, "game_type": "REG",
            "gameday": pd.Timestamp("2024-09-01") + pd.Timedelta(weeks=w),
            "home_team": home, "away_team": away, "home_score": 20 + result, "away_score": 20,
            "result": result, "home_win": float(result > 0), "home_rest": 7, "away_rest": 7,
            "div_game": 1, "roof": "outdoors", "temp": 50.0, "wind": 5.0,
            "spread_line": 3.0, "total_line": 44.0, "home_moneyline": -150, "away_moneyline": 130,
            "home_qb_id": f"{home}_qb1", "away_qb_id": f"{away}_qb1" if w < 4 else f"{away}_qb2",
        })
        for team, opp in [(home, away), (away, home)]:
            epa.append({
                "game_id": gid, "team": team, "opponent": opp,
                "epa_per_play": rng.normal(), "success_rate": rng.uniform(),
                "pass_epa_per_play": rng.normal(), "rush_epa_per_play": rng.normal(),
            })
    inj = pd.DataFrame([
        {"season": 2024, "week": 3, "team": "MIN", "position": "QB", "gsis_id": "q", "report_status": "Out"},
        {"season": 2024, "week": 3, "team": "MIN", "position": "WR", "gsis_id": "w", "report_status": "Out"},
        {"season": 2024, "week": 3, "team": "MIN", "position": "K", "gsis_id": "k", "report_status": "Out"},
        {"season": 2024, "week": 3, "team": "GB", "position": "CB", "gsis_id": "c", "report_status": "Questionable"},
    ])
    return pd.DataFrame(sched), pd.DataFrame(epa), inj


def test_one_row_per_game():
    sched, epa, inj = synthetic_league()
    feats = build.build_features(sched, epa, inj)
    assert len(feats) == len(sched)
    assert feats["game_id"].is_unique


def test_game_stats_never_leak_into_own_features():
    """Changing a game's results must not change that game's features, only later ones."""
    sched, epa, inj = synthetic_league()
    before = build.build_features(sched, epa, inj).set_index("game_id")

    target = sched.loc[3, "game_id"]
    epa2 = epa.copy()
    epa2.loc[epa2.game_id == target, ["epa_per_play", "success_rate"]] = 99.0
    sched2 = sched.copy()
    sched2.loc[sched2.game_id == target, "result"] = 99
    after = build.build_features(sched2, epa2, inj).set_index("game_id")

    cols = build.DIFF_FEATURES
    earlier_and_target = sched.loc[:3, "game_id"]
    later = sched.loc[4:, "game_id"]
    pd.testing.assert_frame_equal(before.loc[earlier_and_target, cols], after.loc[earlier_and_target, cols])
    assert not before.loc[later, "diff_off_epa_ewm"].equals(after.loc[later, "diff_off_epa_ewm"])


def test_first_game_has_no_history():
    sched, epa, inj = synthetic_league()
    feats = build.build_features(sched, epa, inj)
    assert feats.loc[0, "diff_off_epa_ewm"] != feats.loc[0, "diff_off_epa_ewm"]  # NaN


def test_qb_change_flagged_once():
    sched, epa, inj = synthetic_league()
    tg = build.add_rolling(build.team_game_stats(sched, epa))
    gb = tg[tg.team == "GB"].sort_values("gameday")
    # GB switches from qb1 to qb2 when it is away in week 4 or later; week 5 is GB's first away game after that.
    assert gb["qb_changed"].tolist() == [0, 0, 0, 0, 1, 1]


def test_injury_counts_key_positions_only():
    _, _, inj = synthetic_league()
    counts = build.injury_counts(inj).set_index("team")
    assert counts.loc["MIN", "key_players_out"] == 2  # QB + WR; kicker excluded
    assert counts.loc["MIN", "qb_out"] == 1
    assert "GB" not in counts.index  # Questionable doesn't count


@pytest.mark.parametrize("home_ml, away_ml, expected", [(-110, -110, 0.5), (-150, 130, 0.580), (200, -250, 0.318)])
def test_devig_home_prob(home_ml, away_ml, expected):
    prob = build.devig_home_prob(pd.Series([home_ml]), pd.Series([away_ml]))[0]
    assert prob == pytest.approx(expected, abs=1e-3)
