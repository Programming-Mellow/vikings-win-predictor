import pytest

from vikpred.teams import TEAM_NAMES, full_name, prediction_headline


def test_all_32_teams_have_full_names():
    assert len(TEAM_NAMES) == 32
    assert full_name("LV") == "Las Vegas Raiders"
    assert full_name("XYZ") == "XYZ"  # unknown codes fall back to the code


def test_likely_win():
    text, tone = prediction_headline("MIN", "IND", 0.7426)
    assert tone == "win"
    assert text == "The Minnesota Vikings are most likely going to win against the Indianapolis Colts, with a 74% chance."


def test_likely_loss_uses_the_teams_own_win_chance():
    text, tone = prediction_headline("GB", "LV", 0.38)
    assert tone == "loss"
    assert text == ("The Green Bay Packers are most likely not going to win against the Las Vegas Raiders, "
                    "with only a 38% chance.")


@pytest.mark.parametrize("prob", [0.5, 0.496, 0.504])
def test_rounds_to_toss_up_consistently_with_the_shown_percent(prob):
    text, tone = prediction_headline("KC", "SEA", prob)
    assert tone == "tossup" and "50%" in text
