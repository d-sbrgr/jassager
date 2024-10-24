import pytest
import numpy as np
from jass.arena.arena import Arena

from bots import (
    RandomAgent,
    HeuristicTrumpRandomPlay,
    FullHeuristicEgocentric,
    FullHeuristicTableView,
    HeuristicTrumpMCTSPlay,
    FullMCTS
)


@pytest.fixture
def arena():
    return Arena(nr_games_to_play=10)


@pytest.mark.parametrize("bot,opponent,bot_points,opponent_points", (
        (HeuristicTrumpRandomPlay, RandomAgent, 903, 667),
        (FullHeuristicEgocentric, RandomAgent, 963, 607),
        (FullHeuristicTableView, RandomAgent, 931, 639),
        (HeuristicTrumpMCTSPlay, RandomAgent, 1055, 515),
        (FullMCTS, RandomAgent, 1070, 500),
))
def test_performance(arena, bot, opponent, bot_points, opponent_points):
    np.random.seed(0x666)
    arena.set_players(
        bot(),
        opponent(),
        bot(),
        opponent()
    )
    arena.play_all_games()
    assert int(arena.points_team_0.sum()) == bot_points
    assert int(arena.points_team_1.sum()) == opponent_points
