import pytest
from jass.agents.agent_random_schieber import AgentRandomSchieber
from jass.agents.agent import Agent
from jass.arena.arena import Arena
from jass.game.const import *

from bots import HeuristicTrumpRandomPlay, FullHeuristicEgocentric

np.random.seed(1)


@pytest.fixture
def arena() -> Arena:
    return Arena(nr_games_to_play=1000)


def get_game_results(arena: Arena, team_0_type: type[Agent], team_1_type: type[Agent]):
    arena.set_players(
        team_0_type(),
        team_1_type(),
        team_0_type(),
        team_1_type()
    )
    arena.play_all_games()


def test_heuristic_trump_random_play(arena):
    get_game_results(arena, HeuristicTrumpRandomPlay, AgentRandomSchieber)
    assert 87000 < arena.points_team_0.sum() < 92000
    assert 65000 < arena.points_team_1.sum() < 71000


def test_heuristic_trump_egocentric_play(arena):
    get_game_results(arena, FullHeuristicEgocentric, HeuristicTrumpRandomPlay)
    assert 75000 < arena.points_team_0.sum() < 80000
    assert 75000 < arena.points_team_1.sum() < 80000

