from jass.arena.arena import Arena
from jass.agents.agent_random_schieber import AgentRandomSchieber

from bots import HeuristicTrumpRandomPlay, FullHeuristicEgocentric, FullHeuristicTableView, HeuristicTrumpMCTSPlay

MY_TEAM_AGENT_TYPE = HeuristicTrumpMCTSPlay
OPPONENT_TEAM_AGENT_TYPE = FullHeuristicTableView

arena = Arena(nr_games_to_play=1)
arena.set_players(
    MY_TEAM_AGENT_TYPE(),
    OPPONENT_TEAM_AGENT_TYPE(),
    MY_TEAM_AGENT_TYPE(),
    OPPONENT_TEAM_AGENT_TYPE()
)

print("DET: AVG |    MAX   |    MIN   ||--|| ALG: AVG |    MAX   |    MIN")

arena.play_all_games()

print(
    f"My team points: {arena.points_team_0.sum()}\n"
    f"Opponent team points: {arena.points_team_1.sum()}"
)
