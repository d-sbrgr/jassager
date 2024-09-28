# Ja$$ager - Jass bots

## Installation

Make sure Python 3.12 is installed on your system. Then run the file ``setup_repository.sh`` in a bash shell.
This will create a new Python environment in your current working directory and install all the dependencies.

````shell
sh setup_repository.sh
````

## Creating new bots

All bots are stored in the ``bots`` Python package.

New bots can be implemented by creating a new class that inherits from ``jass.agents.agent.Agent``. 
In this derived class, the methods ``action_trump()`` and ``action_play_card()`` must be implemented.

## Playing games with bots

In the file ``game_arena.py`` games can be simulated against different kinds of bots.
Set the constant variables ``MY_TEAM_AGENT_TYPE`` and ``OPPONENT_TEAM_AGENT_TYPE`` to the references of the 
respective agent types you would like to use.

Example:
````python
from jass.agents.agent_random_schieber import AgentRandomSchieber
from bots.heuristic_bot import HeuristicAgent

MY_TEAM_AGENT_TYPE = HeuristicAgent
OPPONENT_TEAM_AGENT_TYPE = AgentRandomSchieber
````

Afterward, the file can be run and the game results will be displayed in the console.
