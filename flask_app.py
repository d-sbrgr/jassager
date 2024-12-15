import logging

from jass.service.player_service_app import PlayerServiceApp
from bots import FullMCTS, RandomAgent, MCTSCNNRollout

logging.basicConfig(level=logging.DEBUG)
app = PlayerServiceApp('flask_app')
app.add_player('jassager', FullMCTS())
app.add_player('random', RandomAgent())
app.add_player('jassager_cnn', MCTSCNNRollout())


if __name__ == '__main__':
   app.run(host="0.0.0.0", port=8080)

