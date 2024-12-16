import logging

from jass.service.player_service_app import PlayerServiceApp
from bots import FullMCTS, RandomAgent, MCTSCNNRollout

logging.basicConfig(level=logging.DEBUG)
app = PlayerServiceApp('flask_app')
app.add_player('jassager', FullMCTS())
app.add_player('random', RandomAgent())
app.add_player('jassager_cnn_01', MCTSCNNRollout(c_param=0.1))
app.add_player('jassager_cnn_10', MCTSCNNRollout(c_param=1.0))
app.add_player('jassager_cnn_13', MCTSCNNRollout())


if __name__ == '__main__':
   app.run(host="0.0.0.0", port=8080)

