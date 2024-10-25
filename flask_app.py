import os
import logging

from jass.service.player_service_app import PlayerServiceApp
from bots import FullMCTS


def create_app():
    """
    This is the factory method for flask. It is automatically detected when flask is run, but we must tell flask
    what python file to use:

        export FLASK_APP=player_service.py
        export FLASK_ENV=development
        flask run --host=0.0.0.0 --port=8888
    """
    logging.basicConfig(level=logging.DEBUG)

    # create and configure the app
    app = PlayerServiceApp('flask_app')

    # add some players
    app.add_player('jassager', FullMCTS())

    return app


if __name__ == '__main__':
   app = create_app()
   app.run(host="0.0.0.0", port=8080)

