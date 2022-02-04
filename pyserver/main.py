"""Flask Python Server for IOTA Simulation

Server for communicating with Reactjs frontend. Server primarily calls ./conductTesting.py

"""

import argparse
import sys
import json
from flask_cors import CORS
from flask import Flask, request, jsonify
from conductTesting import *
import numpy as np

sys.path.append('./tangleEnv')
sys.path.append('./intelligent_tip')
app = Flask(__name__)
CORS(app)
configFile = './pyserver/configs/server_config.yaml'
env = []


def setup_parser():
    """
    Initializes the argument parser for conducting repeatable tests.
    :return: Argument parser with server control parameters
    """
    conf_parser = argparse.ArgumentParser()
    conf_parser.add_argument("--runLocal", default=False,type=bool)
    conf_parser.add_argument("--logData", default=False,type=bool)
    conf_parser.add_argument("--dispData", default=False,type=bool)
    return conf_parser


@app.route('/createTangle', methods=['POST'])
def create_tangle():
    """
    Creates a Tangle object with a specified tip selection algorithm sent by the frontend.
    :return: tangleData - Dict. (JSON) of Tangle vertices and edges.
    """
    global env, configFile
    np.random.seed(0)
    content = request.json
    content = json.loads(content)
    env = gen_tangle_client(content, configFile)
    return jsonify(env.get_tangle())


def shutdown_server():
    """
    Shuts down the server. Also saves the figures from the generated tests.
    :return:
    """
    # Shutdown server for cleanup
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown', methods=['GET'])
def shutdown():
    """
    API call to shutdown the server.
    :return:
    """
    shutdown_server()
    return 'Server shutting down...'


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    if not args.runLocal:
        # Provide computation to React frontend
        app.run(debug=False)
    else:
        # Generate a local tangle with each tip selection algorithm and output barchart
        gen_tangle_local('./pyserver/configs/server_config.yaml')
        local_metrics()
