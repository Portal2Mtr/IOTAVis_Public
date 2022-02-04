""" URTS Data script

Generates data from the URTS algorithm for reinforcement learning in decision tree growth.

"""

import yaml
import numpy as np
from tangle_env.envs.tangleEnv import TangleEnv


def urts_gen(config_file):
    """
    Generates data for reinforcement learning in decision tree growth.
    :param config_file:
    :return:
    """

    np.random.seed(0)
    numtests = 10

    # Load configuration data
    with open(config_file, "r") as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)

    env_conf = config_data['envParams']
    tangle_data = {'nodeCount': 1000, 'lambda': 10, 'alpha': 0.5, 'tipSelectionAlgorithm': 'URTS'}

    # Record number of tips left behind, number of algorithm selections
    env_conf['nodeCount'] = tangle_data['nodeCount']
    env_conf['lambdaVal'] = tangle_data['lambda']
    env_conf['alpha'] = tangle_data['alpha']
    env_conf['tipSelectionAlgorithm'] = tangle_data['tipSelectionAlgorithm']
    env_conf['lambdaStep'] = [0, 0]
    tip_select_algo = tangle_data['tipSelectionAlgorithm']

    # Generate base Tangle
    env = TangleEnv(env_conf).unwrapped
    _ = env.render()

    env.alg_conducted = tip_select_algo
    # Generate Tangle using basic algorithms and return results when complete
    last_test = False
    print("Starting training...")
    for i in range(numtests):

        if i == numtests:
            last_test = True

        while True:
            # Save tip std. dev average on last test
            done = env.basic_action(lastTest=last_test)
            if done:
                break

        print("Trail {} complete...".format(i))

    print("Tests complete!")


if __name__ == "__main__":
    urts_gen('./pyserver/configs/serverconfig.yaml')
