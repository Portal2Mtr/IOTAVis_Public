"""Tangle Environment Wrapper

Interacts with the Tangle environment to conduct tests.

"""

import matplotlib
import statistics
import pickle
import yaml
import matplotlib.pyplot as plt
import numpy as np
import time
from itertools import count
from tangle_env.envs.tangleEnv import TangleEnv


def gen_tangle_local(config_file):
    """
        Runs a test for the given parameters from the client and environment config.
        :param config_file: Configuration file string for environment settings
        :return:
        """

    # Load configuration data
    with open(config_file, "r") as file:
        config_data = yaml.load(file)
    print("Starting simulation...")

    algos_test = ["URTS", "WRW", "EIOTA", "almostURTS", "DT"]
    # algos_test = ['DT']
    env_conf = config_data['envParams']
    tangle_data = {
        'nodeCount': 1000,
        'lambda': 10,
        'alpha': 0.5
    }
    num_trials = 10
    tip_logs = []
    comp_logs = []
    for _ in range(len(algos_test)):
        tip_logs.append([])
        comp_logs.append([])

    for idx, algo in enumerate(algos_test):
        for i in range(num_trials):
            print("Conducting algo: {}, Trial #{}, Transactions: {}".format(algo, i + 1, tangle_data['nodeCount']))
            tangle_data['tipSelectionAlgorithm'] = algo
            np.random.seed(i)
            # Repeat test Data
            env_conf['nodeCount'] = tangle_data['nodeCount']
            env_conf['lambdaVal'] = tangle_data['lambda']
            env_conf['alpha'] = tangle_data['alpha']
            env_conf['tipSelectionAlgorithm'] = tangle_data['tipSelectionAlgorithm']
            tip_select_algo = tangle_data['tipSelectionAlgorithm']

            # Generate base Tangle
            env = TangleEnv(env_conf).unwrapped
            current_state = env.render()

            # If a basic algo, do testing and create nodes/links, otherwise step through RL
            if tip_select_algo != 'DT':
                env.alg_conducted = tip_select_algo
                # Generate Tangle using basic algorithms and return results when complete
                start_time = time.time()
                for j in count():
                    done = env.basic_action()
                    if done:
                        break
                    if j % 100 == 0:

                        print("Completed step #{},Time: {:.2f} sec".format(j, time.time()-start_time))
                        start_time = time.time()
            else:

                save_name = './intelligent_tip/trainedModels/dtLearner'
                learn_tree = pickle.load(open(save_name + '.p', 'rb'))
                start_time = time.time()
                for j in count():
                    # Select and perform an action
                    leaf_state = learn_tree.get_leaf(current_state)
                    action = learn_tree.take_action(leaf_state)
                    # Get the reward for conducting an action in the env
                    _, reward, done, _ = env.step(action)
                    if done:
                        break
                    # Observe new state
                    current_state = env.render()
                    if j % 100 == 0:
                        print("Completed step #{},Time: {:.2f} sec".format(j, time.time()-start_time))
                        start_time = time.time()

            report = env.training_report()
            trial_left_behind = report['tipsLeftBehind']
            tip_logs[idx].append(trial_left_behind)
            comp_logs[idx].append(report['numCalcs'])

    # Compare barchart of transactions left behind
    trials_left_avg = []
    for algo in tip_logs:
        trials_left_avg.append(statistics.mean(algo[:]))

    # Save for figure generation
    pickle.dump([trials_left_avg, algos_test, comp_logs], open("./pyserver/outputFiles/localData.p", 'wb'))


def gen_tangle_client(tangle_data, config_file):
    """
    Runs a test for the given parameters from the client and environment config.
    :param tangle_data: Frontend data parameters
    :param config_file: Configuration file name for environment settings
    :return:
    """

    # Load configuration data
    with open(config_file, "r") as config_file:
        config_data = yaml.load(config_file, Loader=yaml.FullLoader)

    # Override with test data from client
    env_conf = config_data['envParams']
    env_conf['nodeCount'] = tangle_data['nodeCount']
    env_conf['lambdaVal'] = tangle_data['lambda']
    env_conf['alpha'] = tangle_data['alpha']
    env_conf['tipSelectionAlgorithm'] = tangle_data['tipSelectionAlgorithm']
    tip_select_algo = tangle_data['tipSelectionAlgorithm']

    # Generate base Tangle
    env = TangleEnv(env_conf).unwrapped
    current_state = env.render()

    # If a basic algo, do testing and create nodes/links, otherwise step through RL
    if tip_select_algo != 'DT':
        env.alg_conducted = tip_select_algo
        # Generate Tangle using basic algorithms and return results when complete
        done = False
        while not done:
            done = env.basic_action()

    else:

        # Load model
        save_name = './intelligent_tip/trainedModels/dtLearner'
        learn_tree = pickle.load(open(save_name + '.p', 'rb'))

        while True:
            # Select and perform an action
            leaf_state = learn_tree.get_leaf(current_state)
            action = learn_tree.take_action(leaf_state)
            # Get the reward for conducting an action in the env
            _, reward, done, _ = env.step(action)
            if done:
                break
            # Observe new state
            current_state = env.render()

    # Return environment
    return env


def local_metrics():
    """
    Compares different tip selection algorithms by creating a barchart and table of data.
    :return:
    """

    trials_left_avg, algos_test, comp_logs = pickle.load(open("./pyserver/outputFiles/localData.p", 'rb'))

    matplotlib.rcParams.update({'font.size': 14})
    fig = plt.figure()

    def auto_label(rects, data):
        """
        Attach a text label above each bar displaying its height
        """
        for rect, datum in zip(rects, data):
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2., height+1, '{:.2f}'.format(datum), ha='center', va='bottom')

    y_pos = np.arange(len(algos_test))
    temp_rects = plt.bar(y_pos, trials_left_avg, align='center', width=0.5)
    auto_label(temp_rects, trials_left_avg)
    algos_label = ["URTS", "MCMC", "EIOTA", "almostURTS", "DT"]
    plt.xticks(y_pos, algos_label)
    plt.ylim([0, max(trials_left_avg) + 5])
    plt.ylabel('Unapproved Transactions')
    plt.xlabel('Algorithm')
    plt.savefig('./pyserver/outputFiles/leftBehindComp.png', bbox_layout='tight', dpi=400)

    # Compare number of computation type
    means_urts = []
    means_uwrw = []
    means_mcmc = []
    for algo in comp_logs:
        get_urts = [i['urts'] for i in algo]
        means_urts.append(statistics.mean(get_urts))
        get_uwrw = [i['uwrw'] for i in algo]
        means_uwrw.append(statistics.mean(get_uwrw))
        get_mcmc = [i['wrw'] for i in algo]
        means_mcmc.append(statistics.mean(get_mcmc))

    print(algos_test)
    print("Mean Comp. (URTS): {}".format(means_urts))
    print("Mean Comp. (UWRW): {}".format(means_uwrw))
    print("Mean Comp. (WRW): {}".format(means_mcmc))
