# Power execution script
#######################

# Script for executing tip selection for raspberry pi to monitor performance
# Alot of the code here is copied over from other scripts

import sys

sys.path.append('./tangle_env/envs/')
sys.path.append('./')
import time
import argparse
import statistics
import pickle
import yaml
import numpy as np
import glob
from tangleEnv import TangleEnv
import matplotlib.pyplot as plt
import matplotlib


def power_calcs(config_file, algo, trial):
    """
        Runs a test for the given parameters from the client and environment config.
        :param config_file: Configuration file string for environment settings
        :return:
        """

    np.random.seed(trial)

    # Load configuration data
    with open(config_file, "r") as file:
        config_data = yaml.load(file, Loader=yaml.FullLoader)

    env_conf = config_data['envParams']
    tangle_data = {'nodeCount': 1000, 'lambda': 20, 'alpha': 0.5, 'tipSelectionAlgorithm': algo}

    # Record number of tips left behind, number of algorithm selections
    env_conf['nodeCount'] = tangle_data['nodeCount']
    env_conf['lambdaVal'] = tangle_data['lambda']
    env_conf['alpha'] = tangle_data['alpha']
    env_conf['tipSelectionAlgorithm'] = tangle_data['tipSelectionAlgorithm']
    env_conf['lambdaStep'] = [0, 0]
    tip_select_algo = tangle_data['tipSelectionAlgorithm']

    # Generate base Tangle
    env = TangleEnv(env_conf).unwrapped
    current_state = env.render()
    cumul_array = [0]

    # If a basic algo, do testing and create nodes/links, otherwise step through RL
    if tip_select_algo != 'DT':
        if tip_select_algo == 'MCMC':
            tip_select_algo = 'WRW'
        env.alg_conducted = tip_select_algo
        # Generate Tangle using basic algorithms and return results when complete
        start_time = time.time()
        while True:
            init_exec_time = time.time()
            done = env.basic_action()
            end_exec_time = time.time()
            total_exec = end_exec_time - init_exec_time
            cumul_array.append(cumul_array[-1] + total_exec)
            if done:
                break

    else:

        save_name = './intelligent_tip/trainedModels/dtLearner'
        learn_tree = pickle.load(open(save_name + '.p', 'rb'))
        start_time = time.time()
        while True:
            # Select and perform an action
            init_exec_time = time.time()
            leaf_state = learn_tree.get_leaf(current_state)
            action = learn_tree.take_action(leaf_state)
            # Get the reward for conducting an action in the env
            _, reward, done, _ = env.step(action)

            end_exec_time = time.time()
            total_exec = end_exec_time - init_exec_time
            cumul_array.append(cumul_array[-1] + total_exec)
            if done:
                break
            # Observe new state
            current_state = env.render()

    report = env.training_report()
    # Store statistics
    comp_algos = report['numCalcs']
    trial_left_behind = report

    comp_time = time.time() - start_time
    pickle.dump([trial_left_behind, algo, comp_algos, comp_time,cumul_array],
                open("./power_analysis/output/{}Light{}.p".format(algo, trial), 'wb'))

def parse_perf_stat_text(file):
    """
    Parses calculation estimates from output of perf stat.
    :param file:
    :return:
    """

    with open(file) as f:
        lines = f.readlines()

    cleanLines = []
    for line in lines:
        temp = line.strip()
        if len(temp) > 1:
            temp = temp.split()
            cleanLines.append(temp)

    retinstr = 0
    retcycles = 0
    for line in cleanLines:
        if 'instructions' in line:
            retinstr = int(line[0])
        if 'cycles' in line:
            retcycles = int(line[0])

    return retcycles, retinstr


def print_results():
    """
    Prints the results of the calculation estimates
    :return:
    """

    algos_test = ["URTS", "MCMC", "EIOTA", "almostURTS", "DT", "DTLight"]
    print(algos_test)
    trials_left_avg = []
    urtscomp = []
    uwrwcomp = []
    wrwcomp = []
    timeLogs = []
    cycleLogs = []
    instrLogs = []
    cumullogs = []
    for readalgo in algos_test:
        # Read and average all pickle files for each algorithm
        algo_left_avg = []
        algo_comp_log = []
        algo_comp_time = []
        algo_cumul_time = []
        for file in glob.glob("./power_analysis/output/{}*.p".format(readalgo)):
            trial_left_behind, algo, comp_algos, comp_time,cumul_time = pickle.load(open(file, 'rb'))
            algo_left_avg.append(trial_left_behind['tipsLeftBehind'])
            algo_comp_log.append(comp_algos)
            algo_comp_time.append(comp_time)
            algo_cumul_time.append(cumul_time)

        trials_left_avg.append(statistics.mean(algo_left_avg))
        algourts = []
        algouwrw = []
        algowrw = []
        for entry in algo_comp_log:
            algourts.append(entry['urts'])
            algouwrw.append(entry['uwrw'])
            algowrw.append(entry['wrw'])
        urtscomp.append(statistics.mean(algourts))
        uwrwcomp.append(statistics.mean(algouwrw))
        wrwcomp.append(statistics.mean(algowrw))
        timeLogs.append(statistics.mean(algo_comp_time))
        cumullogs.append((readalgo,list(np.mean(algo_cumul_time,axis=0))))

        # Read and average all perf stat outputs
        algo_cycle = []
        algo_instr = []
        for file in glob.glob("./power_analysis/Output/{}*.txt".format(algo)):
            cycles,instrs = parse_perf_stat_text(file)
            algo_cycle.append(cycles/1E9)
            algo_instr.append(instrs/1E9)

        cycleLogs.append(statistics.mean(algo_cycle))
        instrLogs.append(statistics.mean(algo_instr))

    # URTS
    print('URTS # Comps:')
    print(urtscomp)

    # UWRW
    print('UWRW # Comps:')
    print(uwrwcomp)

    # WRW
    print('WRW # Comps:')
    print(wrwcomp)

    # Time
    print('Comp. Time: [sec].')
    print(timeLogs)

    # Cycle
    print('# Cycles [E9]:')
    print(cycleLogs)

    # Instruction
    print('Instr. [E9]:')
    print(instrLogs)

    # Left behind
    print('Tip left behind:')
    print(trials_left_avg)

    # Generate cumulative times plot
    font = {'size': 18}
    matplotlib.rc('font', **font)

    xarray = [i for i in range(len(cumullogs[0][1]))]
    for entry in cumullogs:
        algo = entry[0]
        valarray = entry[1]

        if algo != 'DTLight':
            plt.step(xarray, valarray, label=algo)
        else: # Adjust label for smaller decision tree
            plt.step(xarray, valarray, label='DT (Depth=10)', marker='x', markevery=30, linestyle='None')

    plt.xlabel("Computation Step")
    plt.ylabel("Time [s]")
    plt.xlim([0, xarray[-1]])
    plt.ylim([0, 180])
    plt.legend()
    plt.subplots_adjust(left=0.15)
    plt.subplots_adjust(bottom=0.15)
    plt.savefig('./power_analysis/output/step_fig.png', bbox_layout='tight', dpi=400)


if __name__ == "__main__":

    # Use algorithm from args to evaluate power calcs
    config_file = './pyserver/configs/server_config.yaml'
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", help="Algorithm to use for calcs", type=str)
    parser.add_argument("--plot", help="Determine if plotting", type=bool, default=False)
    parser.add_argument("--trial", help="Trial number and seed", type=int)
    args = parser.parse_args()

    if args.plot:
        print_results()
    else:
        power_calcs(config_file, args.algo, args.trial)
