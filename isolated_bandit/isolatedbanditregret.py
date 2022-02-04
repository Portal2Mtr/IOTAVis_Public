# Plots Isolated Bandit Regret
############

# Simulates several basic multi-arm bandit algorithms and compares the regret, including Biased TS.

# Simulation attack ideas taken from: https://arxiv.org/abs/1906.01528

import matplotlib
import statistics
import pickle
import math
import yaml
import copy
import numpy.random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from multiprocessing import Pool

import logging
from colorlog import ColoredFormatter
# Setup logger
logger = logging.getLogger(__name__)


def configure_logging(verbosity, enable_colors):
    """
    Configures the code logger for reporting various information.
    :param verbosity: Sets logger verbosity level
    :type verbosity: str
    :param enable_colors: Enables logger colors
    :type enable_colors: bool
    :return:
    :rtype:
    """
    root_logger = logging.getLogger()
    console = logging.StreamHandler()

    if enable_colors:
        # create a colorized formatter
        formatter = ColoredFormatter(
            "%(log_color)s[%(filename)s] %(asctime)s %(levelname)-8s%(reset)s %(white)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "cyan,bg_red",
            },
            secondary_log_colors={},
            style="%"
        )
    else:
        # create a plain old formatter
        formatter = logging.Formatter(
            "[%(filename)s] %(asctime)s %(levelname)-8s %(message)s"
        )

    # Add the formatter to the console handler, and the console handler to the root logger
    console.setFormatter(formatter)
    for hdlr in root_logger.handlers:
        root_logger.removeHandler(hdlr)
    root_logger.addHandler(console)

    # Set logging level for root logger
    root_logger.setLevel(verbosity)


class BaseBandit:

    def __init__(self, budget, time):
        self.num_arms = 3
        self.memory = [[] for _ in range(self.num_arms)]
        self.regret = []
        self.total_regret = 0
        self.epoch_log = []
        self.attack_val = budget / time
        self.budget = float(budget)
        self.attack_budget = [self.budget for _ in range(self.num_arms)]
        self.prevent_attack = [False for _ in range(self.num_arms)]
        self.prevent_attack[-1] = True

    def trim_reward(self, reward):
        """
        Bounds reward to [0,1]
        :param reward: Reward value (float)
        :return:
        """
        # Set reward to [0,1] for Biased TS

        return max(min(reward, 1), 0)

    def record_regret(self, regret):
        """
        Records regret for the current bandit.
        :param regret: Regret value (float)
        :return: None
        """

        abs_regret = abs(regret)
        self.total_regret += abs_regret
        self.regret.append(self.total_regret)

    def get_regret(self):
        """
        Returns the regret list for the current bandit.
        :return: Regret (list)
        """

        return self.regret

    def update_attack(self, reward, arm):
        """
        Update reward values for a given arm with attack.
        :param reward: Reward for arm (float)
        :param arm: Bandit arm (int)
        :return: None
        """

        # reward = self.trim_reward(reward)
        if arm == 2:

            # Optimal arm, return reward
            self.memory[arm].append(reward)
        else:
            # Attack with cumulative lsi reward attack
            reward = self.perform_attack(reward, arm)
            self.memory[arm].append(reward)

    def log_epoch(self):
        """
        Logs regret for current epoch of testing.
        :return: None
        """

        self.epoch_log.append(self.regret)

    def get_datalog(self):
        """
        Return regret for all epochs.
        :return: All epoch regret (list of lists)
        """
        return self.epoch_log

    def perform_attack(self, reward, arm):
        """
        Performs an attack on a given arm based on set parameters.
        :param reward: Original reward value.
        :param arm: Arm to attack
        :return: Modified reward (float)
        """

        # Reward attack performed on ucb
        if not self.prevent_attack[arm]:
            reward += self.attack_budget[arm]
            self.attack_budget[arm] = 0
            self.prevent_attack[arm] = True

        return reward


class UCB(BaseBandit):

    def __init__(self, budget, time):
        super().__init__(budget, time)
        self.arm_cnt = [1 for _ in range(self.num_arms)]

    def max_ucb(self,time):
        """
        Return max mean for each arm.
        :return: Max arm mean (float)
        """

        max_ucb = 0
        max_arm = 0
        for idx, arm in enumerate(self.memory):
            arm_mean = statistics.mean(arm)
            if len(arm) < 2:
                arm_std = 0
            else:
                arm_std = statistics.stdev(arm)
            arm_ucb = arm_mean + 3*arm_std*math.sqrt(math.log(time) / self.arm_cnt[idx])
            if arm_ucb > max_ucb:
                max_ucb = arm_ucb
                max_arm = idx

        self.arm_cnt[max_arm] += 1
        return max_arm

    def get_arm(self, time):
        """
        Gets an arm for the current bandit.
        :param time:
        :return:
        """

        if time <= self.num_arms:
            arm_select = time-1
        else:
            arm_select = self.max_ucb(time)

        return arm_select

    def reset_bandit(self):
        """
        Resets the memory and other parameters for the bandit.
        :return: None
        """

        self.memory = [[0] for _ in range(self.num_arms)]
        self.regret = []
        self.total_regret = 0
        self.prevent_attack = [False for _ in range(self.num_arms)]
        self.prevent_attack[-1] = True


class GaussianTS(BaseBandit):

    def __init__(self, budget, time):
        super().__init__(budget, time)
        self.mus = [0 for _ in range(self.num_arms)]
        self.num_select = [0 for _ in range(self.num_arms)]

    def get_arm(self, time):
        """
        Gets the arm for the current bandit that performs an action.
        :return: Arm with maximum value for selection (float)
        """

        # Sample each arm first
        if time <= self.num_arms:
            return time-1

        max_idx = 0
        max_val = 0
        for i in range(self.num_arms):

            muval = self.mus[i]

            sample = numpy.random.normal(muval, 1 / self.num_select[i])
            if sample > max_val:
                max_val = sample
                max_idx = i

        return max_idx

    def update_shape_ts(self, arm, reward):
        """
        Update the mean reward values for Gaussian TS.
        :param arm: Arm to update.
        :param reward: Reward to update arm
        :return: None
        """

        # Perform attack on mean for arm
        if not self.prevent_attack[arm]:
            self.memory[arm].append(reward + self.attack_budget[arm])
            self.attack_budget[arm] = 0
            self.prevent_attack[arm] = True
        else:
            self.memory[arm].append(reward)
        self.mus[arm] = statistics.mean(self.memory[arm])
        self.num_select[arm] += 1

    def reset_bandit(self):
        """
        Resets the parameters for a test with the current bandit.
        :return:
        """

        self.memory = [[0] for _ in range(self.num_arms)]
        self.mus = [1 for _ in range(self.num_arms)]
        self.regret = []
        self.num_select = [1 for _ in range(self.num_arms)]
        self.total_regret = 0


class BiasedTS(BaseBandit):

    def __init__(self, budget, time, batch):
        super().__init__(budget, time)
        self.alphas = [1 for _ in range(self.num_arms)]
        self.betas = [1 for _ in range(self.num_arms)]
        self.num_select = [0 for _ in range(self.num_arms)]
        self.C = batch
        self.biascnt = [0 for _ in range(self.num_arms)]
        self.cutoff = 3

    def get_arm(self,time):
        """
        Returns the maximum random arm value for Biased TS
        :return: Max idx (int)
        """

        if time <= self.num_arms:
            return time-1

        max_sample = 0
        max_idx = 0
        for i in range(self.num_arms):
            new_sample = numpy.random.beta(self.alphas[i], self.betas[i])
            if new_sample > max_sample:
                max_sample = new_sample
                max_idx = i

        self.num_select[max_idx] += 1
        return max_idx

    def filter_reward(self, reward, arm):
        """
        Filters reward values to prevent impulse attacks in Biased TS.
        :param reward: Reward value for arm (float)
        :param arm:  Arm selected (int)
        :return: Filtered reward value (float)
        """

        temp_memory = copy.deepcopy(self.memory)
        temp_memory[arm].append(reward)

        if len(temp_memory[arm]) < self.C:
            return reward

        curr_vals = temp_memory[arm][-self.C:]
        changes = [curr_vals[i+1] - curr_vals[i] for i in range(len(curr_vals)-1)]
        signs = [-1 if changes[i] < 0 else 1 for i in range(len(changes))]
        changes = [abs(i) for i in changes]

        mean_diff = statistics.mean(changes)
        dev_diff = statistics.stdev(changes)
        attack_flags = []
        for i in range(len(changes)):
            attack_flags.append(changes[i] > (mean_diff + self.cutoff*dev_diff))

        for idx, val in enumerate(attack_flags):
            if val:
                for jdx in range(idx, len(attack_flags)):
                    curr_vals[jdx+1] = curr_vals[jdx+1] - signs[jdx] * changes[jdx]

        return statistics.mean(curr_vals)

    def update_attack(self, reward, arm):
        """
        Update arm shape parameters with new reward value
        :param reward: Reward value for updating (float)
        :param arm: Arm to update (int)
        :return: None
        """

        if arm == 2:
            reward = self.trim_reward(reward)
            reward = self.filter_reward(reward, arm)
            self.memory[arm].append(reward)
        else:
            # Attack with cumulative lsi reward attack after
            reward = self.perform_attack(reward, arm)
            # Trimming the reward after the attack is performed maximizes it to 1
            reward = self.trim_reward(reward)
            reward = self.filter_reward(reward, arm)
            self.memory[arm].append(reward)

    def update_biased_shape(self, arm):
        """
        Update arm parameters for Biased TS.
        :param arm: Arm to update (int)
        :return: None
        """

        self.num_select[arm] += 1
        if self.num_select[arm] < self.C:
            return

        # Update each arm with batch-like update
        for i in range(self.num_arms):
            mean_rew = statistics.mean(self.memory[i][-self.num_select[i]:])
            i_val = max(1, self.num_select[i] * mean_rew)
            f_val = max(1, self.num_select[i] * (1 - mean_rew))
            update_val = numpy.random.beta(i_val, f_val)
            self.alphas[i] = self.alphas[i] + update_val
            self.betas[i] = self.betas[i] + (1-update_val)
            self.num_select[i] = 0

    def reset_bandit(self):
        """
        Resets the parameters for Biased TS.
        :return: None
        """

        self.memory = [[0] for _ in range(self.num_arms)]
        self.regret = []
        self.alphas = [1 for _ in range(self.num_arms)]
        self.betas = [1 for _ in range(self.num_arms)]
        self.num_select = [0 for _ in range(self.num_arms)]
        self.biascnt = [0 for _ in range(self.num_arms)]
        self.total_regret = 0


def get_reward_regret(arm, sim_params):
    """
    Get rewards for each arm and calculate regret (acts like dummy environment)
    :param arm: Arm selected by bandit.
    :param sim_params: Configuration values for arm reward distributions.
    :return: Reward, regret (float,float)
    """

    arm_mean_1 = sim_params['arm_mean_1']
    arm_mean_2 = sim_params['arm_mean_2']
    arm_mean_3 = sim_params['arm_mean_3']
    optimal_arm_idx = 2

    arms = [arm_mean_1, arm_mean_2, arm_mean_3]
    arm_var = sim_params['arm_var']

    reward = numpy.random.normal(arms[arm], math.sqrt(arm_var))
    if arm == optimal_arm_idx:
        regret = 0
    else:
        regret = numpy.random.normal(arms[optimal_arm_idx]) - reward

    return reward, regret


def plot_regret():
    """
    Plots regret for all simulated trials.
    :return: Outputs images in ./output
    """

    font = {'size': 18}
    matplotlib.rc('font', **font)

    # Plot control case
    x_log, ucb_avg_log, ts_avg_log, biased_avg_log = pickle.load(open('./output/banditControl.p', 'rb'))

    # Plot average regret
    fig, ax = plt.subplots()
    ax.plot(x_log, ucb_avg_log, label='UCB')
    ax.plot(x_log, ts_avg_log, label='Gaussian TS')
    ax.plot(x_log, biased_avg_log, label='Biased TS')
    ax.plot()
    ax.set_xlabel('ln(t)')
    ax.set_ylabel('Regret')
    ax.set_xlim([0, x_log[-1]])
    # ax.set_ylim([0, 900])
    ax.legend(loc='upper left')
    plt.savefig('./output/banditControl.png', bbox_inches='tight', dpi=400)

    # Plot bandit attack
    x_log_attack, ucb_avg_attack, ts_avg_attack, biased_avg_attack = \
        pickle.load(open('./output/banditAttack.p', 'rb'))
    plt.cla()
    plt.clf()

    # Plot average regret

    fig2, ax2 = plt.subplots()

    ax2.plot(x_log_attack, ucb_avg_attack, label='UCB')
    ax2.plot(x_log_attack, ts_avg_attack, label='Gaussian TS')
    ax2.plot(x_log_attack, biased_avg_attack, label='Biased TS')
    ax2.set_xlabel('ln(t)')
    ax2.set_ylabel('Regret')
    ax2.set_xlim([0, x_log_attack[-1]])
    # ax2.set_ylim([0, 900])
    ax2.legend(loc='upper left')
    plt.savefig('./output/banditAttack.png', bbox_inches='tight', dpi=400)


def isolated_ts(seed, config, budget):
    """
    Isolated Gaussian Thompson Sampling simulation for parallel processing.
    :param seed: Seed to run single trial repeatedly
    :param config: Configuration parameters for trial
    :param budget: Budget for attacker (control or attack budget)
    :return: Datalog of regret for trial
    """

    # logger.info("Starting trial #{} for Thompson Sampling".format(seed))
    sim_params = config['simParams']
    budget_arm = budget
    num_epochs = sim_params['num_epochs']

    reg_ts_bandit = GaussianTS(budget_arm, num_epochs)
    numpy.random.seed(seed=seed)
    for t in range(1, num_epochs + 1):
        arm_select = reg_ts_bandit.get_arm(t)  # Attack during arm pull
        reward, regret = get_reward_regret(arm_select, sim_params)
        reg_ts_bandit.update_shape_ts(arm_select, reward)  # Perform attack in here
        reg_ts_bandit.record_regret(regret)

    reg_ts_bandit.log_epoch()
    reg_ts_bandit.reset_bandit()

    datalog = numpy.array(reg_ts_bandit.get_datalog())
    return datalog


def isolated_biasedts(seed, config, budget):
    """
    Isolated Gaussian Thompson Sampling simulation for parallel processing.
    :param seed: Seed to run single trial repeatedly
    :param config: Configuration parameters for trial
    :param budget: Budget for attacker (control or attack budget)
    :return: Datalog of regret for trial
    """

    sim_params = config['simParams']
    budget_arm = budget
    num_epochs = sim_params['num_epochs']

    # Biased TS
    biased_ts_bandit = BiasedTS(budget_arm, num_epochs, sim_params['biased_ts_batch'])

    numpy.random.seed(seed=seed)
    for t in range(1, num_epochs + 1):
        arm_select = biased_ts_bandit.get_arm(t)
        reward, regret = get_reward_regret(arm_select, sim_params)
        biased_ts_bandit.update_attack(reward, arm_select) # Attack reward values
        biased_ts_bandit.update_biased_shape(arm_select)
        biased_ts_bandit.record_regret(regret)

    biased_ts_bandit.log_epoch()
    biased_ts_bandit.reset_bandit()

    biased_datalog = numpy.array(biased_ts_bandit.get_datalog())
    return biased_datalog


def isolated_ucb(seed, config, budget):
    """
    Isolate UCB simulation for parallel processing.
    :param seed: Seed to run single trial repeatedly
    :param config: Configuration parameters for trial
    :param budget: Budget for attacker (control or attack budget)
    :return: Datalog with regret
    """

    sim_params = config['simParams']
    budget_arm = budget
    num_epochs = sim_params['num_epochs']
    ucb_bandit = UCB(budget_arm, num_epochs)

    numpy.random.seed(seed=seed)
    for t in range(1, num_epochs + 1):
        arm_select = ucb_bandit.get_arm(t)
        reward, regret = get_reward_regret(arm_select, sim_params)
        ucb_bandit.update_attack(reward, arm_select)
        ucb_bandit.record_regret(regret)

    ucb_bandit.log_epoch()
    ucb_bandit.reset_bandit()

    datalog = numpy.array(ucb_bandit.get_datalog())

    return datalog


def run_study(config, num_jobs):
    """
    Simple parameter study to find good configuration for regret
    :param config: Config dict from file
    :param num_jobs: Number of processors for parallel execution
    :return:
    """

    # Run parallel study for control budget
    sim_params = config['simParams']
    seed_array = [i for i in range(sim_params['num_trials'])]
    logger.info("Starting multiprocessing!")
    budget = sim_params['control_budget']
    with Pool(num_jobs) as p:
        ucb_vals = p.starmap(isolated_ucb, [(i, config, budget) for i in seed_array])
    with Pool(num_jobs) as p:
        ts_vals = p.starmap(isolated_ts, [(i, config, budget) for i in seed_array])
    with Pool(num_jobs) as p:
        biasedts_vals = p.starmap(isolated_biasedts, [(i, config, budget) for i in seed_array])
    logger.info("Multiprocessing complete!")
    ucb_mean = np.mean(ucb_vals,axis=0).tolist()[0]
    ts_mean = np.mean(ts_vals,axis=0).tolist()[0]
    bts_mean = np.mean(biasedts_vals,axis=0).tolist()[0]

    pickle.dump([ucb_mean,ts_mean,bts_mean],open('./output/control_array.p','wb'))

    # Run parallel study for attack budget
    seed_array = [i for i in range(sim_params['num_trials'])]
    logger.info("Starting multiprocessing!")
    budget = sim_params['attack_budget']
    with Pool(num_jobs) as p:
        ucb_vals = p.starmap(isolated_ucb, [(i, config, budget) for i in seed_array])
    with Pool(num_jobs) as p:
        ts_vals = p.starmap(isolated_ts, [(i, config, budget) for i in seed_array])
    with Pool(num_jobs) as p:
        biasedts_vals = p.starmap(isolated_biasedts, [(i, config, budget) for i in seed_array])
    logger.info("Multiprocessing complete!")
    ucb_mean = np.mean(ucb_vals, axis=0).tolist()[0]
    ts_mean = np.mean(ts_vals, axis=0).tolist()[0]
    bts_mean = np.mean(biasedts_vals, axis=0).tolist()[0]

    pickle.dump([ucb_mean, ts_mean, bts_mean], open('./output/attack_array.p', 'wb'))

    # Plotting Control Data

    font = {'size': 18}
    matplotlib.rc('font', **font)

    [ucb_mean, ts_mean, bts_mean] = pickle.load(open('./output/control_array.p','rb'))
    num_epochs = sim_params['num_epochs']
    x_log = [math.log(i) for i in range(1, num_epochs+1)]
    fig, ax = plt.subplots()
    ax.plot(x_log, ucb_mean, label='UCB')
    ax.plot(x_log, ts_mean, label='TS')
    ax.plot(x_log, bts_mean, label='Biased TS')
    ax.plot()
    ax.set_xlabel('ln(t)')
    ax.set_ylabel('Regret')
    ax.set_xlim([0, x_log[-1]])
    # ax.set_ylim([0, 900])
    ax.legend(loc='upper left')
    plt.savefig('./output/control_test.png', bbox_inches='tight', dpi=400)

    # Plot Attack Data

    [ucb_mean, ts_mean, bts_mean] = pickle.load(open('./output/attack_array.p', 'rb'))
    num_epochs = sim_params['num_epochs']
    x_log = [math.log(i) for i in range(1, num_epochs+1)]
    fig, ax = plt.subplots()
    ax.plot(x_log, ucb_mean, label='UCB')
    ax.plot(x_log, ts_mean, label='TS')
    ax.plot(x_log, bts_mean, label='Biased TS')
    ax.plot()
    ax.set_xlabel('ln(t)')
    ax.set_ylabel('Regret')
    ax.set_xlim([0, x_log[-1]])
    # ax.set_ylim([0, 900])
    ax.legend(loc='upper left')
    plt.savefig('./output/attack_test.png', bbox_inches='tight', dpi=400)


if __name__ == "__main__":

    configure_logging("INFO", False)

    with open("./config/regret_config.yaml", "r") as read_file:
        config = yaml.load(read_file, Loader=yaml.FullLoader)

    parser = argparse.ArgumentParser()
    parser.add_argument('--numJobs', type=str, default=None)
    args = parser.parse_args()

    numJobs = int(args.numJobs)
    run_study(config, numJobs)

