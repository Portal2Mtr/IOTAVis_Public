"""tangleEnv.py

Contains the main class for the Tangle OpenAI RL environment. Uses
the functions in ./algorithms.py and ./tipselection.py for conducting actions
in building the ledger.

Resource used for buidling this environment: https://github.com/iotaledger/iotavisualization

"""

import gym
from statistics import stdev
from numpy.random import exponential
from statistics import mean
from .tipSelection import *
from collections import deque
import numpy as np
import pickle

class TangleEnv(gym.Env):
    """
    Tangle OpenAI gym environment class.

    Creates a Tangle distributed ledger that can be simulated in discrete steps.
    The resulting ledger with appropriate tip selection algorithm can be passed
    to the react client for viewing or for the local python script for testing.

    """
    metadata = {'render.modes': ['human']}

    def __init__(self, env_conf):
        np.random.seed(env_conf['envSeed'])
        # Setup config vars
        self.env_conf = env_conf
        self.tip_select_dict = {
            'URTS': uniform_random,
            'UWRW': unweighted_mcmc,
            'WRW': weighted_mcmc,
            'EIOTA': eiota,
            'almostURTS': almost_urts
        }
        self.time_window = [self.env_conf['mcmcWindowStop'], self.env_conf['mcmcWindowStart']]
        self.node_count = self.env_conf['nodeCount']
        self.lambda_rate = self.env_conf['lambdaVal']
        self.alpha = env_conf['alpha']
        self.h = 1
        # Initialize ledger transaction arrival values
        self.curr_tips = []
        self.links = []
        self.genesis = {}
        self.num_steps = 0
        self.lambda_guess = 0
        self.nodes = None
        self.init_tangle()

        # Setup RL Parameters
        self.num_select = 2
        self.alg_conducted = "URTS"
        self.reward = 0
        self.status = 0
        self.counter = 0
        self.curr_time = 0
        self.node_idx = 0
        self.recent_tip = 0
        self.tips_time_mean = 0
        self.tips_target = 0
        self.tips_error = 0
        # Used to smooth render values for learner
        self.env_history = [deque([]), deque([]), deque([]), deque([])]

        # Reporting vars for logging tangle generation
        self.alg_report = self.env_conf['tipSelectionAlgorithm']
        self.record_tips = []
        # Record the number of tip calculations for logging
        self.num_calc = {'urts': 0, 'uwrw': 0, 'wrw': 0}
        self.record_num_tips()
        self.last_left = None
        self.urts_logging = []

    def init_tangle(self):
        """
        Generates the transaction times for the ledger from a poisson process model.
        :return:
        """
        # Generate nodes from poisson dist.
        self.links = []
        self.num_steps = 0
        self.genesis = {
            'name': '0',
            'time': 0,
            'cumWeight': 1
        }
        nodes = [self.genesis]

        # Generate node times for simulated poisson process using lambda
        time = self.h
        while len(nodes) < self.node_count:
            delay = float(exponential(1 / self.lambda_rate, 1))
            time += delay
            nodes.append({
                'name': str(len(nodes)),
                'time': time,
                'x': 300,
                'y': 200,
                'cumWeight': 1
            })

        self.num_steps = 0
        self.nodes = nodes
        self.links = []  # Initialize list for transaction links
        self.curr_tips = [self.genesis]

    def create_links(self, tips, node):
        """
        Creates links between transactions after tip selection algorithm is conducted
        :param tips: New tips selected from tip algorithm.
        :param node: New transaction for creating links.
        :return:
        """
        if len(tips) > 0:
            tip_nums = [i['name'] for i in tips]
            unique_tips = set(tip_nums)
            self.curr_tips.append(node)
            for tip in unique_tips:
                new_tip = next(d for i, d in enumerate(tips) if d['name'] == tip)
                self.links.append({'source': node, 'target': new_tip})
                if new_tip in self.curr_tips:
                    self.curr_tips.remove(new_tip)

        # Update max time for reward
        self.curr_time = max(self.curr_time, node['time'])
        self.record_num_tips()

    def basic_action(self,logging=None,lastTest=False):
        """
        Conducts a basic action for simple tip selection algorithms
        :return: Environment completion bool variable
        """

        self.num_steps += 1
        self.node_idx += 1

        # Check if conducted max number of steps, then we are done
        self.status = (self.num_steps >= self.node_count)
        if self.status:
            return True

        node = self.nodes[self.node_idx]
        candidates = [i for i in self.nodes if i['time'] < (node['time'] - self.h)]
        candidate_links = [i for i in self.links if i['source']['time'] < (node['time'] - self.h)]
        tips, num_calc = self.tip_select_dict[self.alg_conducted](candidates, candidate_links, node['time'], self)

        # Add tips to current tips to record unconfirmed transactions
        self.create_links(tips, node)

        if logging is not None and self.status:
            # Get time std dev and mean for the current tips
            tip_time = [i['time'] for i in self.curr_tips]
            self.tips_time_mean = 1
            tip_std = 0.0
            if len(tip_time) > 2:
                self.tips_time_mean = max(math.fsum(tip_time) / len(tip_time), 1)
                tip_std = stdev(tip_time)

            self.urts_logging.append(tip_std)
            if not self.lastTest:
                mean_tip_avg = mean(self.urts_logging)
                pickle.dump(mean_tip_avg,open('./intelligent_tip/output/compareData.p','rb'))

        return self.status

    def step(self, action):
        """
        Perform an environment step with the selected action from the learner
        :param action: Learner action (see self.tip_select_dict)
        :return:
        """
        # Conduct algorithm step
        self.take_action(action)
        if self.status:
            # Max steps for tangle execution
            ob = [self.nodes, self.links]
            return ob, self.reward, self.status, {}

        self.reward = self.get_reward(action)
        ob = [self.nodes, self.links]
        return ob, self.reward, self.status, {}

    def reset(self):
        """
        Resets the ledger state and transaction times.
        :return:
        """
        self.reward = 0
        self.node_idx = -1
        self.counter = 0
        self.status = False

        self.curr_time = 0
        self.num_select = 2
        self.h = 1
        # Reinitialize tangle transaction arrival values
        self.init_tangle()

        self.recent_tip = 0
        self.num_calc = {'urts': 0, 'uwrw': 0, 'wrw': 0}
        self.env_history = [deque([]), deque([]), deque([]), deque([])]
        self.last_left = None

    def render(self, mode='human'):
        """
        Returns the state of the ledger for the machine learner.
        Returns: [lambda, total tips,mean tip time, tim time std. dev.]
        :return:
        """
        # Get distribution of arriving transactions and calculate the incoming rate
        recent_times = []
        self.lambda_guess = 1
        if self.recent_tip > 1:
            recent_times = [i['time'] for i in self.nodes if int(i['name']) < self.recent_tip]
            time_diffs = []
            for i in range(1, len(recent_times)):
                time_diffs.append(recent_times[i] - recent_times[i - 1])
            # Guess lambda rate value based on arrival of incoming transactions
            self.lambda_guess = 1 / mean(time_diffs)

        # Get time std dev and mean for the current tips
        tip_time = [i['time'] for i in self.curr_tips]
        self.tips_time_mean = 1
        tip_std = 0.0
        if len(tip_time) > 2:
            self.tips_time_mean = max(math.fsum(tip_time) / len(tip_time), 1)
            tip_std = stdev(tip_time)

        num_curr_tips = len(self.curr_tips)
        self.tips_target = max((2 * self.lambda_guess) / (2 - 1), 1)

        # Calculate an 'ideal' number of tips to target for the learner in reward func from urts
        self.tips_error = abs(self.tips_target - num_curr_tips)
        self.tips_time_mean = max(self.tips_time_mean - min(tip_time), 0.01)

        # Smooth the render values to reduce environment noise
        env_render = [self.lambda_guess, self.tips_error, self.tips_time_mean, tip_std]
        for idx, _ in enumerate(self.env_history):
            self.env_history[idx].append(env_render[idx])

        # Handle if just starting to build ledger
        return env_render


    def take_action(self, action):
        """
        Conducts an action based on the decision from the machine learner.
        :param action: Machine learner action
        :return: None
        """

        self.num_steps += 1
        self.node_idx += 1

        # Check if conducted max number of steps
        self.status = (self.num_steps >= self.node_count)
        if self.status:
            return

        node = self.nodes[self.node_idx]
        candidates = [i for i in self.nodes if i['time'] < (node['time'] - self.h)]
        max_cand = 0
        for newNode in candidates:
            max_cand = max(int(newNode['name']), max_cand)
        self.recent_tip = max_cand
        cand_links = [i for i in self.links if i['source']['time'] < (node['time'] - self.h)]

        # Switch/case for algorithm actions
        if action == 3:  # High alpha WRW
            self.num_select = 2
            self.alpha = self.env_conf['alphaHigh']
            tips, num_calcs = self.tip_select_dict['WRW'](candidates, cand_links, node['time'], self)
            self.alg_conducted = 'WRW'

        elif action == 2:  # Low alpha WRW
            self.num_select = 2
            self.alpha = self.env_conf['alphaLow']
            tips, num_calcs = self.tip_select_dict['WRW'](candidates, cand_links, node['time'], self)
            self.alg_conducted = 'WRW'

        elif action == 1:  # High count URTS
            self.num_select = 3
            tips, num_calcs = self.tip_select_dict['URTS'](candidates, cand_links, node['time'], self)
            self.alg_conducted = 'URTS'
        else:  # Normal URTS
            self.num_select = 2
            tips, num_calcs = self.tip_select_dict['URTS'](candidates, cand_links, node['time'], self)
            self.alg_conducted = 'URTS'

        self.create_links(tips, node)

        # Update max time for reward
        self.curr_time = max(self.curr_time, node['time'])
        self.record_num_tips()
        self.num_calc = num_calcs

    def get_reward(self, action):
        """
        Calculate the reward for the machine learner.
        :return: reward
        """

        # Find trxns falling behind
        tip_set = self.curr_tips

        if len(tip_set) > 1:
            times = [i['time'] for i in tip_set]
            time_mean = mean(times)
            time_std = stdev(times, time_mean)
        else:
            time_mean = 1.0
            time_std = 0.0

        left_behind = 0
        for tip in tip_set:
            if tip['time'] < (time_mean - 3 * time_std):
                left_behind += 1

        if time_std != 0:
            left_reward = (left_behind/len(tip_set))**(self.num_select/time_std)
            rew_start = 0.9 ** (1+time_std)
        else:
            left_reward = 0
            rew_start = 0.9

        did_markov = 0
        if self.alg_conducted == "WRW":

            if left_behind == 0:
                if action == 2:
                    did_markov = 0.1 * self.env_conf['alphaLow']
                elif action == 3:
                    did_markov = 0.1 * self.env_conf['alphaHigh']

        catch_urts = 0.0
        if self.alg_conducted == "URTS":
            catch_urts = 0.01 * left_behind

        reward = rew_start - 0.05*(self.num_select - 2) - left_reward + did_markov + catch_urts
        reward = max(0, min(1, reward))

        return reward

    def record_num_tips(self):
        """
        Records the total number of tips for the Tangle at a given time for data logging
        :return:
        """
        num_tips = len(self.curr_tips)
        self.record_tips.append(num_tips)

    def get_tangle(self):
        """
        Returns a dict with the ledger nodes and links for plotting.
        :return:
        """
        return {
            'nodes': self.nodes,
            'links': self.links
        }

    def training_report(self):
        """
        Reports various information about the environment trial.
        Conducts separate trial to estimate tips left behind compared
        to ideal case.
        :return:
        """
        data = {
            'tipsLog': self.record_tips,
            'numCalcs': self.num_calc,
            'algoUsed': self.alg_report,
            'lambdaVal': self.lambda_rate,
            'numNodes': len(self.nodes),
        }

        # Get number of transactions left behind at end of testing
        # from URTS statistics

        num_std = 3
        tip_times = [i['time'] for i in self.curr_tips]
        time_mean = mean(tip_times)
        # Compare to ideal URTS case
        urts_std_dev_avg = pickle.load(open('./intelligent_tip/output/compareData.p','rb'))

        left_behind = [tipTime < (time_mean-(num_std*urts_std_dev_avg)) for tipTime in tip_times]
        left_cnt = 0
        for idx in left_behind:
            if idx:
                left_cnt += 1

        data['tipsLeftBehind'] = left_cnt
        return data
