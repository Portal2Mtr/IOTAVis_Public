"""
Bandit algorithm functions

Bandit algorithm functions for decision tree action making and updates.
Novel algorithm included here for simple trimmed mean.
"""

import copy
from statistics import mean
import statistics
import numpy as np


def filter_mean(self, rew_vals):
    """
    Filtered mean to prevent adversarial corruption, inspired from: https://arxiv.org/abs/1910.05625
    :param self: Decision tree object
    :param rew_vals: Reward values for taking trimmed mean
    :return: Final reward value for updating beta distributions
    """

    rew_samples = len(rew_vals)
    filt_window = self.bandit_batch
    if rew_samples < filt_window:
        return rew_vals[-1]

    rew_vals = rew_vals[-filt_window:]

    changes = []
    for i in range(len(rew_vals)-1):
        changes.append(rew_vals[i+1] - rew_vals[i])

    signs = [-1 if changes[i] < 0 else 1 for i in range(len(changes))]
    changes = [abs(i) for i in changes]
    mean_diff = mean(changes)
    dev_diff = statistics.stdev(changes)
    attack_flags = []
    for i in range(len(changes)):
        attack_flags.append(changes[i] > (mean_diff + 3*dev_diff))

    for idx, val in enumerate(attack_flags):

        if val:
            # Not doing attack simulation, so not adding transactions to set M from paper
            # Remove reward outlier from history
            for jdx in range(idx, len(rew_vals)-1):
                rew_vals[jdx+1] = rew_vals[jdx+1] - signs[jdx] * changes[jdx]

    return mean(rew_vals)


def update_bandit(self, state, action):
    """
    Update beta distribution values in bandit algorithm.
    :param self: Decision tree object
    :param state: Current environment state
    :param action: Last selected action
    :return:
    """

    self.tree.nodes[state]['bandCnt'] += 1
    self.tree.nodes[state]['freqArms'][action] += 1
    if self.tree.nodes[state]['bandCnt'] < self.bandit_batch:
        return

    curr_state = self.tree.nodes[state]
    freq_list = curr_state['freqArms']

    # Update all arms simultaneously
    for idx, freq in enumerate(freq_list):
        if freq != 0:
            # Perform trimmed mean for adversarial tolerance
            rew_vals = copy.deepcopy(curr_state['rewvalues'][idx])
            rew_mean = mean(rew_vals[-freq:])
            r_arm = max(0.001, freq * rew_mean)
            f_arm = max(0.001, freq * (1-rew_mean))
            update_val = np.random.beta(r_arm, f_arm)
            curr_state['bandAlpha'][idx] += update_val
            curr_state['bandBeta'][idx] += (1-update_val)

    curr_state['bandCnt'] = 0
    for i in range(len(curr_state['freqArms'])):
        curr_state['freqArms'][i] = 0

    self.tree.nodes[state]['bandAlpha'] = curr_state['bandAlpha']
    self.tree.nodes[state]['bandBeta'] = curr_state['bandBeta']
    self.tree.nodes[state]['bandCnt'] = curr_state['bandCnt']
    self.tree.nodes[state]['freqArms'] = curr_state['freqArms']


def reset_bandit(self):
    """
    Reset bandit parameters for Biased TS
    :param self:
    :return:
    """
    for node, data in self.branches.nodes(data=True):
        data['bandCnt'] = 0
        for i in range(len(data['freqArms'])):
            data['freqArms'][i] = 0
