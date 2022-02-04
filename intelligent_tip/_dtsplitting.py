"""
Decision tree node splitting class

Management class for handling node splits in the decision tree
"""

import copy


class NodeSplits(object):
    """
    Node split handling management class for conservative Q learning
    """

    def __init__(self, visit_decay, num_feat, num_actions):
        """
        Initialize split object
        :param visit_decay: Decay value for reducing node split decay
        :param num_feat: Number of environment features
        :param num_actions: Number of actions available
        """

        self.splits = []
        act_dict = {}
        for idx in range(num_actions):
            act_dict[idx] = [0]
        left_dict = copy.deepcopy(act_dict)
        right_dict = copy.deepcopy(act_dict)

        self.split_q_tables = []
        for _ in range(num_feat):
            self.split_q_tables.append(copy.deepcopy({
                'left': left_dict,
                'right': right_dict,
                'leftVisit': 0,
                'rightVisit': 0}
            ))
        self.best_split = 0
        self.visit_decay = visit_decay

    def get_best_splits(self):
        """
        Returns the best splits to grow the decision tree
        :return: Two lists with [Q,visitFreq],[Q,visitFreq]
        """

        lmax = 0
        left_visit = 0
        reward_max = 0
        right_visit = 0
        for split in self.split_q_tables:

            if split['left'] > lmax:
                lmax = split['left']
                left_visit = split['leftVisit']

            if split['right'] > reward_max:
                reward_max = split['right']
                right_visit = split['rightVisit']

        left_split = [lmax, left_visit]
        right_split = [reward_max, right_visit]

        return left_split, right_split

    def update_q_val(self, env_state, q_change, action, crit_vec):
        """
        Updates the q values for each potential side of the split table
        :param env_state: Last environment state
        :param q_change: Change in q value from bellman equation
        :param action: Last action
        :param crit_vec: Criterion vector for current decision tree node
        :return: None
        """

        for idx, dimension in enumerate(env_state):
            if dimension < crit_vec[idx]:
                # Update left side of split
                self.split_q_tables[idx]['left'][action].append(q_change)
                self.split_q_tables[idx]['leftVisit'] = self.split_q_tables[idx]['leftVisit'] * self.visit_decay \
                    + (1 - self.visit_decay)
            else:
                # Update right side of split
                self.split_q_tables[idx]['right'][action].append(q_change)
                self.split_q_tables[idx]['rightVisit'] = self.split_q_tables[idx]['rightVisit'] * self.visit_decay \
                    + (1 - self.visit_decay)
