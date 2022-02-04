"""
Decision tree generation methods

Methods for generating decision tree, including node splitting
"""
import copy

import numpy as np
from statistics import mean
from copy import deepcopy
from ._dtsplitting import NodeSplits


def update_poss_splits(self, leaf_state, env_state, action, q_change):
    """
    Updates the possible node splits for a given leaf node (conservative q improvement
    :param self: Decision tree object
    :param leaf_state: Current node in decision tree
    :param env_state: Last environment state
    :param action: Last action chosen
    :param q_change: Q value change
    :return: None
    """

    curr_node = self.tree.nodes[leaf_state]
    crit_vals = []
    for min_val, max_val, multi in zip(self.min_state_vector, self.max_state_vector, curr_node['critVec']):
        crit_vals.append((max_val-min_val) * multi)

    # Update Q Values and visits
    self.tree.nodes[leaf_state]['splits'].update_q_val(env_state, q_change, action, crit_vals)


def best_split(self, leaf_state, action):
    """
    Finds the best split for a given tree leaf node.
    :param self: Decision tree object
    :param leaf_state: Current leaf state in tree
    :param action: Most recent action
    :return: [The best split feature, best split value]
    """

    # Get product of all visit vals from leaf to root
    curr_node = leaf_state
    visit_prod = self.tree.nodes[curr_node]['visitFreq']
    while self.tree.nodes[curr_node]['name'] != 'root':
        curr_node = self.tree.nodes[curr_node]['parent']
        visit_prod *= self.tree.nodes[curr_node]['visitFreq']

    curr_node = self.tree.nodes[leaf_state]
    sq_vals = []  # Mapping of split to split values
    for idx, split in enumerate(curr_node['splits'].split_q_tables):

        max_left = 0
        max_right = 0
        for key, value in split['left'].items():
            max_left = max(max_left, mean(value))

        for key, value in split['right'].items():
            max_right = max(max_right, mean(value))

        curr_left = max_left - abs(curr_node['qvalues'][action][-1])
        curr_right = max_right - abs(curr_node['qvalues'][action][-1])
        sq_vals.append(visit_prod * (curr_left * curr_node['splits'].split_q_tables[idx]['leftVisit'] +
                                     curr_right * curr_node['splits'].split_q_tables[idx]['rightVisit']))

    best_split_idx = np.argmax(sq_vals)
    best_val = sq_vals[best_split_idx]
    return best_split_idx, best_val


def check_perform_split(self, leaf_state, best_split_idx, best_value):
    """
    Decide tree split point based on input data distribution.
    :param self: Decision tree object
    :param leaf_state: Current leaf state
    :param best_split_idx: Best feature for splitting
    :param best_value: Best split value
    :return: None
    """

    if best_value > self.hs:
        leaf_node = self.tree.nodes[leaf_state]
        if leaf_node['depth'] >= self.max_tree_depth:
            return  # Skip leaf generation if reached max depth
        self.perform_split(leaf_state, best_split_idx)
        # Reset split threshold to slow growth
        self.hs = self.init_split_thresh

    else:
        self.hs = self.hs * self.visit_decay


def perform_split(self, leaf_state, best_split_idx):
    """
    Performs split in decision tree and creates new children nodes
    :param self: Decision tree object
    :param leaf_state: Current leaf state idx
    :param best_split_idx: Best split feature idx
    :return: None
    """
    # Create children and new branch node
    crit_val = (self.max_state_vector[best_split_idx] - self.min_state_vector[best_split_idx]) * \
        self.tree.nodes[leaf_state]['critVec'][best_split_idx]

    self.tree.nodes[leaf_state]['criterion'] = [best_split_idx, crit_val]
    self.tree.nodes[leaf_state]['type'] = 'Branch'
    left_num = max(self.tree.nodes) + 1
    right_num = max(self.tree.nodes) + 2
    self.tree.nodes[leaf_state]['children'] = [left_num, right_num]
    parent_depth = self.tree.nodes[leaf_state]['depth']
    parent_visits = self.tree.nodes[leaf_state]['numVisits']
    parent_alpha = self.tree.nodes[leaf_state]['bandAlpha']
    parent_beta = self.tree.nodes[leaf_state]['bandBeta']
    parent_rewards = self.tree.nodes[leaf_state]['rewvalues']
    split_dict = self.tree.nodes[leaf_state]['splits'].split_q_tables[best_split_idx]

    left_q_array = []
    right_q_array = []
    for key, item in split_dict['left'].items():
        left_q_array.append(item)
    for key, item in split_dict['right'].items():
        right_q_array.append(item)

    base_vec = self.tree.nodes[leaf_state]['critVec']
    left_crit_vec = deepcopy(base_vec)
    left_crit_vec[best_split_idx] += -left_crit_vec[best_split_idx] / 2
    right_crit_vec = deepcopy(base_vec)
    right_crit_vec[best_split_idx] += right_crit_vec[best_split_idx] / 2

    self.tree.add_node(left_num,
                       name=str(left_num),
                       qvalues=left_q_array,
                       rewvalues=copy.deepcopy(parent_rewards),
                       prevState=0,
                       qMax=0,
                       children=[None, None],
                       parent=leaf_state,
                       type='Leaf',
                       critVec=left_crit_vec,
                       criterion=[0, 0],
                       visitFreq=split_dict['leftVisit'],
                       splits=NodeSplits(self.visit_decay, self.num_env_feat, self.num_actions),
                       depth=parent_depth+1,
                       numVisits=parent_visits,
                       bandAlpha=parent_alpha,  # Used parent's learned experience
                       bandBeta=parent_beta,
                       bandCnt=0,
                       freqArms=[0 for _ in range(self.num_actions)])

    self.tree.add_node(right_num,
                       name=str(right_num),
                       qvalues=right_q_array,
                       rewvalues=copy.deepcopy(parent_rewards),
                       prevState=0,
                       qMax=0,
                       children=[None, None],
                       parent=leaf_state,
                       type='Leaf',
                       critVec=right_crit_vec,
                       criterion=[0, 0],
                       visitFreq=split_dict['rightVisit'],
                       splits=NodeSplits(self.visit_decay, self.num_env_feat, self.num_actions),
                       depth=parent_depth+1,
                       numVisits=parent_visits,
                       bandAlpha=parent_alpha,  # Used parent's learned experience
                       bandBeta=parent_beta,
                       bandCnt=0,
                       freqArms=[0 for _ in range(self.num_actions)])

    self.tree.add_edge(leaf_state, left_num)
    self.tree.add_edge(leaf_state, right_num)
    self.update_leaf_branch()


def update_leaf_branch(self):
    """
    Updates the leaf and branch graphs when nodes values are changed
    :param self:
    :return:
    """
    self.leaves = self.tree.subgraph([node for node, data in self.tree.nodes(data=True) if data['type'] == 'Leaf'])
    self.branches = self.tree.subgraph([node for node, data in self.tree.nodes(data=True) if data['type'] == 'Branch'])


def both_child(self, parent_node):
    """
    Checks if both children in DT are leaves
    :param self: Decision tree object
    :param parent_node: Data for current parent node of both children
    :return:
    """
    left_child, right_child = parent_node['children']
    left_true = self.tree.nodes[left_child]['type'] == 'Leaf'
    right_true = self.tree.nodes[right_child]['type'] == 'Leaf'
    return left_true and right_true


def prune_tree(self):
    """
    Prunes the decision tree to reduce tree bloat
    :param self: Decision tree object
    :return:
    """

    did_prune = True
    while did_prune:
        did_prune = False
        # Find branch node from leaves that is not visited
        for name, leaf in self.leaves.nodes(data=True):
            leaf_parent = leaf['parent']
            if leaf_parent is None:
                did_prune = False
                break
            parent_node = self.tree.nodes[leaf_parent]
            if self.both_child(parent_node):  # Check that children are leaves
                if parent_node['visitFreq'] <= self.trim_visit_perc:
                    # Remove leaf and parent nodes
                    left_child, right_child = parent_node['children']
                    self.tree.remove_node(left_child)
                    self.tree.remove_node(right_child)
                    parent_node['type'] = 'Leaf'
                    parent_node['children'] = [None, None]
                    did_prune = True
                    break
                else:
                    # Nodes have enough visits that they do not need to be trimmed
                    continue

        if did_prune:
            self.update_leaf_branch()

    return
