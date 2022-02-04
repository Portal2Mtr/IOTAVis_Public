"""
Decision tree learner class main file

Based on Conservative Q-improvement decision tree

"""

import numpy as np
import networkx as nx
import pickle
from numpy.random import gamma
import copy
from ._dtsplitting import NodeSplits


class ConsDTLearner(object):
    """
    Decision tree learner based on: https://arxiv.org/abs/1907.01180
    """

    def __init__(self, env_state, sim_conf):
        """
        Create DT learner and setup learning parameters for bandit algorithm
        :param env_state: Initial state of the environment
        :param sim_conf: Simulation config file
        """

        # Binary tree parameters
        self.init_split_thresh = sim_conf['initThresh']
        self.hs = self.init_split_thresh
        self.split_decay = sim_conf['splitDecay']
        self.visit_decay = sim_conf['visitDecay']  # Decay for split
        self.tree = nx.Graph()
        self.steps_done = 0

        # State vectors for criterion generation
        self.min_state_vector = [0] * len(env_state)
        self.max_state_vector = [0] * len(env_state)
        self.num_env_feat = len(env_state)
        self.num_actions = sim_conf['numActions']

        # Random dimension to start decision making
        self.tree.add_node(0,
                           name="root",
                           qvalues=[[] for _ in range(self.num_actions)],
                           rewvalues=[[] for _ in range(self.num_actions)],
                           prevState=0,
                           qMax=0,
                           children=[None, None],
                           parent=None,
                           type='Leaf',
                           history=[],
                           critVec=[0.5] * self.num_env_feat,
                           criterion=[0, 0],
                           visitFreq=0,
                           splits=NodeSplits(self.visit_decay, self.num_env_feat, self.num_actions),
                           depth=0,
                           numVisits=1,
                           bandAlpha=[1] * self.num_actions,
                           bandBeta=[1] * self.num_actions,
                           bandCnt=0,
                           freqArms=[0 for _ in range(self.num_actions)])

        self.leaves = self.tree.subgraph(0)
        self.branches = self.tree.subgraph(0)

        # Q Learning parameters
        self.learn_alpha = sim_conf['learnAlpha']
        self.gamma = sim_conf['gamma']  # Bias towards next state values
        self.max_q_vals = sim_conf['maxQVals']

        # Bandit algo parameters
        self.K = sim_conf['numActions']
        self.max_tree_depth = sim_conf['maxTreeDepth']
        self.trim_visit_perc = sim_conf['trimVisitPerc']
        self.bandit_batch = sim_conf['banditbatch']
        self.bandit_cnt = 0

    from ._dtbandit import update_bandit, filter_mean, reset_bandit
    from ._dtgeneration import perform_split, check_perform_split,\
        update_leaf_branch, update_poss_splits, best_split, both_child, prune_tree

    def take_action(self, state):
        """
        Main bandit algorithm for selecting actions
        :param state: Current environment state
        :return: Action selection
        """
        # Using modified Bernoulli TS for bandit algorithm with filtered
        curr_state = self.tree.nodes[state]

        actions = []
        for i in range(self.num_actions):
            actions.append(i)
        alpha = curr_state['bandAlpha']
        beta = curr_state['bandBeta']

        # Calculate beta dist vals for each action
        pdf_vals = []
        for action in actions:
            pdf_vals.append(np.random.beta(alpha[action], beta[action]))

        selection = np.argmax(pdf_vals)
        # Update number of visits
        curr_state['numVisits'] += 1
        return selection

    def update_q_values(self, current_state, reward, action):
        """
        Updates the q values based on maximum reward for the decision tree node
        :param current_state:
        :param reward:
        :param action:
        :return:
        """

        curr_state = self.get_leaf(current_state)
        # Filter reward values with FilteredMean() approach
        templist = copy.deepcopy(self.tree.nodes[curr_state]['rewvalues'][action])
        templist.append(reward)
        self.tree.nodes[curr_state]['rewvalues'][action].append(self.filter_mean(templist))
        qvalues = self.tree.nodes[curr_state]['qvalues']
        if len(qvalues[action]) == 0:
            last_q_value = 0
            max_curr_reward = 0
        else:
            last_q_value = qvalues[action][-1]
            max_curr_reward = max(qvalues[action])

        delta = self.learn_alpha * (-last_q_value + reward + self.gamma * max_curr_reward)
        new_q = last_q_value + delta
        if len(self.tree.nodes[curr_state]['qvalues'][action]) > self.max_q_vals:
            self.tree.nodes[curr_state]['qvalues'][action].pop(0)
            self.tree.nodes[curr_state]['qvalues'][action].append(new_q)
        else:
            self.tree.nodes[curr_state]['qvalues'][action].append(new_q)

        return new_q

    def update_crits(self, input_vec):
        """
        Update the decision criterion for each branch in the tree based on new info.
        :param input_vec: Most recent leaf state
        :return:
        """
        for idx, feature in enumerate(input_vec):
            self.min_state_vector[idx] = min(self.min_state_vector[idx], feature)
            self.max_state_vector[idx] = max(self.max_state_vector[idx], feature)

        # Update the criterion for each branch node
        for node, data in self.branches.nodes(data=True):
            criterion = data['criterion']
            self.branches.nodes[node]['criterion'][1] = \
                (self.max_state_vector[criterion[0]] - self.min_state_vector[criterion[0]]) * \
                data['critVec'][criterion[0]]

    def update_visit_freq(self, leaf_state):
        """
        Updates decision tree node visit frequency for node splitting
        :param leaf_state: Node to update
        :return: None
        """

        while self.tree.nodes[leaf_state]['name'] != 'root':
            self.tree.nodes[leaf_state]['visitFreq'] = self.tree.nodes[leaf_state]['visitFreq'] *\
                self.visit_decay + (1 - self.visit_decay)
            parent = self.tree.nodes[leaf_state]['parent']
            if self.tree.nodes[parent]['children'][0] is not None:
                for child in self.tree.nodes[parent]['children']:
                    if child != leaf_state:
                        # Update sibling in binary tree
                        self.tree.nodes[child]['visitFreq'] = self.tree.nodes[child]['visitFreq'] * self.visit_decay
                        break
            leaf_state = parent

        # Update root
        self.tree.nodes[leaf_state]['visitFreq'] = self.tree.nodes[leaf_state]['visitFreq'] * \
            self.visit_decay + (1 - self.visit_decay)

    def get_leaf(self, state):
        """
        Traverses binary decision tree to find state representing current ledger.
        :param state: Environment state
        :return: Leaf index
        """

        curr_leaf = 0
        while self.tree.nodes[curr_leaf]['type'] != 'Leaf':
            curr_crit = self.tree.nodes[curr_leaf]['criterion']
            if state[curr_crit[0]] < curr_crit[1]:
                # Traverse left node to get appropriate state space
                curr_leaf = self.tree.nodes[curr_leaf]['children'][0]
            else:
                # Traverse right node to get appropriate state space
                curr_leaf = self.tree.nodes[curr_leaf]['children'][1]
        return curr_leaf

    def get_max_depth(self):
        """
        Get max depth for current tree
        :return:
        """
        max_depth = 0
        for idx, data in self.leaves.nodes(data=True):
            max_depth = max(max_depth, data['depth'])
        return max_depth

    def save_tree(self,fileNum=None):
        """
        Pickle the working decision tree
        :return:
        """

        if fileNum is not None:
            pickle.dump(self,
                        open('./intelligent_tip/output/study/dtLearner{}.p'.format(fileNum),
                             'wb'))
        else:
            pickle.dump(self,
                        open('./intelligent_tip/trainedModels/dtLearner.p',
                             'wb'))
