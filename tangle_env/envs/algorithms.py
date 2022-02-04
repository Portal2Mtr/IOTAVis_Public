"""
General Tip Selection Algorithms

Functions used for tip selection in ./tipSelection.py.
"""

import copy
import math
import numpy as np


def is_tip(links, node):
    """
    Checks if a given node is a tip
    :param links: Current cryptographic links dictionary
    :param node: Node to check status for
    :return: Boolean if tip
    """
    for link in links:
        if link['target'] == node:
            return False
    return True


def get_approvers(links, node):
    """
    # Get cryptographic approvers for a given node
    :param links: Links to check for approvers
    :param node: Node to check for
    :return:
    """
    approvers = []
    for link in links:
        if link['target'] == node:
            approvers.append(link['source'])
    return approvers


def random_walk(links, start):
    """
    # Take an unweighted random walk through transactions
    :param links:
    :param start:
    :return:
    """
    particle = start
    while not is_tip(links, particle):
        approvers = get_approvers(links, particle)
        particle = np.random.choice(approvers)

    return particle


def weighted_choose(nodes, weights):
    """
    # Weighted choice between nodes in 'nodes' list
    :param nodes: Nodes to choose between
    :param weights: Weights for random selection
    :return:
    """

    total_weight = sum(weights)
    sample_weights = [i/total_weight for i in weights]
    selection = np.random.choice(nodes, p=sample_weights)
    return selection


def weight_rand_walk(links, start, alpha):
    """
    # Weighted random walk for MCMC and others influenced by alpha parameter
    :param links: Current transaction links
    :param start: Starting transaction in tangle
    :param alpha: Bias parameter for already confirmed transactions
    :return:
    """

    particle = start
    while not is_tip(links, particle):
        approvers = get_approvers(links, particle)
        if len(approvers) == 0:
            return particle  # Handle for early genesis case
        cum_weights = [i['cumWeight'] for i in approvers]
        max_weight = max(cum_weights)
        norm_weights = [i - max_weight for i in cum_weights]
        weights = [math.exp(alpha*i) for i in norm_weights]
        particle = weighted_choose(approvers, weights)

    return particle


def get_child_lists(nodes, links):
    """
    # Get lists of all children nodes for given nodes
    :param nodes: Nodes to get children for
    :param links: Current cryptographic links
    :return:
    """
    child_lists = dict((d['name'], []) for d in nodes)
    for link in links:
        child_lists[link['source']['name']].append(link['target'])

    return child_lists


def topological_sort(nodes, links):
    """
    # Depth-first-search based topological sorting
    :param nodes: Nodes for sorting IDs in a topological list
    :param links: Current cryptographic links
    :return: Sorted list of node ids
    """
    child_lists = get_child_lists(nodes, links)
    unvisited = nodes
    result = []

    def visit(node_id):
        if node_id not in unvisited:
            return

        for child in child_lists[node_id['name']]:
            visit(child)  # Recursive
            break

        unvisited.remove(node_id)
        result.append(node_id)
        return

    while len(unvisited) > 0:
        node = unvisited[0]
        visit(node)

    result.reverse()
    return result


def calc_weights(nodes, links, time_window=None, curr_time=None):
    """
    Calculate weights for nodes in the weighted random walk. Option for starting from time window.
    :param nodes: Nodes to calculate weights for
    :param links: Current cryptographic links
    :param time_window: Time window for starting weight calculations from
    :param curr_time: Current time of ledger construction
    """

    work_dict = dict((d['name'], d) for d in nodes)
    work_links = copy.deepcopy(links)
    sorted_nodes = topological_sort(nodes, links)
    base_weight = 1

    past_nodes = []
    if time_window is not None:
        # Reduce MCMC time window
        time_end = curr_time - time_window[1]
        time_beg = curr_time - time_window[0]
        elig_nodes = []
        for node in sorted_nodes:
            if time_beg < 0:
                elig_nodes.append(node)
                continue  # Calc weights for all nodes if early
            if node['time'] < time_beg:
                past_nodes.append(node)
                continue
            if time_beg < node['time'] < time_end:
                elig_nodes.append(node)

    if len(work_links) != 0:
        for node in sorted_nodes:
            anc_weight = 0
            idx_list = []
            for idx, link in enumerate(work_links):
                if link['target']['name'] == node['name']:
                    anc_weight += work_dict[link['source']['name']]['cumWeight']
                    idx_list.append(idx)
            for idx in sorted(idx_list, reverse=True):
                work_links.pop(idx)
            node['cumWeight'] = base_weight + anc_weight

        if time_window is not None:
            for node in past_nodes:
                node['cumWeight'] = 0


def get_tip_window(nodes, curr_time, time_window):
    """
    # Returns a tip within a given time window to start the weighted random walk from.
    :param nodes: Current nodes to check for a new starting point.
    :param curr_time: Current ledger construction time
    :param time_window: Time window for tip checking
    :return: Tip to start walk from
    """

    if curr_time - time_window[0] < 0:
        return nodes[0]
    else:
        time_beg = curr_time - time_window[0]
        time_end = curr_time - time_window[1]
        candidates = []
        for node in nodes:
            if time_beg < node['time'] < time_end:
                candidates.append(node)

        if len(candidates) == 0:
            return []
        new_node = np.random.choice(candidates)
        return new_node
