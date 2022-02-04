"""
Tip Selection Algorithms

Tip selection functions and algorithms for standard IOTA construction and
others from the literature.
"""

from .algorithms import *


def uniform_random(nodes, links, _, tangle_env):
    """
    Uniform random selection of tips
    :param nodes: Current nodes
    :param links: Current cryptographic links
    :param tangle_env: Environment for managing data logging
    :return: Selected tips and calculation info
    """

    num_calcs = tangle_env.num_calc
    num_tips = tangle_env.num_select
    candidates = [i for i in nodes if is_tip(links, i)]
    if len(candidates) == 0:
        return [], num_calcs
    else:
        tips = []
        for i in range(num_tips):

            tips.append(np.random.choice(candidates))

        num_calcs['urts'] += num_tips
        return tips, num_calcs


def unweighted_mcmc(nodes, links, _, tangle_env):
    """
    Random walk for tip selection without transaction weight consideration.
    :param nodes: Current transaction nodes
    :param links: Current transaction cryptographic links
    :param tangle_env: Tangle environment for passing data management vars
    :return: Selected tips and calculation info
    """

    num_tips = tangle_env.num_select
    num_calcs = tangle_env.num_calc
    if len(nodes) == 0:
        return [], num_calcs

    start = nodes[0]
    tips = []
    for i in range(num_tips):
        tips.append(random_walk(links, start))

    num_calcs['uwrw'] += num_tips
    return tips, num_calcs


def weighted_mcmc(nodes, links, curr_time, tangle_env):
    """
    Weighted random walk for transaction selection
    :param nodes: Current transaction nodes
    :param links: Current cryptographic links
    :param curr_time: Current ledger construction time
    :param tangle_env: Tangle environment for managing data vars
    :return: New transaction tips and calculation recording
    """

    num_calcs = tangle_env.num_calc
    alpha = tangle_env.alpha
    time_window = tangle_env.time_window
    num_tips = tangle_env.num_select

    if len(nodes) == 0:
        return [], num_calcs

    if time_window is [0, 0]:
        # Start from genesis
        start = nodes[0]
        calc_weights(nodes, links)
        tips = []
        for i in range(num_tips):
            tips.append(weight_rand_walk(links, start, alpha))
        num_calcs['wrw'] += num_tips
    else:
        start = get_tip_window(nodes, curr_time, time_window)
        if not start:
            start = nodes[0]
        calc_weights(nodes, links, time_window, curr_time)
        tips = []
        for i in range(num_tips):
            tips.append(weight_rand_walk(links, start, alpha))
        num_calcs['wrw'] += num_tips

    return tips, num_calcs


def eiota(nodes, links, curr_time, tangle_env):
    """
    EIOTA tip selection algorithm with changing alpha from: https://arxiv.org/abs/1907.03628
    :param nodes: Current transaction nodes
    :param links: Current cryptographic links
    :param curr_time: Current ledger construction time
    :param tangle_env: Tangle environment for managing variables
    :return:
    """

    # Use 'best' parameters given in paper
    p1 = 0.1
    p2 = 0.35
    rem = 1 - (p1 + p2)
    weights = [p1, p2, rem]
    select = np.random.choice([0, 1, 2], p=weights)

    if select == 0:
        # Uniform random walk with prob 0.1
        return unweighted_mcmc(nodes, links, None, tangle_env)
    elif select == 1:
        # Walk with high alpha and prob 0.35
        high_alpha = 5
        tangle_env.alpha = high_alpha
        return weighted_mcmc(nodes, links, curr_time, tangle_env)
    else:
        # Weighted walks with random alpha value
        low_alpha = np.random.uniform(0.1, 2.0)
        tangle_env.alpha = low_alpha
        return weighted_mcmc(nodes, links, curr_time, tangle_env)


def almost_urts(nodes, links, _, tangle_env):
    """
    Almost random tip selection from: https://iota.cafe/t/almost-urts-on-a-subset/234
    :param nodes: Current transaction nodes
    :param links: Current transaction cryptographic links
    :param tangle_env: Tangle environment for managing variables
    :return:
    """

    num_calcs = tangle_env.num_calc
    candidates = [i for i in nodes if is_tip(links, i)]

    # No need for 'M' (below max depth or C1' for simple simulation)
    # Get number of non-lazy approvals for each tip
    c1 = 1.1
    c1p = 1
    c2 = 1
    m = 10

    # Dont select transactions falling behind
    rem_idx = []
    rand_rec = [(np.random.random() * 2*c1 - c1) + node['time'] for node in candidates]

    for idx, i in enumerate(candidates):
        if not ((rand_rec[idx]-c1) < i['time'] < (rand_rec[idx] + c1p)):
            if len(nodes) < 2:
                continue  # pass for genesis
            rem_idx.append(idx)
            continue

        work_link = []
        for link in links:
            if link['source']['name'] == i['name']:
                work_link.append(link)

        if len(work_link) != 0:
            max_diff = max([rand_rec[idx] - j['target']['time'] for j in work_link])
            if max_diff > m:  # Approved two lazy tips
                rem_idx.append(idx)

    for idx in reversed(rem_idx):
        candidates.pop(idx)

    # Create weights for the number of non-lazy tips that each tip approved
    cand_weights = [1] * len(candidates)
    for idx, i in enumerate(candidates):
        for link in links:
            if link['source']['name'] == i['name']:
                if i['time'] - link['target']['time'] <= c2:
                    cand_weights[idx] += 1

    if len(candidates) == 0:
        return [], num_calcs
    else:
        tips = []
        total_weight = sum(cand_weights)
        probs = [i/total_weight for i in cand_weights]
        for i in range(2):
            tips.append(np.random.choice(candidates, p=probs))
        num_calcs['urts'] += 2
        return tips, num_calcs
