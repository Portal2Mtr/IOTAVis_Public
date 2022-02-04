"""
Decision tree plotting

Plotting methods for visualizing decision tree learner/learning and saving objects (Old)
"""

import pickle
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.drawing.nx_agraph import graphviz_layout


def vis_tree(self):
    """
    Use networkx to visualize generated decision tree.
    :return:
    """
    fig = plt.figure()
    args = '-Gnodesep=100 -Granksep=20 -Gpad=1 -Grankdir=TD'
    pos = graphviz_layout(self.tree, prog='dot', args=args)
    nx.draw(self.tree, pos)
    plt.title("Tree Visualization")
    plt.savefig('./adaptiveTip/outputs/saveTree.png')


def create_node_text_box(self, subgraph, ax):
    """
    Create text box for displaying node criterion and q values
    :param self: Decision tree object
    :param subgraph: Subgraph for head of decision tree
    :param ax: Matplotlib axis for displaying text
    :return: None
    """

    feature_ref = ['$\lambda$','$L_{Err}$', '$\mu$', '$\sigma$']
    box_width = 0.35
    # Just do first three nodes
    positions = [[0.15, 1], [0.1, 0.75], [0.825, 0.75]]

    for idx, pos_data in enumerate(positions):

        max_val = 0
        for arm in subgraph.nodes[idx]['qvalues']:
            max_arm = max(arm)
            max_val = max(max_arm, max_val)

        node_crit_idx = subgraph.nodes[idx]['criterion'][0]
        node_crit_val = subgraph.nodes[idx]['criterion'][1]
        text_str = '\n'.join([
            'Criterion: ' + feature_ref[node_crit_idx] + '<{:.2f}'.format(node_crit_val),
            'Q-value: '+'{:.2f}'.format(max_val),
            '# Visits: '+'{}'.format(subgraph.nodes[idx]['numVisits']),
            ])
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        text_x = pos_data[0] - box_width/2
        text_y = pos_data[1]
        ax.text(text_x, text_y, text_str, fontsize=14,
                verticalalignment='top', bbox=props, transform=ax.transAxes)


def plot_detailed_tree(self):
    """
    Plots a detailed trimmed version of the decision tree
    :param self: Decision tree object
    :return: None
    """

    tree_level = 3
    fig, ax = plt.subplots(1, 1)

    node_list = []
    for idx, node in self.tree.nodes(data=True):
        if node['depth'] < tree_level:
            node_list.append(idx)

    sub_tree = self.tree.subgraph(node_list)
    args = '-Gnodesep=20 -Granksep=20 -Gpad=1 -Grankdir=TD'
    pos = graphviz_layout(sub_tree,prog='dot',args=args)
    nx.draw(sub_tree, pos, ax=ax, with_labels=False)

    # Place a text box on axes
    self.create_node_text_box(sub_tree, ax)
    plt.savefig('./adaptiveTip/outputs/detailTree.png')


def gen_edge_labels(self):
    """
    Generate labels for showing the criterion in each node
    :param self: Decision tree object
    :return: Edge label dict
    """

    labels = {}
    feature_ref = ['$\lambda$','$Tip_{Err}$','$\mu$','$\sigma$']
    for idx,node in self.branches.nodes(data=True):
        if node['name'] == 'root':
            curr_node = 0
        else:
            curr_node = int(node['name'])

        curr_crit = node['criterion']
        left_child, right_child = node['children']
        labels[(curr_node, left_child)] = feature_ref[curr_crit[0]] + '<' + '{:.2}'.format(round(curr_crit[1], 2))
        labels[(curr_node, right_child)] = feature_ref[curr_crit[0]] + '>' + '{:.2}'.format(round(curr_crit[1], 2))

    return labels



