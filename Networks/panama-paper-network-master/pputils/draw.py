
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.patches as mpatches

import networkx as nx
import matplotlib.pylab as plt
import pandas as pd

from pputils.utils import get_node_label


node_types = [
    "address",
    "entities",
    "intermediates",
    "officers"
]

def build_patches(n2i, sm):
    patches = []

    for k,v in n2i.items():
        patches.append(mpatches.Patch(color=sm.to_rgba(v), label=k))

    return patches


def plot_graph(g, label_nodes=True, label_edges=False, figsize=(15,15)):
    """

    :param g:
    :return:
    """
    node_to_int = {k: node_types.index(k) for k in node_types}
    node_colours = [node_to_int[n[1]["node_type"]] for n in g.nodes(data=True)]
    node_labels = {k:get_node_label(v) for k,v in g.nodes(data=True)}

    cmap = plt.cm.rainbow
    cNorm  = colors.Normalize(vmin=0, vmax=len(node_to_int)+1)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

    plt.figure(figsize=figsize)
    plt.legend(handles=build_patches(node_to_int, scalarMap))

    pos = nx.spring_layout(g, iterations=100)

    # nodes
    nx.draw_networkx_nodes(g, pos, node_color=node_colours,
                           cmap=cmap, vmin=0, vmax=len(node_to_int)+1)

    # edges
    nx.draw_networkx_edges(g, pos, edgelist=g.edges(), arrows=True)

    # labels
    if label_nodes:
        nx.draw_networkx_labels(g, pos, labels=node_labels,
                            font_size=12, font_family='sans-serif')
    if label_edges:
        edge_labels = {(e[0], e[1]): e[2]["rel_type"] for e in g.edges(data=True)}
        nx.draw_networkx_edge_labels(g, pos, edge_labels)


