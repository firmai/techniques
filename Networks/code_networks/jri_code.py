"""
Case study of psychological trauma types
"""

import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.bipartite import sets, weighted_projected_graph
from networkx.drawing.nx_agraph import graphviz_layout
import scipy.spatial.distance as dist
from scipy.stats import pearsonr
import community
import generalized
import dzcnapy_plotlib as dzcnapy
import matplotlib.pyplot as plt

matrix = pd.read_csv("jri_data.csv")
print(matrix.columns, matrix.shape)

# Make a multi-index of patients+traumas
stacked = matrix.stack()
# Select the patients who _have_ traumas
edges = stacked[stacked > 0].index.tolist()
patients_traumas = nx.Graph(edges)
print(nx.is_bipartite(patients_traumas))

def similarity_mtx(biadj_mtx, similarity_f):
    """
    Convert a bi-adjacency matrix to a similarity matrix,
    based on the distance measure
    """
    similarity = [[similarity_f(biadj_mtx[x], biadj_mtx[y])
                   for x in biadj_mtx] for y in biadj_mtx]
    # Discard the main diagonal of ones
    similarity_nodiag = similarity * (1 - np.eye(biadj_mtx.shape[1]))
    similarity_df = pd.DataFrame(similarity_nodiag,
                                 index=biadj_mtx.columns,
                                 columns=biadj_mtx.columns)
    return similarity_df

def similarity_net(sim_mtx, threshold=None, density=None):
    """
    Convert a similarity to a sliced similarity network
    """
    stacked = sim_mtx.stack()
    if threshold is not None:
        stacked = stacked[stacked >= threshold]
    else:
        count = int(sim_mtx.shape[0] * (sim_mtx.shape[0] - 1) * density)
        stacked = stacked.sort_values(ascending=False)[:count]
    edges = stacked.reset_index()
    edges.columns = "source", "target", "weight"

    network = nx.from_pandas_dataframe(edges, "source", "target",
                                       edge_attr=["weight"])
    # Some nodes may be isolated; they have no incident edges
    network.add_nodes_from(sim_mtx.columns)
    return network

DENSITY = 0.35

def cosine_sim(x, y):
    return 1 - dist.cosine(x, y)

cosine_mtx = similarity_mtx(matrix, cosine_sim)
cosine_network = similarity_net(cosine_mtx, density=DENSITY)

def pearson_sim(x, y):
    return pearsonr(x, y)[0]

pearson_mtx = similarity_mtx(matrix, pearson_sim)
pearson_network = similarity_net(pearson_mtx, density=DENSITY)

# Shall we discard the statistically insignificant ties?
def pearson_sim_sign(x, y):
    r, pvalue = pearsonr(x, y)
    return r if pvalue < 0.01 else 0

pearson_mtx_sign = similarity_mtx(matrix, pearson_sim_sign)
pearson_network_sign = similarity_net(pearson_mtx_sign, density=DENSITY)

def slice_projected(net, threshold=None, density=None):
    """
    Slice a projected similarity network by threshold or density
    """
    if threshold is not None:
        weak_edges = [(n1, n2) for n1, n2, w in net.edges(data=True)
                      if w["weight"] < threshold]
    else:
        count = int(len(net) * (len(net) - 1) / 2 * density)
        weak_edges = [(n1, n2) for n1, n2, w in
                      sorted(net.edges(data=True),
                             key=lambda x: x[2]["weight"],
                             reverse=True)[count:]]
    net.remove_edges_from(weak_edges)

net1, net2 = sets(patients_traumas)
_, traumas = (net1, net2) if "WAR" in net2 else (net2, net1)
hamming_network = weighted_projected_graph(patients_traumas,
                                           traumas, ratio=True)
slice_projected(hamming_network, density=DENSITY)

net1, net2, eps, n = generalized.generalized_similarity(patients_traumas)
_, generalized_network = (net1, net2) if "WAR" in net2 else (net2, net1)
slice_projected(generalized_network, density=DENSITY)
generalized_network.remove_edges_from(generalized_network.selfloop_edges())

networks = {
    "generalized" : generalized_network,
    "pearson" : pearson_network_sign,
    "cosine" : cosine_network,
    "hamming" : hamming_network,
    }

partitions = [community.best_partition(x) for x in networks.values()]
statistics = sorted([
        (name,
         community.modularity(best_part, netw),
         len(set(best_part.values())),
         len(nx.isolates(netw))
         ) for (name, netw), best_part in zip(networks.items(), partitions)],
                    key=lambda x: x[1], reverse=True)
print(statistics)

pos = graphviz_layout(generalized_network)

for i, (name, _, _, _) in enumerate(statistics):
    net = networks[name]
    ax = plt.subplot(2, 2, i + 1)
    nx.draw_networkx_edges(net, pos, ax=ax, alpha=0.5, **dzcnapy.medium_attrs)
    nx.draw_networkx_nodes(net, pos, ax=ax, **dzcnapy.medium_attrs)
    nx.draw_networkx_labels(net, pos, ax=ax, **dzcnapy.medium_attrs)
    dzcnapy.set_extent(pos, ax, name)

dzcnapy.plot("compare_traumas")
