"""
Projection-related exercises, including the generalized similarity
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import dzcnapy_plotlib as dzcnapy
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from networkx.algorithms import bipartite
N = pickle.load(open("nutrients.pickle", "rb"))
print(bipartite.is_bipartite(N))

bip1, bip2 = bipartite.sets(N)
print("C" in bip1, "C" in bip2)

foods, nutrients = (bip2, bip1) if "C" in bip1 else (bip1, bip2)
print(foods, nutrients)

n_graph = bipartite.projected_graph(N, nutrients)
f_graph = bipartite.projected_graph(N, foods)

fw_graph = bipartite.weighted_projected_graph(N, foods, True)

# Edge width represents weights
dzcnapy.attrs["width"] = [d['weight'] * 75 for n1, n2, d in
                          fw_graph.edges(data=True)]
dzcnapy.thick_attrs["width"] = 10

pos = graphviz_layout(f_graph)
nx.draw_networkx_edges(f_graph, pos, **dzcnapy.thick_attrs)
nx.draw_networkx_nodes(f_graph, pos, **dzcnapy.attrs)
nx.draw_networkx_labels(f_graph, pos, **dzcnapy.attrs)
dzcnapy.set_extent(pos, plt)
dzcnapy.plot("projected_foods")

adj = bipartite.biadjacency_matrix(N, f_graph).toarray()
foods = pd.DataFrame([[stats.pearsonr(x, y)[0] for x in adj]
                      for y in adj], columns=f_graph, index=f_graph)

SLICING_THRESHOLD = 0.375
stacked = foods.stack()
edges = stacked[stacked >= SLICING_THRESHOLD].index.tolist()
f_pearson = nx.Graph(edges)

nx.draw_networkx_edges(f_pearson, pos, **dzcnapy.thick_attrs)
nx.draw_networkx_nodes(f_graph, pos, **dzcnapy.attrs)
nx.draw_networkx_labels(f_graph, pos, **dzcnapy.attrs)
dzcnapy.set_extent(pos, plt)
dzcnapy.plot("pearson_foods")

from generalized import generalized_similarity
bip1, bip2, eps, n_iter = generalized_similarity(N, min_eps=0.001,
                                                 max_iter=100)
foods, nutrients = (bip1, bip2) if "C" in bip2 else (bip2, bip1)
SLICING_THRESHOLD = 0.9
foods.remove_edges_from((n1, n2) for n1, n2, d in foods.edges(data=True)
                        if d['weight'] < SLICING_THRESHOLD)

nx.draw_networkx_edges(foods, pos, alpha=0.5, **dzcnapy.attrs)
nx.draw_networkx_nodes(foods, pos, **dzcnapy.attrs)
nx.draw_networkx_labels(foods, pos, **dzcnapy.attrs)
dzcnapy.set_extent(pos, plt)
dzcnapy.plot("generalized_foods")
