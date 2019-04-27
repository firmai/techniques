"""
Use DAG topological sort to produce a list of qualitative adjectives
"""
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import toposort
from networkx.drawing.nx_agraph import graphviz_layout
import dzcnapy_plotlib as dzcnapy

ranks = pd.read_csv("Adjectives_by_the_rank.csv",
                    header=1).set_index("ResponseID").fillna(0)
Q1 = "Rank the words from the most positive to the most negative-"
ranks = ranks.loc[:, ranks.columns.str.startswith(Q1)].astype(int)
ranks.columns = ranks.columns.str.replace(Q1, "")

dominance = pd.DataFrame([[(ranks[j] > ranks[i]).sum()
                           for i in ranks] for j in ranks],
                         columns=ranks.columns, index=ranks.columns)

QUORUM = 115
edges = sorted(dominance[dominance >= QUORUM].stack().index.tolist())
G = nx.DiGraph(edges)

# Sort in the reverse order
print(nx.topological_sort(G)[::-1])

edge_dict = {n1: set(ns) for n1, ns in nx.to_dict_of_lists(G).items()}
topo_order = list(toposort.toposort(edge_dict))
print(topo_order)

pos = graphviz_layout(G)
nx.draw_networkx_edges(G, pos, alpha=0.5, **dzcnapy.attrs)
nx.draw_networkx_nodes(G, pos, **dzcnapy.attrs)
nx.draw_networkx_labels(G, pos, **dzcnapy.attrs)

dzcnapy.set_extent(pos, plt)
dzcnapy.plot("adjectives")

