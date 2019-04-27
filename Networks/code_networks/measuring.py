"""
Measuring network properties.
"""
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dzcnapy_plotlib as dzcnapy

# Drawing attributes
attrs = {"edge_color" : "gray", "node_color" : "pink",
         "node_size" : 250, "width" : 2, "font_size" : 12,
         "font_family" : "Liberation Sans Narrow",
         "font_weight" : "bold",}

# Read the network, choose the central node
G = nx.read_graphml("cna.graphml")
ego = "Neighbourhood (Graph Theory)"

# Calculate two neighborhoods
alters1 = G[ego]
alters2 = list(nx.all_neighbors(G, ego))

# Extract the neighborhood subgraph and the egonet, prepare the layout
nhood = nx.subgraph(G, list(alters1.keys()) + [ego])
egonet = nx.ego_graph(G, ego)
pos = graphviz_layout(nhood)

# Locate the chord edges and remove them
chords = [(n1, n2) for n1, n2 in nhood.edges() if n1 != ego and n2 != ego]
nhood.remove_edges_from(chords)

# Draw the neighborhood and the ego-centric network
for g, ofile in zip((nhood, egonet), ("neighborhood", "egonet")):
    nx.draw_networkx_edges(g, pos, alpha=0.7, **dzcnapy.attrs)
    nx.draw_networkx_nodes(g, pos, **dzcnapy.attrs)
    nx.draw_networkx_labels(g, pos, **dzcnapy.medium_attrs)
    dzcnapy.set_extent(pos, plt)

    dzcnapy.plot(ofile)

# This part of the script calculates degrees and clustering coefficients
# and plots a scatter plot of them
F = nx.Graph(G)
deg = pd.Series(nx.degree(G))
cc = pd.Series({e: nx.clustering(F, e) for e in F})
deg_cc = pd.concat([deg, cc], axis=1)
deg_cc.columns = ("Degree", "CC")
deg_cc.groupby("Degree").mean().reset_index()\
    .plot(kind="scatter", x="Degree", y="CC", s=100)
plt.xscale("log")
plt.ylim(ymin = 0)
plt.grid()
dzcnapy.plot("deg_cc")

# A study of centralities
dgr = nx.degree_centrality(G)
clo = nx.closeness_centrality(G)
har = nx.harmonic_centrality(G)
eig = nx.eigenvector_centrality(G)
bet = nx.betweenness_centrality(G)
pgr = nx.pagerank(G)
hits = nx.hits(G)

centralities = pd.concat(
    [pd.Series(c) for c in (hits[1], eig, pgr, har, clo, hits[0], dgr, bet)],
    axis=1)

centralities.columns = ("Authorities", "Eigenvector", "PageRank",
                        "Harmonic Closeness", "Closeness", "Hubs",
                        "Degree", "Betweenness")
centralities["Harmonic Closeness"] /= centralities.shape[0]

# Calculate the correlations for each pair of centralities
c_df = centralities.corr()
ll_triangle = np.tri(c_df.shape[0], k=-1)
c_df *= ll_triangle
c_series = c_df.stack().sort_values()
c_series.tail()

X = "Harmonic Closeness"
Y = "Eigenvector"
limits = pd.concat([centralities[[X, Y]].min(),
                    centralities[[X, Y]].max()], axis=1).values
centralities.plot(kind="scatter", x=X, y=Y, xlim=limits[0], ylim=limits[1],
                  s=75, logy=True, alpha=0.6)
plt.grid()
dzcnapy.plot("eig_vs_harm")
