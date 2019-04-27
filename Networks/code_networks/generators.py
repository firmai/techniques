"""
Generate a variety of synthetic graphs.
"""
import re
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import dzcnapy_plotlib as dzcnapy

# Generate and draw classic networks
G0 = nx.    path_graph(20)
G1 = nx.   cycle_graph(20)
G4 = nx.    star_graph(20)
G5 = nx.complete_graph(20)
G2 = nx. balanced_tree(2, 5)
G3 = nx. grid_2d_graph(5, 4)
names = ("Linear (Path)", "Ring (Cycle)", "Balanced Tree", "Mesh (Grid)", 
         "Star", "Complete")
graphs = G0, G1, G2, G3, G4, G5
layouts = (graphviz_layout, ) * len(graphs)

for i, (g, name, layout) in  enumerate(zip(graphs, names, layouts)):
    ax = plt.subplot(3, 2, i + 1)
    pos = layout(g)
    nx.draw_networkx_edges(g, pos, alpha=0.5, ax=ax, **dzcnapy.small_attrs)
    nx.draw_networkx_nodes(g, pos, ax=ax, **dzcnapy.small_attrs)
    dzcnapy.set_extent(pos, ax, name)

dzcnapy.plot("synthetic3")

# Generate and draw random networks
G0 = nx.             erdos_renyi_graph(50,    0.05)
G1 = nx.connected_watts_strogatz_graph(50, 4, 0.5 )
G2 = nx.         barabasi_albert_graph(50, 4      )
G3 = nx.        powerlaw_cluster_graph(50, 4, 0.5 )
names = ("Erdös-Rényi (p=0.05)", "Watts-Strogatz (k=4, p=0.5)", 
         "Barabási-Albert (k=4)", "Holme-Kim (k=4, p=0.5)")
graphs = G0, G1, G2, G3
layouts = (nx.circular_layout, nx.circular_layout,
           graphviz_layout, graphviz_layout)

for i, (g, name, layout) in  enumerate(zip(graphs, names, layouts)):
    ax = plt.subplot(2, 2, i + 1)
    pos = layout(g)
    nx.draw_networkx_edges(g, pos, alpha=0.5, ax=ax, **dzcnapy.small_attrs)
    nx.draw_networkx_nodes(g, pos, ax=ax, **dzcnapy.small_attrs)
    dzcnapy.set_extent(pos, ax, name)

dzcnapy.plot("synthetic1")

# Generate and draw famous social networks
G0 = nx.karate_club_graph()
G1 = nx.davis_southern_women_graph()
G2 = nx.florentine_families_graph()
names = ("Zachary's Karate Club", "Davis Southern women", 
         "Florentine families")
graphs = G0, G1, G2
layouts = (graphviz_layout, graphviz_layout, graphviz_layout)
locations = (2, 2, 1), (2, 1, 2), (2, 2, 2)

for g, name, layout, loc in zip(graphs, names, layouts, locations):
    ax = plt.subplot(*loc)
    pos = layout(g)
    nx.draw_networkx_edges(g, pos, alpha=0.5, ax=ax, **dzcnapy.medium_attrs)
    nx.draw_networkx_nodes(g, pos, ax=ax, **dzcnapy.medium_attrs)
    nx.draw_networkx_labels(g, pos, ax=ax, **dzcnapy.medium_attrs)
    dzcnapy.set_extent(pos, ax, name)

dzcnapy.plot("synthetic2")

# We will need the network of Southern women again!
pos = graphviz_layout(G1)
nx.draw_networkx_edges(G1, pos, alpha=0.5, **dzcnapy.medium_attrs)
nx.draw_networkx_nodes(G1, nodelist=[x for x in G1 if re.match("E\d+",x)], 
                       pos=pos, **dzcnapy.medium_attrs)
dzcnapy.medium_attrs["node_color"] = "yellow"
nx.draw_networkx_nodes(G1, nodelist=[x for x in G1 if not re.match("E\d+",x)], 
                       pos=pos, **dzcnapy.medium_attrs)
nx.draw_networkx_labels(G1, pos, **dzcnapy.medium_attrs)

dzcnapy.set_extent(pos, plt)
dzcnapy.plot("southern", True)
