# This file produces all simple figures for chapter Cooccurrences
import networkx as nx, community, csv
import matplotlib.style as style, matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import dzcnapy_plotlib as dzcnapy

F = nx.DiGraph()
F.add_node("C")
F.add_edges_from([("B", "b0"), ("b0", "b1"), ("b1", "B")])
F.add_edges_from([("A", "a0"), ("a0", "a1"), ("a1", "a2"), ("a1", "a3"),
                  ("a3", "A")])
pos = graphviz_layout(F)
nx.draw_networkx(F, pos, **dzcnapy.attrs)
dzcnapy.set_extent(pos, plt)
dzcnapy.plot("abcNetwork")

G = nx.Graph(
    (("Alpha", "Bravo"), ("Bravo", "Charlie"), ("Charlie", "Delta"),
     ("Charlie", "Echo"), ("Charlie", "Foxtrot"), ("Delta", "Echo"),
     ("Delta", "Foxtrot"), ("Echo", "Foxtrot"), ("Echo", "Golf"), 
     ("Echo", "Hotel"), ("Foxtrot", "Golf"), ("Foxtrot", "Hotel"), 
     ("Delta", "Hotel"), ("Golf", "Hotel"), ("Delta", "India"), 
     ("Charlie", "India"), ("India", "Juliet"), ("Golf", "Kilo"), 
     ("Alpha", "Kilo"), ("Bravo", "Lima")))
pos = graphviz_layout(G)
core = nx.k_core(G)
crust = nx.k_crust(G)
corona3 = nx.k_corona(G, k=3).nodes()
nx.draw_networkx(G, pos, nodelist=core, **dzcnapy.attrs)
nx.draw_networkx_edges(G, pos, core.edges(), **dzcnapy.thick_attrs)
nx.draw_networkx_edges(G, pos, crust.edges(), **dzcnapy.thick_attrs)
nx.draw_networkx_nodes(G, pos, crust, node_shape='v', **dzcnapy.attrs)
nx.draw_networkx_nodes(G, pos, corona3, node_shape='s', **dzcnapy.attrs)
dzcnapy.set_extent(pos, plt)
dzcnapy.plot("CoresAndCoronas")

# Generate a 5-clique
G = nx.complete_graph(5, nx.Graph()) 
nx.relabel_nodes(G, 
        dict(enumerate(("Alpha", "Bravo", "Charlie", "Delta", "Echo"))), 
                 copy=False)
# Attach a pigtail to it
G.add_edges_from([
        ("Echo", "Foxtrot"), ("Foxtrot", "Golf"), ("Foxtrot", "Hotel"), 
        ("Golf", "Hotel")])
pos = graphviz_layout(G)
nx.draw_networkx(G, pos, **dzcnapy.attrs)
nx.find_cliques(G)
for g in nx.find_cliques(G):
    if len(g) > 2:
        nx.draw_networkx_edges(G, pos, nx.subgraph(G,g).edges(),
                               **dzcnapy.thick_attrs)
dzcnapy.set_extent(pos, plt)
dzcnapy.plot("MaximalClique")

G = nx.complete_graph(5, nx.Graph())
nx.relabel_nodes(G, dict(enumerate(("Alpha", "Bravo", "Charlie", 
                                    "Delta", "Echo"))), copy=False)
missing = ("Delta", "Echo")
G.remove_edge(*missing)
pos = graphviz_layout(G)
nx.draw_networkx(G, pos, **dzcnapy.attrs)
nx.draw_networkx_edges(G, pos, edgelist=(missing,), style='dashed', 
                       **dzcnapy.attrs)
dzcnapy.set_extent(pos, plt)
dzcnapy.plot("CliqueCommunity")
