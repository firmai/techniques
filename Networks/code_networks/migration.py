"""
Directed network based on the three top migration destinations for each state
"""

import pandas as pd
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import dzcnapy_plotlib as dzcnapy

migrations = pd.read_csv("migration_2015.csv",
                         thousands=",").set_index("Unnamed: 0")

table_migrations = migrations.stack().reset_index()\
                                     .sort_values(0, ascending=False)\
                                     .groupby("Unnamed: 0").head(3)

table_migrations.columns = "From", "To", "weight"

G = nx.from_pandas_dataframe(table_migrations, "From", "To", 
                             edge_attr=["weight"],
                             create_using=nx.DiGraph())
nx.relabel_nodes(G, pd.read_csv("states.csv", header=None)\
                 .set_index(0)[2].to_dict(), copy=False)

print(sorted(nx.weakly_connected_components(G), key=len, reverse=True))
print(sorted(nx.strongly_connected_components(G), key=len, reverse=True))
attracting = sorted(nx.attracting_components(G), key=len, reverse=True)[0]
print(attracting)

pos = graphviz_layout(G)
dzcnapy.attrs["node_color"] = ["palegreen" if n in attracting 
                               else "pink" for n in G]

nx.draw_networkx_edges(G, pos, alpha=0.5, **dzcnapy.attrs)
nx.draw_networkx_nodes(G, pos, **dzcnapy.attrs)
nx.draw_networkx_labels(G, pos, **dzcnapy.attrs)

dzcnapy.set_extent(pos, plt)
dzcnapy.plot("migration", True)
