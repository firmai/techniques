"""
Build a "Panama" network using Pandas.
"""
import networkx as nx
import pandas as pd
import numpy as np

# Read the edge list and convert it to a network
edges = pd.read_csv("all_edges.csv")
edges = edges[edges["rel_type"] != "registered address"]
F = nx.from_pandas_dataframe(edges, "node_1", "node_2")

# Read node lists
officers = pd.read_csv("Officers.csv", index_col="node_id")
intermediaries = pd.read_csv("Intermediaries.csv", index_col="node_id")
entities = pd.read_csv("Entities.csv", index_col="node_id")

# Combine the node lists into one dataframe
officers["type"] = "officer"
intermediaries["type"] = "intermediary"
entities["type"] = "entity"

all_nodes = pd.concat([officers, intermediaries, entities])

# Do some cleanup of names
all_nodes["name"] = all_nodes["name"].str.upper().str.strip()

# Ensure that all "Bearers" do not become a single node
all_nodes["name"].replace(
    to_replace=[r"MRS?\.\s+", r"\.", r"\s+", "LIMITED", "THE BEARER",
                 "BEARER", "BEARER 1", "EL PORTADOR", "AL PORTADOR"],
    value=["", "", " ", "LTD", np.nan, np.nan, np.nan, np.nan, np.nan],
    inplace=True, regex=True)

# The network is ready to use!
# As an exercise, let's have a look at some assets
CCODES = "UZB", "TKM", "KAZ", "KGZ", "TJK"
seeds = all_nodes[all_nodes["country_codes"].isin(CCODES)].index
nodes_of_interest = set.union(*[\
        set(nx.single_source_shortest_path_length(F, seed, cutoff=2).keys())
        for seed in seeds])

# Extract the subgraph and relabel it
ego = nx.subgraph(F, nodes_of_interest)

nodes = all_nodes.ix[ego]
nodes = nodes[~nodes.index.duplicated()]
nx.set_node_attributes(ego, "cc", nodes["country_codes"])
valid_names = nodes[nodes["name"].notnull()]["name"].to_dict()
nx.relabel_nodes(ego, valid_names, copy=False)

# Save and proceed to Gephi
with open("panama-ca.graphml", "wb") as ofile:
    nx.write_graphml(ego, ofile)
