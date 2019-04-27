"""
Analyze "Panama Papers"
"""
import csv
import pickle
import itertools
from collections import Counter
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import dzcnapy_plotlib as dzcnapy

EDGES = "beneficiary"
NODES = (("Entities.csv", "jurisdiction", "name"),
         ("Officers.csv", "country_codes", "name"),
         ("Intermediaries.csv", "country_codes", "name"))

panama = nx.Graph()

with open("all_edges.csv") as infile:
    data = csv.DictReader(infile)
    panama.add_edges_from((link["node_1"], link["node_2"])
                          for link in data
                          if link["rel_type"].lower().startswith(EDGES))


nodes = set(panama.nodes())
relabel = {}

for f, cc, name in NODES:
    with open(f) as infile:
        kind = f.split(".")[0]
        data = csv.DictReader(infile)
        names_countries = {node["node_id"] :
                           (node[name].strip().upper(), node[cc])
                           for node in data
                           if node["node_id"] in nodes}
    names =     {nid: values[0] for nid, values in names_countries.items()}
    countries = {nid: values[1] for nid, values in names_countries.items()}
    kinds =     {nid: kind      for nid, _      in names_countries.items()}
    nx.set_node_attributes(panama, "country", countries)
    nx.set_node_attributes(panama, "kind", kinds)
    relabel.update(names)

nx.relabel_nodes(panama, relabel, copy=False)

if "ISSUES OF:" in panama:
    panama.remove_node("ISSUES OF:")

if "" in panama:
    panama.remove_node("")

print(nx.number_of_nodes(panama), nx.number_of_edges(panama))

components = [p.nodes() for p in nx.connected_component_subgraphs(panama)
              if nx.number_of_nodes(p) >= 20
              or nx.number_of_edges(p) >= 20]
panama0 = panama.subgraph(itertools.chain.from_iterable(components))

print(nx.number_of_nodes(panama0), nx.number_of_edges(panama0))

with open("panama-beneficiary.pickle", "wb") as outfile:
    pickle.dump(panama0, outfile)

cdict = {"Entities": "pink", "Officers": "blue", 
         "Intermediaries" : "green"}
c = [cdict[panama0.node[n]["kind"]] for n in panama0]
dzcnapy.small_attrs["node_color"] = c
pos = graphviz_layout(panama0)
nx.draw_networkx(panama0, pos=pos, with_labels=False, **dzcnapy.small_attrs)
dzcnapy.set_extent(pos, plt)
dzcnapy.plot("panama0")

nx.attribute_assortativity_coefficient(panama0, "kind")
nx.attribute_mixing_matrix(panama0, "kind",
                           mapping={"Entities": 0, "Officers": 1,
                                    "Intermediaries" : 2})
nx.attribute_assortativity_coefficient(panama0, "country")
nx.degree_assortativity_coefficient(panama0)

deg = nx.degree(panama0)
x, y = zip(*Counter(deg.values()).items())

plt.scatter(x, y, s=100, c="pink")
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.xlim(0.9, max(x))
plt.ylim(0.9, max(y))
plt.xlabel("Degree")
plt.ylabel("Frequency")
dzcnapy.plot("panama-beneficiaries")

top10 = sorted([(n, panama0.node[n]["kind"], v) for n, v in deg.items()],
               key=lambda x: x[2], reverse=True)[:10]
print("\n".join(["{} ({}): {}".format(*t) for t in top10]))
