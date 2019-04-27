import csv
from collections import Counter
from operator import itemgetter
from itertools import chain, groupby
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import community
import matplotlib.pyplot as plt
import dzcnapy_plotlib as dzcnapy

with open("use-with.csv") as usewith_file:
    reader = csv.reader(usewith_file)
    next(reader)
    G = nx.from_edgelist((n1, n2) for _, n1, n2 in reader)

with open("products.csv") as product_file:
    reader = csv.reader(product_file)
    next(reader)

    brands = {}
    cats = {}
    star_ratings = {}

    for ppid, brand, star_rating, category in reader:
        brands[ppid] = brand
        cats[ppid] = category
        star_ratings[ppid] = float(star_rating if star_rating else 0)

# Set node attributes, based on product attributes
attributes = {"brand" : brands, "category" : cats, "star" : star_ratings}
for att_name, att_value in attributes.items():
    nx.set_node_attributes(G, att_name, att_value)

TOP_HOWMANY = 1
gccs_nodes = chain.from_iterable(sorted(nx.connected_components(G),
                                        key=len)[-TOP_HOWMANY:])
gccs = nx.subgraph(G, gccs_nodes)

for att_name in attributes:
    print("Assortativity by {}: {}"\
              .format(att_name,
                      nx.attribute_assortativity_coefficient(gccs, att_name)))
# Assortativity by category: 0.05177097453406202
# Assortativity by brand: 0.9109900760994668
# Assortativity by star: -0.003504502182675165

part = community.best_partition(gccs)
print("Modularity: {}".format(community.modularity(part, gccs)))
# Modularity: 0.854691821704231

groups = groupby(sorted(part.items(), key=itemgetter(1)), itemgetter(1))
community_labels = [list(map(itemgetter(0), group)) for _, group in groups]
subgraphs = [nx.subgraph(gccs, labels) for labels in community_labels]

induced = community.induced_graph(part, gccs)
induced.remove_edges_from(induced.selfloop_edges())

def top_cat_label(community_subgraph):
    items = [atts["category"] for _, atts
             in community_subgraph.nodes(data=True)]
    top_category = Counter(items).most_common(1)[0]
    top_label_path = top_category[0]
    return top_label_path.split(":")[-1]

mapping = {comm_id: "{}/{}".format(top_cat_label(subgraph), comm_id)
           for comm_id, subgraph in enumerate(subgraphs)}
induced = nx.relabel_nodes(induced, mapping, copy=True)

attrs = {"edge_color" : "gray", "font_size" : 12, "font_weight" : "bold",
         "node_size" : 700, "node_color" : "pink", "width" : 2,
         "font_family" : "Liberation Sans Narrow"}

# Calculate best node positions
pos = graphviz_layout(induced)

# Draw the network
nx.draw_networkx(induced, pos, **dzcnapy.attrs)

# Adjust the extents
dzcnapy.set_extent(pos, plt)

# Save and show
dzcnapy.plot("ProductNetwork")
