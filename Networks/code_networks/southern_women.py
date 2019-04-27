"""
Explore different ways to build an event network of Southern women
"""
import re
import pandas as pd
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import scipy.spatial.distance as dist
import scipy.stats as stats
import matplotlib.pyplot as plt
import dzcnapy_plotlib as dzcnapy

pd.set_option("display.max_colwidth", 15)

G1 = nx.davis_southern_women_graph()
attendees = [pd.DataFrame({event: 1}, index=list(women.keys()))
             for event, women in G1.edge.items() if re.match("E\d+", event)]
att_mtx = pd.concat(attendees, axis=1).fillna(0).astype(int)
print(att_mtx)

sim_funcs = {
    "Hamming" :  lambda x, y:     (x == y).mean(),
    "Manhattan": lambda x, y: 1 - dist.cityblock(x, y) / att_mtx.shape[0],
    "Cosine":    lambda x, y: 1 - dist.cosine   (x, y),
    "Pearson":   lambda x, y:     stats.pearsonr(x, y)[0]
}

sim_data = [pd.DataFrame([[func(att_mtx[e1], att_mtx[e2])
                               for e1 in att_mtx] for e2 in att_mtx],
                             columns=att_mtx.columns,
                             index=att_mtx.columns)
                for func in sim_funcs.values()]

def simm2net(simm):
    stacked = simm.stack()
    sliced = stacked[stacked >= 0.7]
    net = nx.from_pandas_dataframe(sliced.reset_index(),
                                   "level_0", "level_1")
    net.add_nodes_from(att_mtx.columns)
    return net

networks = map(simm2net, sim_data)

pos = None
for i, (g, name) \
        in enumerate(zip(networks, sim_funcs.keys())):
    ax = plt.subplot(2, 2, i + 1)
    if pos is None:
        pos = graphviz_layout(g)
    nx.draw_networkx_edges(g, pos, ax=ax, **dzcnapy.thick_attrs)
    nx.draw_networkx_nodes(g, pos, ax=ax, **dzcnapy.attrs)
    nx.draw_networkx_labels(g, pos, ax=ax, **dzcnapy.attrs)
    dzcnapy.set_extent(pos, ax, name)

dzcnapy.plot("event_networks")
